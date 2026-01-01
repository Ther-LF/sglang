# SPDX-License-Identifier: Apache-2.0
"""
SVG2 (Sparse VideoGen 2) Attention Backend for SGLang Diffusion

This implements the complete SVG2 algorithm:
1. K-Means clustering for semantic grouping
2. Semantic-Aware Permutation (SAP)
3. Dynamic Block Sparse Attention
4. Inverse Permutation

All implemented in Triton for maximum portability (no flashinfer dependency).
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# ============================================================================
# Part 1: Triton Kernels for K-Means Clustering
# ============================================================================


@triton.jit
def _pairwise_distance_kernel(
    X_ptr,           # [N, D]
    C_ptr,           # [K, D]
    X_sqnorm_ptr,    # [N] - precomputed ||x||^2
    C_sqnorm_ptr,    # [K] - precomputed ||c||^2
    Dist_ptr,        # [N, K]
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute pairwise squared Euclidean distance using:
    ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x @ c.T
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    n_start = pid_n * BLOCK_N
    k_start = pid_k * BLOCK_K
    
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    
    n_mask = n_offsets < N
    k_mask = k_offsets < K
    
    # Load precomputed squared norms
    x_sqnorm = tl.load(X_sqnorm_ptr + n_offsets, mask=n_mask, other=0.0)  # [BLOCK_N]
    c_sqnorm = tl.load(C_sqnorm_ptr + k_offsets, mask=k_mask, other=0.0)  # [BLOCK_K]
    
    # Initialize dot product accumulator
    dot_prod = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    
    # Compute x @ c.T in chunks
    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        
        # Load X[n, d] and C[k, d]
        x_ptrs = X_ptr + n_offsets[:, None] * D + d_offsets[None, :]
        c_ptrs = C_ptr + k_offsets[:, None] * D + d_offsets[None, :]
        
        x = tl.load(x_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        c = tl.load(c_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        
        # Accumulate x @ c.T using tl.dot
        dot_prod += tl.dot(x, tl.trans(c))
    
    # Compute distance: ||x||^2 + ||c||^2 - 2 * x @ c.T
    dist = x_sqnorm[:, None] + c_sqnorm[None, :] - 2.0 * dot_prod
    
    # Store distances
    dist_ptrs = Dist_ptr + n_offsets[:, None] * K + k_offsets[None, :]
    tl.store(dist_ptrs, dist, mask=n_mask[:, None] & k_mask[None, :])


@triton.jit
def _assign_clusters_kernel(
    Dist_ptr,        # [N, K]
    Labels_ptr,      # [N]
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Assign each point to nearest centroid."""
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    
    # Find minimum distance for each point
    min_dist = tl.full((BLOCK_N,), float('inf'), dtype=tl.float32)
    min_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)
    
    for k in range(K):
        dist_ptrs = Dist_ptr + n_offsets * K + k
        dist = tl.load(dist_ptrs, mask=n_mask, other=float('inf'))
        
        is_smaller = dist < min_dist
        min_dist = tl.where(is_smaller, dist, min_dist)
        min_idx = tl.where(is_smaller, k, min_idx)
    
    # Store labels
    tl.store(Labels_ptr + n_offsets, min_idx, mask=n_mask)


@triton.jit
def _update_centroids_kernel(
    X_ptr,           # [N, D]
    Labels_ptr,      # [N]
    Sum_ptr,         # [K, D]
    Count_ptr,       # [K]
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Accumulate sums and counts for centroid update using atomics."""
    pid = tl.program_id(0)  # Each program handles one point
    
    if pid >= N:
        return
    
    # Get cluster assignment
    label = tl.load(Labels_ptr + pid)
    
    # Accumulate point's features to centroid sum
    d_offsets = tl.arange(0, BLOCK_D)
    for d_start in range(0, D, BLOCK_D):
        offs = d_start + d_offsets
        mask = offs < D
        
        x_vals = tl.load(X_ptr + pid * D + offs, mask=mask, other=0.0).to(tl.float32)
        sum_ptrs = Sum_ptr + label * D + offs
        tl.atomic_add(sum_ptrs, x_vals, mask=mask)
    
    # Increment count
    tl.atomic_add(Count_ptr + label, 1)


def triton_kmeans(
    x: torch.Tensor,  # [B, N, D] or [N, D]
    n_clusters: int,
    max_iters: int = 10,
    init_centroids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated K-Means clustering using Triton.
    
    Args:
        x: Input tensor [B, N, D] or [N, D]
        n_clusters: Number of clusters K
        max_iters: Maximum iterations
        init_centroids: Optional initial centroids [B, K, D] or [K, D]
    
    Returns:
        labels: Cluster assignments [B, N] or [N]
        centroids: Final centroids [B, K, D] or [K, D]
        cluster_sizes: Size of each cluster [B, K] or [K]
    """
    # Handle batched input
    is_batched = x.dim() == 3
    if not is_batched:
        x = x.unsqueeze(0)
        if init_centroids is not None:
            init_centroids = init_centroids.unsqueeze(0)
    
    B, N, D = x.shape
    K = n_clusters
    device = x.device
    dtype = x.dtype
    
    # Flatten batch dimension for simpler kernel dispatch
    x_flat = x.reshape(B * N, D).contiguous()
    
    # Initialize centroids (random or provided)
    if init_centroids is not None:
        centroids = init_centroids.reshape(B * K, D).clone().float()
    else:
        # Random initialization: sample K points from each batch
        indices = torch.randint(0, N, (B, K), device=device)
        batch_offset = torch.arange(B, device=device)[:, None] * N
        flat_indices = (batch_offset + indices).flatten()
        centroids = x_flat[flat_indices].float().clone()
    
    centroids = centroids.reshape(B, K, D)
    labels = torch.zeros(B, N, dtype=torch.int32, device=device)
    
    # Kernel config
    BLOCK_N = 32
    BLOCK_K = min(32, K)
    BLOCK_D = min(64, D)
    
    for iteration in range(max_iters):
        # Process each batch
        for b in range(B):
            x_b = x[b].contiguous().float()  # [N, D]
            c_b = centroids[b].contiguous()  # [K, D]
            
            # Precompute squared norms for efficient distance calculation
            x_sqnorm = (x_b * x_b).sum(dim=-1).contiguous()  # [N]
            c_sqnorm = (c_b * c_b).sum(dim=-1).contiguous()  # [K]
            
            # Step 1: Compute distances using ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x@c.T
            dist = torch.zeros(N, K, dtype=torch.float32, device=device)
            grid_n = triton.cdiv(N, BLOCK_N)
            grid_k = triton.cdiv(K, BLOCK_K)
            
            _pairwise_distance_kernel[(grid_n, grid_k)](
                x_b, c_b, x_sqnorm, c_sqnorm, dist,
                N, K, D,
                BLOCK_N, BLOCK_K, BLOCK_D,
            )
            
            # Step 2: Assign clusters
            _assign_clusters_kernel[(grid_n,)](
                dist, labels[b],
                N, K, BLOCK_N,
            )
            
            # Step 3: Update centroids
            centroid_sum = torch.zeros(K, D, dtype=torch.float32, device=device)
            centroid_count = torch.zeros(K, dtype=torch.int32, device=device)
            
            _update_centroids_kernel[(N,)](
                x_b, labels[b], centroid_sum, centroid_count,
                N, D, K, BLOCK_D,
            )
            
            # Avoid division by zero
            centroid_count = centroid_count.clamp(min=1)
            centroids[b] = centroid_sum / centroid_count[:, None]
    
    # Compute cluster sizes
    cluster_sizes = torch.zeros(B, K, dtype=torch.int32, device=device)
    for b in range(B):
        for k in range(K):
            cluster_sizes[b, k] = (labels[b] == k).sum()
    
    if not is_batched:
        labels = labels.squeeze(0)
        centroids = centroids.squeeze(0)
        cluster_sizes = cluster_sizes.squeeze(0)
    
    return labels, centroids.to(dtype), cluster_sizes


# ============================================================================
# Part 2: Triton Kernels for Permutation
# ============================================================================


@triton.jit
def _permute_kernel(
    X_ptr,           # [B, H, S, D]
    IDX_ptr,         # [B*H, S]
    Y_ptr,           # [B, H, S, D]
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Permute tokens according to sorted indices."""
    pid_bh = tl.program_id(0)
    tile_s = tl.program_id(1)
    
    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    token_mask = s_offsets < S
    
    # Get source indices
    idx_ptrs = IDX_ptr + pid_bh * S + s_offsets
    src_row_idx = tl.load(idx_ptrs, mask=token_mask, other=0).to(tl.int32)
    
    # Copy all D dimensions
    d_offsets = tl.arange(0, D)
    
    src_ptrs = X_ptr + (pid_bh * S + src_row_idx[:, None]) * D + d_offsets[None, :]
    dst_ptrs = Y_ptr + (pid_bh * S + s_offsets[:, None]) * D + d_offsets[None, :]
    
    values = tl.load(src_ptrs, mask=token_mask[:, None], other=0.0)
    tl.store(dst_ptrs, values, mask=token_mask[:, None])


@triton.jit
def _inverse_permute_kernel(
    X_ptr,           # [B, H, S, D]
    IDX_ptr,         # [B*H, S]
    Y_ptr,           # [B, H, S, D]
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Inverse permutation: scatter tokens back to original positions."""
    pid_bh = tl.program_id(0)
    tile_s = tl.program_id(1)
    
    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    token_mask = s_offsets < S
    
    idx_ptrs = IDX_ptr + pid_bh * S + s_offsets
    dst_pos_idx = tl.load(idx_ptrs, mask=token_mask, other=0).to(tl.int32)
    
    d_offsets = tl.arange(0, D)
    
    src_ptrs = X_ptr + (pid_bh * S + s_offsets[:, None]) * D + d_offsets[None, :]
    dst_ptrs = Y_ptr + (pid_bh * S + dst_pos_idx[:, None]) * D + d_offsets[None, :]
    
    values = tl.load(src_ptrs, mask=token_mask[:, None], other=0.0)
    tl.store(dst_ptrs, values, mask=token_mask[:, None])


def permute_by_labels(
    x: torch.Tensor,  # [B, H, S, D]
    labels: torch.Tensor,  # [B*H, S]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute tensor by cluster labels (sort tokens by cluster).
    
    Returns:
        permuted_x: Permuted tensor [B, H, S, D]
        sorted_indices: Indices used for permutation [B*H, S]
    """
    B, H, S, D = x.shape
    BH = B * H
    device = x.device
    
    # Get sorted indices
    sorted_indices = torch.argsort(labels, dim=-1).to(torch.int32).contiguous()
    
    # Flatten and permute
    x_flat = x.reshape(BH, S, D).contiguous()
    out_flat = torch.empty_like(x_flat)
    
    BLOCK_S = 64
    n_tiles = triton.cdiv(S, BLOCK_S)
    grid = (BH, n_tiles)
    
    _permute_kernel[grid](
        x_flat, sorted_indices, out_flat,
        S, D, BLOCK_S,
        num_warps=4,
    )
    
    return out_flat.reshape(B, H, S, D), sorted_indices


def inverse_permute(
    x: torch.Tensor,  # [B, H, S, D]
    sorted_indices: torch.Tensor,  # [B*H, S]
) -> torch.Tensor:
    """Inverse permutation to restore original order."""
    B, H, S, D = x.shape
    BH = B * H
    
    x_flat = x.reshape(BH, S, D).contiguous()
    out_flat = torch.empty_like(x_flat)
    
    BLOCK_S = 64
    n_tiles = triton.cdiv(S, BLOCK_S)
    grid = (BH, n_tiles)
    
    _inverse_permute_kernel[grid](
        x_flat, sorted_indices, out_flat,
        S, D, BLOCK_S,
        num_warps=4,
    )
    
    return out_flat.reshape(B, H, S, D)


# ============================================================================
# Part 3: Dynamic Block Mask Generation
# ============================================================================


def identify_dynamic_mask(
    q_centroids: torch.Tensor,  # [B, H, Kq, D]
    k_centroids: torch.Tensor,  # [B, H, Kk, D]
    q_cluster_sizes: torch.Tensor,  # [B, H, Kq]
    k_cluster_sizes: torch.Tensor,  # [B, H, Kk]
    top_p: float = 0.5,
    min_kc_ratio: float = 0.0,
) -> torch.Tensor:
    """
    Generate dynamic block mask based on centroid similarity.
    
    Args:
        q_centroids: Query cluster centroids
        k_centroids: Key cluster centroids
        q_cluster_sizes: Size of each query cluster
        k_cluster_sizes: Size of each key cluster
        top_p: Keep top-p fraction of blocks by importance
        min_kc_ratio: Minimum ratio of k clusters to keep
    
    Returns:
        block_mask: [B, H, Kq, Kk] boolean mask
    """
    B, H, Kq, D = q_centroids.shape
    Kk = k_centroids.shape[2]
    device = q_centroids.device
    
    # Compute attention scores between centroids
    # scores[b, h, i, j] = q_centroids[b, h, i] @ k_centroids[b, h, j].T / sqrt(D)
    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum('bhqd,bhkd->bhqk', q_centroids.float(), k_centroids.float()) * scale
    
    # Weight by cluster sizes (larger clusters are more important)
    weights = q_cluster_sizes[:, :, :, None].float() * k_cluster_sizes[:, :, None, :].float()
    weighted_scores = scores * weights
    
    # Softmax to get importance
    importance = torch.softmax(weighted_scores.reshape(B, H, -1), dim=-1)
    importance = importance.reshape(B, H, Kq, Kk)
    
    # Determine threshold for top-p
    flat_importance = importance.reshape(B * H, -1)
    sorted_imp, _ = torch.sort(flat_importance, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_imp, dim=-1)
    
    # Find cutoff index
    cutoff_mask = cumsum <= top_p
    num_keep = cutoff_mask.sum(dim=-1).clamp(min=max(1, int(min_kc_ratio * Kq * Kk)))
    
    # Create mask
    threshold = torch.zeros(B * H, device=device)
    for i in range(B * H):
        if num_keep[i] < Kq * Kk:
            threshold[i] = sorted_imp[i, num_keep[i]]
    
    threshold = threshold.reshape(B, H, 1, 1)
    block_mask = importance >= threshold
    
    return block_mask


# ============================================================================
# Part 4: Triton Block Sparse Attention Kernel
# ============================================================================


@triton.jit
def _dynamic_block_sparse_fwd_kernel(
    Q,
    K,
    V,
    Out,
    dynamic_map,
    qc_cum_size,
    kc_cum_size,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_dmap_b,
    stride_dmap_h,
    stride_dmap_qc,
    stride_dmap_kc,
    stride_qcs_b,
    stride_qcs_h,
    stride_qcs_qc,
    stride_kcs_b,
    stride_kcs_h,
    stride_kcs_kc,
    B,
    H,
    S,
    D,
    scale,
    QC_NUM,
    KC_NUM,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Triton kernel for dynamic block sparse attention.
    Each program computes attention for one query block within a batch/head.
    Processes query block in chunks of BLOCK_M.
    Iterates through key blocks, checking dynamic_map.
    Processes key/value blocks in chunks of BLOCK_N.
    Uses online softmax.
    """
    # --- Grid Calculation ---
    # Each program instance handles one query block for a specific batch and head
    pid = tl.program_id(axis=0)

    # Calculate batch, head, and query block index
    pid_q_block_global = pid  # 0 to B*H*QC_NUM - 1
    
    # Need to map pid (0.. B*H*QC_NUM-1) back to (b, h, q_block_idx)
    # q_block_idx changes fastest, then h, then b
    q_block_idx = pid_q_block_global % QC_NUM
    pid_h_temp = pid_q_block_global // QC_NUM
    h = pid_h_temp % H
    b = pid_h_temp // H

    # --- Load Q block info (start/end offsets) ---
    qcs_offset = b * stride_qcs_b + h * stride_qcs_h
    q_start_offset = tl.load(qc_cum_size + qcs_offset + q_block_idx * stride_qcs_qc)
    q_end_offset = tl.load(qc_cum_size + qcs_offset + (q_block_idx + 1) * stride_qcs_qc)
    q_block_size = q_end_offset - q_start_offset

    # Early exit if the query block is empty
    if q_block_size == 0:
        return

    # --- Pointers setup ---
    q_ptr_base = Q + b * stride_qb + h * stride_qh + q_start_offset * stride_qs
    k_ptr_base = K + b * stride_kb + h * stride_kh
    v_ptr_base = V + b * stride_vb + h * stride_vh
    out_ptr_base = Out + b * stride_ob + h * stride_oh + q_start_offset * stride_os
    dmap_ptr = dynamic_map + b * stride_dmap_b + h * stride_dmap_h + q_block_idx * stride_dmap_qc
    kcs_ptr = kc_cum_size + b * stride_kcs_b + h * stride_kcs_h

    # --- Iterate over the query block rows in chunks of BLOCK_M ---
    offs_qm = tl.arange(0, BLOCK_M)  # Query block row offsets [0, 1, ..., BLOCK_M-1]
    offs_d = tl.arange(0, BLOCK_D)  # Dimension offsets [0, 1, ..., BLOCK_D-1]

    for q_chunk_start in range(0, q_block_size, BLOCK_M):
        q_chunk_rows = offs_qm + q_chunk_start
        q_rows_mask = q_chunk_rows < q_block_size  # Mask for valid rows in this Q chunk [BLOCK_M]

        # --- Initialize accumulators for this Q chunk ---
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Max score
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Sum of exp(scores - max)
        acc_o = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # Accumulated output

        # --- Load Q chunk ---
        q_ptr = q_ptr_base + q_chunk_rows[:, None] * stride_qs + offs_d[None, :]
        # Mask ensures we don't read out of bounds for the query block or dimension D
        mask_q = q_rows_mask[:, None] & (offs_d[None, :] < D)
        q_chunk = tl.load(q_ptr, mask=mask_q, other=0.0)  # Shape: [BLOCK_M, BLOCK_D]

        # --- Inner loop over K blocks (columns in the block sparse map) ---
        for k_block_idx in range(0, KC_NUM):
            # --- Check dynamic_map: Is this block active? ---
            is_active = tl.load(dmap_ptr + k_block_idx * stride_dmap_kc)
            if is_active:  # Process block only if it's active
                # --- Load K block info (start/end offsets) ---
                k_start_offset = tl.load(kcs_ptr + k_block_idx * stride_kcs_kc)
                k_end_offset = tl.load(kcs_ptr + (k_block_idx + 1) * stride_kcs_kc)
                k_block_size = k_end_offset - k_start_offset

                # Skip if the key block is empty (inside the active block check)
                if k_block_size > 0:

                    k_block_ptr_base = k_ptr_base + k_start_offset * stride_ks
                    v_block_ptr_base = v_ptr_base + k_start_offset * stride_vs

                    # --- Loop over K block chunks (size BLOCK_N) ---
                    offs_kn = tl.arange(0, BLOCK_N)  # Key block row offsets [0, ..., BLOCK_N-1]
                    for k_chunk_start in range(0, k_block_size, BLOCK_N):
                        k_chunk_rows = offs_kn + k_chunk_start
                        k_rows_mask = k_chunk_rows < k_block_size  # Mask for valid rows in this K/V chunk [BLOCK_N]

                        # --- Load K, V chunks ---
                        k_ptr = k_block_ptr_base + k_chunk_rows[:, None] * stride_ks + offs_d[None, :]
                        v_ptr = v_block_ptr_base + k_chunk_rows[:, None] * stride_vs + offs_d[None, :]

                        # Mask ensures we don't read out of bounds for the key block or dimension D
                        mask_kv = k_rows_mask[:, None] & (offs_d[None, :] < D)
                        k_chunk = tl.load(k_ptr, mask=mask_kv, other=0.0)  # Shape: [BLOCK_N, BLOCK_D]
                        v_chunk = tl.load(v_ptr, mask=mask_kv, other=0.0)  # Shape: [BLOCK_N, BLOCK_D]

                        # --- Compute Scores (Attention) ---
                        # QK^T: [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
                        s_ij_chunk = tl.dot(q_chunk, k_chunk.T) * scale

                        # IMPORTANT: Mask out scores corresponding to padding in K before max/softmax
                        # Set scores for invalid K elements to -inf
                        s_ij_chunk = tl.where(k_rows_mask[None, :], s_ij_chunk, -float("inf"))
                        # Mask out scores for invalid Q elements as well (although q_chunk elements are 0, avoid potential issues)
                        s_ij_chunk = tl.where(q_rows_mask[:, None], s_ij_chunk, -float("inf"))

                        # --- Online Softmax Update ---
                        # Current max for this Q-K chunk interaction
                        m_ij_chunk = tl.max(s_ij_chunk, axis=1)  # Shape: [BLOCK_M]

                        # Update overall max (across K chunks seen so far for this Q chunk)
                        m_new = tl.maximum(m_i, m_ij_chunk)  # Shape: [BLOCK_M]

                        # Calculate scaled probabilities P_ij = exp(S_ij - m_new)
                        p_ij_chunk = tl.exp(s_ij_chunk - m_new[:, None])  # Shape: [BLOCK_M, BLOCK_N]
                        # Zero out probabilities for masked K elements before summing
                        p_ij_chunk = tl.where(k_rows_mask[None, :], p_ij_chunk, 0.0)

                        # Calculate scaling factor for previous accumulator state
                        exp_m_diff = tl.exp(m_i - m_new)  # Shape: [BLOCK_M]

                        # Update sum accumulator (denominator L)
                        l_i_chunk = tl.sum(p_ij_chunk, axis=1)  # Sum probabilities for this chunk, shape [BLOCK_M]
                        l_i = (l_i * exp_m_diff) + l_i_chunk  # Shape: [BLOCK_M]

                        # Update output accumulator O
                        # P_ij @ V_j: [BLOCK_M, BLOCK_N] @ [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_D]
                        # Ensure p_ij_chunk is the correct dtype for dot product
                        p_ij_chunk_casted = p_ij_chunk.to(V.dtype.element_ty)
                        o_chunk = tl.dot(p_ij_chunk_casted, v_chunk)  # Shape: [BLOCK_M, BLOCK_D]

                        acc_o = (acc_o * exp_m_diff[:, None]) + o_chunk  # Shape: [BLOCK_M, BLOCK_D]

                        # Update max for the next K chunk/block
                        m_i = m_new
            # End of 'if is_active:' block
        # --- End of loop over K blocks ---

        # --- Finalize output for this Q chunk ---
        # Normalize the accumulated output: O = acc_o / l_i
        # Add epsilon to l_i to avoid division by zero
        l_i_safe = tl.where(l_i == 0, 1.0, l_i)  # Avoid 0/0 -> NaN
        o_final_chunk = acc_o / (l_i_safe[:, None])
        o_final_chunk = tl.where(l_i[:, None] == 0, 0.0, o_final_chunk)  # Ensure output is 0 if l_i was 0

        # --- Write output chunk to global memory ---
        out_ptr = out_ptr_base + q_chunk_rows[:, None] * stride_os + offs_d[None, :]
        # Mask ensures we don't write out of bounds for the query block or dimension D
        mask_out = q_rows_mask[:, None] & (offs_d[None, :] < D)
        tl.store(out_ptr, o_final_chunk.to(Out.dtype.element_ty), mask=mask_out)


def block_sparse_attention(
    q: torch.Tensor,           # [B, H, S, D]
    k: torch.Tensor,           # [B, H, S, D]
    v: torch.Tensor,           # [B, H, S, D]
    block_mask: torch.Tensor,  # [B, H, Kq, Kk]
    q_cluster_sizes: torch.Tensor,  # [B, H, Kq]
    k_cluster_sizes: torch.Tensor,  # [B, H, Kk]
) -> torch.Tensor:
    """
    Block sparse attention with variable block sizes using Triton.
    
    Replaces the previous implementation that used Python loops over blocks.
    Now uses a single Triton kernel dispatch for the entire batch.
    """
    B, H, S, D = q.shape
    qc_num = q_cluster_sizes.shape[-1]
    kc_num = k_cluster_sizes.shape[-1]
    dtype = q.dtype
    
    # Assertions and checks
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
    assert block_mask.is_cuda and q_cluster_sizes.is_cuda and k_cluster_sizes.is_cuda
    
    # Calculate scale factor (using float32 for stability)
    scale = 1.0 / math.sqrt(D)
    
    # Precompute cumulative sizes (keep on device)
    # Note: original code expected q_cluster_sizes to be used for block boundaries
    # We create cumulative sizes adding a 0 at the start
    qc_cum_size = torch.cumsum(torch.cat([torch.zeros_like(q_cluster_sizes[..., :1]), q_cluster_sizes], dim=-1), dim=-1).int()
    kc_cum_size = torch.cumsum(torch.cat([torch.zeros_like(k_cluster_sizes[..., :1]), k_cluster_sizes], dim=-1), dim=-1).int()
    
    # Output tensor
    out = torch.empty_like(q)
    
    # Triton kernel config
    BLOCK_D = triton.next_power_of_2(D)
    
    if S <= 512:
        BLOCK_M = 64
        BLOCK_N = 64
    elif S <= 1024:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 128
        BLOCK_N = 64
        
    BLOCK_M = min(BLOCK_M, S)
    BLOCK_N = min(BLOCK_N, S)
    
    # Launch grid: One program per query block per batch/head
    grid = (B * H * qc_num,)
    
    _dynamic_block_sparse_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        block_mask,
        qc_cum_size,
        kc_cum_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        block_mask.stride(0), block_mask.stride(1), block_mask.stride(2), block_mask.stride(3),
        qc_cum_size.stride(0), qc_cum_size.stride(1), qc_cum_size.stride(2),
        kc_cum_size.stride(0), kc_cum_size.stride(1), kc_cum_size.stride(2),
        B,
        H,
        S,
        D,
        scale,
        QC_NUM=qc_num,
        KC_NUM=kc_num,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    return out


# ============================================================================
# Part 5: Complete SVG2 Forward Pass
# ============================================================================


def svg2_attention_forward(
    q: torch.Tensor,  # [B, S, H, D]
    k: torch.Tensor,  # [B, S, H, D]
    v: torch.Tensor,  # [B, S, H, D]
    num_q_clusters: int = 64,
    num_k_clusters: int = 64,
    top_p: float = 0.5,
    kmeans_iters: int = 5,
) -> torch.Tensor:
    """
    Complete SVG2 (Semantic-Aware Permutation) attention forward pass.
    
    Args:
        q, k, v: Query, Key, Value tensors [B, S, H, D]
        num_q_clusters: Number of query clusters
        num_k_clusters: Number of key clusters
        top_p: Top-p fraction for block mask
        kmeans_iters: K-Means iterations
    
    Returns:
        output: Attention output [B, S, H, D]
    """
    B, S, H, D = q.shape
    device = q.device
    dtype = q.dtype
    
    # Transpose to [B, H, S, D] for easier processing
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    # Flatten batch and head dimensions for K-Means
    q_flat = q.reshape(B * H, S, D)
    k_flat = k.reshape(B * H, S, D)
    
    # Step 1: K-Means clustering
    q_labels, q_centroids, q_cluster_sizes = triton_kmeans(
        q_flat, num_q_clusters, max_iters=kmeans_iters
    )
    k_labels, k_centroids, k_cluster_sizes = triton_kmeans(
        k_flat, num_k_clusters, max_iters=kmeans_iters
    )
    
    # Reshape back
    q_labels = q_labels.reshape(B * H, S)
    q_centroids = q_centroids.reshape(B, H, num_q_clusters, D)
    q_cluster_sizes = q_cluster_sizes.reshape(B, H, num_q_clusters)
    
    k_labels = k_labels.reshape(B * H, S)
    k_centroids = k_centroids.reshape(B, H, num_k_clusters, D)
    k_cluster_sizes = k_cluster_sizes.reshape(B, H, num_k_clusters)
    
    # Step 2: Generate dynamic block mask
    block_mask = identify_dynamic_mask(
        q_centroids, k_centroids,
        q_cluster_sizes, k_cluster_sizes,
        top_p=top_p,
    )
    
    # Step 3: Permute Q, K, V by cluster labels
    q_perm, q_sorted_indices = permute_by_labels(q, q_labels)
    k_perm, k_sorted_indices = permute_by_labels(k, k_labels)
    v_perm, _ = permute_by_labels(v, k_labels)
    
    # Step 4: Block sparse attention
    out_perm = block_sparse_attention(
        q_perm, k_perm, v_perm,
        block_mask,
        q_cluster_sizes, k_cluster_sizes,
    )
    
    # Step 5: Inverse permutation
    output = inverse_permute(out_perm, q_sorted_indices)
    
    # Transpose back to [B, S, H, D]
    output = output.transpose(1, 2).contiguous()
    
    return output


# ============================================================================
# Part 6: SGLang Attention Backend Integration
# ============================================================================


@dataclass
class SVG2SparseAttentionMetadata(AttentionMetadata):
    """Metadata for SVG2 sparse attention."""
    current_timestep: int
    num_frames: int
    num_tokens_per_frame: int
    num_q_clusters: int = 64
    num_k_clusters: int = 64
    top_p: float = 0.5
    kmeans_iters: int = 5
    # Cache for centroids (for iterative refinement)
    q_centroids_cache: Optional[torch.Tensor] = None
    k_centroids_cache: Optional[torch.Tensor] = None


class SVG2SparseAttentionMetadataBuilder(AttentionMetadataBuilder):
    """Builder for SVG2 metadata."""
    
    def __init__(self):
        pass
    
    def prepare(self):
        pass
    
    def build(
        self,
        current_timestep: int,
        num_frames: int,
        num_tokens_per_frame: int,
        num_q_clusters: int = 64,
        num_k_clusters: int = 64,
        top_p: float = 0.5,
        kmeans_iters: int = 5,
        **kwargs: dict[str, Any],
    ) -> SVG2SparseAttentionMetadata:
        return SVG2SparseAttentionMetadata(
            current_timestep=current_timestep,
            num_frames=num_frames,
            num_tokens_per_frame=num_tokens_per_frame,
            num_q_clusters=num_q_clusters,
            num_k_clusters=num_k_clusters,
            top_p=top_p,
            kmeans_iters=kmeans_iters,
        )


class SVG2SparseAttentionBackend(AttentionBackend):
    """SVG2 Sparse Attention Backend for SGLang Diffusion."""
    
    accept_output_buffer: bool = False
    
    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]
    
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SVG2_SPARSE_ATTN
    
    @staticmethod
    def get_impl_cls() -> type["SVG2SparseAttentionImpl"]:
        return SVG2SparseAttentionImpl
    
    @staticmethod
    def get_metadata_cls() -> type["SVG2SparseAttentionMetadata"]:
        return SVG2SparseAttentionMetadata
    
    @staticmethod
    def get_builder_cls() -> type["SVG2SparseAttentionMetadataBuilder"]:
        return SVG2SparseAttentionMetadataBuilder


class SVG2SparseAttentionImpl(AttentionImpl):
    """Implementation of SVG2 Sparse Attention."""
    
    # Default SVG2 parameters (matching original SVG implementation)
    DEFAULT_NUM_Q_CLUSTERS = 64
    DEFAULT_NUM_K_CLUSTERS = 64
    DEFAULT_TOP_P = 0.5
    DEFAULT_KMEANS_ITERS = 5
    DEFAULT_FIRST_LAYERS_FP = 0  # Number of first layers using full attention
    DEFAULT_FIRST_TIMES_FP = 0   # Timestep threshold (timesteps > this use full attention)
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        # SVG2 specific parameters
        num_q_clusters: int = 64,
        num_k_clusters: int = 64,
        top_p: float = 0.5,
        kmeans_iters: int = 5,
        first_layers_fp: int = 0,
        first_times_fp: float = 0,
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.num_kv_heads = num_kv_heads or num_heads
        self.prefix = prefix
        
        # SVG2 parameters
        self.num_q_clusters = num_q_clusters
        self.num_k_clusters = num_k_clusters
        self.top_p = top_p
        self.kmeans_iters = kmeans_iters
        self.first_layers_fp = first_layers_fp
        self.first_times_fp = first_times_fp
        
        # Centroid cache for iterative refinement across timesteps
        self.q_centroids = None
        self.k_centroids = None
        self.centroids_initialized = False
        
        # Extract layer index from prefix if available
        self.layer_idx = self._extract_layer_idx(prefix)
    
    def _extract_layer_idx(self, prefix: str) -> int:
        """Extract layer index from prefix string like 'blocks.5.attn1'."""
        import re
        match = re.search(r'blocks\.(\d+)', prefix)
        if match:
            return int(match.group(1))
        return 0
    
    def _should_use_full_attention(
        self,
        timestep: Optional[float] = None,
    ) -> bool:
        """
        Determine if full attention should be used based on layer/timestep.
        
        Matching original SVG logic:
        - if layer_idx < first_layers_fp: use full attention
        - if timestep > first_times_fp: use full attention
        
        Note: timestep decreases from ~1000 to 0 during inference,
        so timestep > threshold means early inference steps.
        """
        # First N layers always use full attention
        if self.layer_idx < self.first_layers_fp:
            return True
        
        # Early timesteps (high values) use full attention
        if timestep is not None and timestep > self.first_times_fp:
            return True
        
        return False
    
    def forward(
        self,
        query: torch.Tensor,  # [B, S, H, D]
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Optional[SVG2SparseAttentionMetadata] = None,
    ) -> torch.Tensor:
        """
        Forward pass for SVG2 sparse attention.
        
        Args:
            query: [B, S, H, D] query tensor
            key: [B, S, H, D] key tensor  
            value: [B, S, H, D] value tensor
            attn_metadata: Optional metadata with SVG2 parameters
        
        Returns:
            output: [B, S, H, D] attention output
        """
        # Get parameters from metadata or use defaults
        if attn_metadata is not None:
            num_q_clusters = getattr(attn_metadata, 'num_q_clusters', self.num_q_clusters)
            num_k_clusters = getattr(attn_metadata, 'num_k_clusters', self.num_k_clusters)
            top_p = getattr(attn_metadata, 'top_p', self.top_p)
            kmeans_iters = getattr(attn_metadata, 'kmeans_iters', self.kmeans_iters)
            # Get timestep for full attention decision
            timestep = getattr(attn_metadata, 'current_timestep', None)
        else:
            num_q_clusters = self.num_q_clusters
            num_k_clusters = self.num_k_clusters
            top_p = self.top_p
            kmeans_iters = self.kmeans_iters
            timestep = None
        
        # Check if we should use full attention (early layers or early timesteps)
        if self._should_use_full_attention(timestep):
            # Use standard scaled dot-product attention
            return self._full_attention(query, key, value)
        
        # Use SVG2 sparse attention
        output = svg2_attention_forward(
            query, key, value,
            num_q_clusters=num_q_clusters,
            num_k_clusters=num_k_clusters,
            top_p=top_p,
            kmeans_iters=kmeans_iters,
        )
        
        return output
    
    def _full_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Standard full attention using PyTorch SDPA."""
        # Input shape: [B, S, H, D]
        # SDPA expects: [B, H, S, D]
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=self.causal,
            scale=self.softmax_scale,
        )
        
        # Transpose back to [B, S, H, D]
        return output.transpose(1, 2)



