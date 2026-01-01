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
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sm_scale,
    M,  # query length
    N,  # key length  
    D: tl.constexpr,
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_om, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Flash attention forward kernel for a single Q block."""
    pid_m = tl.program_id(0)
    
    # Compute offsets
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)
    
    m_mask = m_offs < M
    
    # Load Q block
    q_ptrs = Q_ptr + m_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_mask[:, None] & (d_offs[None, :] < D), other=0.0).to(tl.float32)
    
    # Initialize accumulators for online softmax
    o_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    l_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_acc = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    
    # Iterate over K/V blocks
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        
        # Load K, V
        k_ptrs = K_ptr + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd
        v_ptrs = V_ptr + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=n_mask[:, None] & (d_offs[None, :] < D), other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=n_mask[:, None] & (d_offs[None, :] < D), other=0.0).to(tl.float32)
        
        # Compute attention scores: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(m_mask[:, None] & n_mask[None, :], s, float('-inf'))
        
        # Online softmax update
        m_new = tl.maximum(m_acc, tl.max(s, axis=1))
        alpha = tl.exp(m_acc - m_new)
        p = tl.exp(s - m_new[:, None])
        
        l_acc = alpha * l_acc + tl.sum(p, axis=1)
        # Both p and v must have same dtype for tl.dot
        o_acc = alpha[:, None] * o_acc + tl.dot(p.to(v.dtype), v)
        m_acc = m_new
    
    # Normalize output
    o_acc = o_acc / (l_acc[:, None] + 1e-6)
    
    # Store output
    o_ptrs = O_ptr + m_offs[:, None] * stride_om + d_offs[None, :] * stride_od
    tl.store(o_ptrs, o_acc.to(Q_ptr.dtype.element_ty), mask=m_mask[:, None] & (d_offs[None, :] < D))


def _triton_flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Triton Flash Attention for a single head.
    
    Args:
        q: [M, D] query
        k: [N, D] key
        v: [N, D] value
        scale: attention scale
    
    Returns:
        output: [M, D]
    """
    M, D = q.shape
    N = k.shape[0]
    
    output = torch.empty_like(q)
    
    # Choose block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    
    grid = (triton.cdiv(M, BLOCK_M),)
    
    _flash_attn_fwd_kernel[grid](
        q, k, v, output,
        scale,
        M, N, D,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_D,
        num_warps=4,
    )
    
    return output


@triton.jit  
def _gather_kv_kernel(
    K_ptr, V_ptr,
    K_out_ptr, V_out_ptr,
    Indices_ptr,  # [total_gathered]
    total_gathered,
    D: tl.constexpr,
    stride_k_s, stride_k_d,
    stride_v_s, stride_v_d,
    stride_ko_s, stride_ko_d,
    stride_vo_s, stride_vo_d,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather K, V by indices."""
    pid_s = tl.program_id(0)
    
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)
    
    s_mask = s_offs < total_gathered
    
    # Load indices
    indices = tl.load(Indices_ptr + s_offs, mask=s_mask, other=0)
    
    # Gather K
    k_ptrs = K_ptr + indices[:, None] * stride_k_s + d_offs[None, :] * stride_k_d
    k_vals = tl.load(k_ptrs, mask=s_mask[:, None] & (d_offs[None, :] < D), other=0.0)
    
    ko_ptrs = K_out_ptr + s_offs[:, None] * stride_ko_s + d_offs[None, :] * stride_ko_d
    tl.store(ko_ptrs, k_vals, mask=s_mask[:, None] & (d_offs[None, :] < D))
    
    # Gather V
    v_ptrs = V_ptr + indices[:, None] * stride_v_s + d_offs[None, :] * stride_v_d
    v_vals = tl.load(v_ptrs, mask=s_mask[:, None] & (d_offs[None, :] < D), other=0.0)
    
    vo_ptrs = V_out_ptr + s_offs[:, None] * stride_vo_s + d_offs[None, :] * stride_vo_d
    tl.store(vo_ptrs, v_vals, mask=s_mask[:, None] & (d_offs[None, :] < D))


def _gather_kv_triton(k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather K, V using Triton kernel."""
    total_gathered = indices.shape[0]
    D = k.shape[1]
    
    k_out = torch.empty(total_gathered, D, dtype=k.dtype, device=k.device)
    v_out = torch.empty(total_gathered, D, dtype=v.dtype, device=v.device)
    
    BLOCK_S = 64
    BLOCK_D = triton.next_power_of_2(D)
    
    grid = (triton.cdiv(total_gathered, BLOCK_S),)
    
    _gather_kv_kernel[grid](
        k, v, k_out, v_out,
        indices.to(torch.int32),
        total_gathered, D,
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        k_out.stride(0), k_out.stride(1),
        v_out.stride(0), v_out.stride(1),
        BLOCK_S, BLOCK_D,
        num_warps=4,
    )
    
    return k_out, v_out


def _compute_block_boundaries(cluster_sizes: torch.Tensor) -> torch.Tensor:
    """
    Compute block start indices from cluster sizes.
    
    Args:
        cluster_sizes: [BH, K] cluster sizes
    
    Returns:
        block_starts: [BH, K+1] cumulative start indices
    """
    BH, K = cluster_sizes.shape
    device = cluster_sizes.device
    block_starts = torch.zeros(BH, K + 1, dtype=torch.int64, device=device)
    block_starts[:, 1:] = cluster_sizes.to(torch.int64).cumsum(dim=-1)
    return block_starts


@triton.jit
def _gather_selected_kv_kernel(
    # Input tensors
    K_ptr, V_ptr,
    # Output tensors (gathered)
    K_out_ptr, V_out_ptr,
    # Indices
    K_starts_ptr,  # [Kk+1] block boundaries for this bh
    selected_blocks_ptr,  # [num_selected] indices of selected blocks
    # Dimensions
    num_selected,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Gather selected key/value blocks into contiguous memory.
    Grid: (num_tokens_to_gather,)
    """
    token_idx = tl.program_id(0)
    
    # Find which selected block this token belongs to
    cumsum = 0
    block_local_idx = 0
    selected_block_idx = 0
    
    for i in range(num_selected):
        ki = tl.load(selected_blocks_ptr + i)
        k_start = tl.load(K_starts_ptr + ki)
        k_end = tl.load(K_starts_ptr + ki + 1)
        block_len = k_end - k_start
        
        if token_idx >= cumsum and token_idx < cumsum + block_len:
            selected_block_idx = i
            block_local_idx = token_idx - cumsum
            # Source index in original K/V
            src_idx = k_start + block_local_idx
            
            # Copy K
            d_range = tl.arange(0, BLOCK_D)
            mask = d_range < D
            k_val = tl.load(K_ptr + src_idx * D + d_range, mask=mask, other=0.0)
            tl.store(K_out_ptr + token_idx * D + d_range, k_val, mask=mask)
            
            # Copy V
            v_val = tl.load(V_ptr + src_idx * D + d_range, mask=mask, other=0.0)
            tl.store(V_out_ptr + token_idx * D + d_range, v_val, mask=mask)
            return
        
        cumsum += block_len


@triton.jit
def _flash_attn_fwd_inner_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Dimensions
    M, N, D: tl.constexpr,  # M = query length, N = key length
    stride_qm, stride_kn, stride_vn, stride_om,
    # Scale
    scale,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Standard FlashAttention forward kernel for a single (batch, head) pair.
    Used after gathering selected K/V tokens.
    
    Grid: (ceil(M / BLOCK_M),)
    """
    pid_m = tl.program_id(0)
    
    # Query block range
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M
    
    # Load query block [BLOCK_M, D]
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D
    
    q_ptrs = Q_ptr + m_offsets[:, None] * stride_qm + d_range[None, :]
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    
    # Initialize online softmax accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Iterate over key blocks
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        
        # Load key block [BLOCK_N, D]
        k_ptrs = K_ptr + n_offsets[:, None] * stride_kn + d_range[None, :]
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        
        # Compute attention scores [BLOCK_M, BLOCK_N]
        # q: [BLOCK_M, D], k: [BLOCK_N, D] -> scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # Mask out invalid positions
        scores = tl.where(
            m_mask[:, None] & n_mask[None, :],
            scores,
            float('-inf')
        )
        
        # Online softmax update
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        
        l_ij = tl.sum(p, axis=1)
        l_new = alpha * l_i + l_ij
        
        # Load value block [BLOCK_N, D]
        v_ptrs = V_ptr + n_offsets[:, None] * stride_vn + d_range[None, :]
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        
        # Update output accumulator
        o_i = alpha[:, None] * o_i + tl.dot(p.to(v.dtype), v)
        
        m_i = m_new
        l_i = l_new
    
    # Normalize
    o_i = o_i / l_i[:, None]
    
    # Store output
    o_ptrs = O_ptr + m_offsets[:, None] * stride_om + d_range[None, :]
    tl.store(o_ptrs, o_i.to(tl.float16), mask=m_mask[:, None] & d_mask[None, :])


def _triton_block_attention(
    q_block: torch.Tensor,  # [q_len, D]
    k_selected: torch.Tensor,  # [k_len, D]
    v_selected: torch.Tensor,  # [k_len, D]
    scale: float,
) -> torch.Tensor:
    """
    Compute attention for one query block with selected K/V using Triton.
    """
    M, D = q_block.shape
    N = k_selected.shape[0]
    
    if M == 0 or N == 0:
        return torch.zeros_like(q_block)
    
    output = torch.zeros_like(q_block)
    
    BLOCK_M = min(64, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_D = triton.next_power_of_2(D)
    
    grid = (triton.cdiv(M, BLOCK_M),)
    
    _flash_attn_fwd_inner_kernel[grid](
        q_block, k_selected, v_selected, output,
        M, N, D,
        D,  # stride_qm
        D,  # stride_kn
        D,  # stride_vn
        D,  # stride_om
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    return output


def block_sparse_attention(
    q: torch.Tensor,           # [B, H, S, D]
    k: torch.Tensor,           # [B, H, S, D]
    v: torch.Tensor,           # [B, H, S, D]
    block_mask: torch.Tensor,  # [B, H, Kq, Kk]
    q_cluster_sizes: torch.Tensor,  # [B, H, Kq]
    k_cluster_sizes: torch.Tensor,  # [B, H, Kk]
) -> torch.Tensor:
    """
    Block sparse attention with variable block sizes.
    
    Strategy: For each query block, gather selected K/V into contiguous memory,
    then use a Triton FlashAttention kernel. 
    
    Performance notes:
    - Still has O(BH * Kq) Python iterations, but each iteration is fast
    - GPU-CPU sync happens once per BH (not per iteration)
    - Triton FlashAttention kernel for actual attention computation
    - True O(1) sparse attention requires flashinfer integration
    
    Memory: O(max_q_block * selected_k_tokens) instead of O(S^2)
    """
    B, H, S, D = q.shape
    device = q.device
    dtype = q.dtype
    scale = 1.0 / math.sqrt(D)
    
    Kq = q_cluster_sizes.shape[-1]
    Kk = k_cluster_sizes.shape[-1]
    
    # Reshape for processing
    q_flat = q.reshape(B * H, S, D).contiguous()
    k_flat = k.reshape(B * H, S, D).contiguous()
    v_flat = v.reshape(B * H, S, D).contiguous()
    block_mask_flat = block_mask.reshape(B * H, Kq, Kk)
    q_sizes_flat = q_cluster_sizes.reshape(B * H, Kq).to(torch.int64)
    k_sizes_flat = k_cluster_sizes.reshape(B * H, Kk).to(torch.int64)
    
    # Compute block boundaries on GPU
    q_starts = _compute_block_boundaries(q_sizes_flat)  # [BH, Kq+1]
    k_starts = _compute_block_boundaries(k_sizes_flat)  # [BH, Kk+1]
    
    # ========== KEY OPTIMIZATION: Single GPU->CPU sync for all boundaries ==========
    # Transfer all block boundaries to CPU at once (one sync instead of many)
    q_starts_cpu = q_starts.cpu().numpy()  # [BH, Kq+1]
    k_starts_cpu = k_starts.cpu().numpy()  # [BH, Kk+1]
    # Also transfer mask to CPU for fast iteration
    mask_cpu = block_mask_flat.cpu().numpy()  # [BH, Kq, Kk]
    
    # Output tensor
    output = torch.zeros_like(q_flat)
    
    BH = B * H
    
    # Process each (batch, head) 
    for bh in range(BH):
        q_bh = q_flat[bh]  # [S, D]
        k_bh = k_flat[bh]  # [S, D]
        v_bh = v_flat[bh]  # [S, D]
        
        # Use pre-transferred CPU arrays (no sync here)
        q_starts_bh = q_starts_cpu[bh]  # [Kq+1]
        k_starts_bh = k_starts_cpu[bh]  # [Kk+1]
        mask_bh = mask_cpu[bh]  # [Kq, Kk]
        
        for qi in range(Kq):
            q_start = int(q_starts_bh[qi])
            q_end = int(q_starts_bh[qi + 1])
            if q_start >= q_end:
                continue
            
            q_block = q_bh[q_start:q_end]  # [q_size, D]
            
            # Find selected key blocks from CPU mask
            k_block_indices = mask_bh[qi].nonzero()[0]
            
            if len(k_block_indices) == 0:
                continue
            
            # Build k indices list (CPU operations, fast)
            k_indices_list = []
            for ki in k_block_indices:
                k_start = int(k_starts_bh[ki])
                k_end = int(k_starts_bh[ki + 1])
                if k_start < k_end:
                    k_indices_list.extend(range(k_start, k_end))
            
            if len(k_indices_list) == 0:
                continue
            
            # Single GPU operation to gather K/V
            k_indices = torch.tensor(k_indices_list, device=device, dtype=torch.long)
            k_selected = k_bh[k_indices]  # [total_k, D]
            v_selected = v_bh[k_indices]  # [total_k, D]
            
            # Use Triton FlashAttention for this block
            out_block = _triton_block_attention(q_block, k_selected, v_selected, scale)
            
            output[bh, q_start:q_end] = out_block
    
    return output.reshape(B, H, S, D)


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



