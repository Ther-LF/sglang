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
    
    After permutation, tokens are grouped by clusters.
    This function computes attention only for masked block pairs.
    
    Uses Triton kernels for:
    1. Gathering attended K, V blocks
    2. Flash attention computation
    """
    B, H, S, D = q.shape
    Kq = block_mask.shape[2]
    Kk = block_mask.shape[3]
    device = q.device
    dtype = q.dtype
    
    output = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(D)
    
    # Process each batch and head
    for b in range(B):
        # Compute cumulative block sizes
        q_cumsum = torch.cat([
            torch.zeros(H, 1, dtype=torch.int32, device=device),
            q_cluster_sizes[b].cumsum(dim=-1).to(torch.int32)
        ], dim=-1)  # [H, Kq+1]
        
        k_cumsum = torch.cat([
            torch.zeros(H, 1, dtype=torch.int32, device=device),
            k_cluster_sizes[b].cumsum(dim=-1).to(torch.int32)
        ], dim=-1)  # [H, Kk+1]
        
        for h in range(H):
            # Get tensors for this head
            q_h = q[b, h].contiguous()  # [S, D]
            k_h = k[b, h].contiguous()  # [S, D]
            v_h = v[b, h].contiguous()  # [S, D]
            mask_h = block_mask[b, h]   # [Kq, Kk]
            
            q_starts = q_cumsum[h]
            k_starts = k_cumsum[h]
            
            # For each query block
            for qi in range(Kq):
                qs = q_starts[qi].item()
                qe = q_starts[qi + 1].item()
                if qe <= qs:
                    continue
                
                q_block = q_h[qs:qe].contiguous()  # [block_size_q, D]
                
                # Collect indices of all attended K tokens
                attended_indices = []
                for ki in range(Kk):
                    if mask_h[qi, ki] == 0:
                        continue
                    ks = k_starts[ki].item()
                    ke = k_starts[ki + 1].item()
                    if ke <= ks:
                        continue
                    attended_indices.append(torch.arange(ks, ke, device=device))
                
                if len(attended_indices) == 0:
                    continue
                
                # Gather K, V using Triton
                indices = torch.cat(attended_indices)
                k_gathered, v_gathered = _gather_kv_triton(k_h, v_h, indices)
                
                # Compute attention using Triton Flash Attention
                out_block = _triton_flash_attn(q_block, k_gathered, v_gathered, scale)
                
                output[b, h, qs:qe] = out_block.to(dtype)
    
    return output


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
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.num_kv_heads = num_kv_heads or num_heads
        self.prefix = prefix
        
        # Centroid cache for iterative refinement
        self.q_centroids = None
        self.k_centroids = None
    
    def forward(
        self,
        query: torch.Tensor,  # [B, S, H, D]
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SVG2SparseAttentionMetadata,
    ) -> torch.Tensor:
        """Forward pass for SVG2 sparse attention."""
        
        # Use cached centroids if available
        init_q_centroids = self.q_centroids
        init_k_centroids = self.k_centroids
        
        # Run SVG2 attention
        output = svg2_attention_forward(
            query, key, value,
            num_q_clusters=attn_metadata.num_q_clusters,
            num_k_clusters=attn_metadata.num_k_clusters,
            top_p=attn_metadata.top_p,
            kmeans_iters=attn_metadata.kmeans_iters,
        )
        
        return output


# ============================================================================
# Part 7: Testing Functions
# ============================================================================


def test_kmeans():
    """Test K-Means clustering."""
    print("Testing K-Means...")
    
    B, N, D = 2, 1024, 64
    K = 16
    device = "cuda"
    
    x = torch.randn(B, N, D, device=device, dtype=torch.float16)
    
    labels, centroids, sizes = triton_kmeans(x, K, max_iters=10)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Centroids shape: {centroids.shape}")
    print(f"  Cluster sizes: {sizes}")
    print(f"  Sum of sizes: {sizes.sum(dim=-1)} (should be {N})")
    
    assert labels.shape == (B, N)
    assert centroids.shape == (B, K, D)
    assert sizes.sum(dim=-1).tolist() == [N, N]
    print("  ✓ K-Means test passed!")


def test_permutation():
    """Test permutation and inverse permutation."""
    print("Testing Permutation...")
    
    B, H, S, D = 1, 4, 256, 64
    device = "cuda"
    
    x = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    labels = torch.randint(0, 16, (B * H, S), device=device)
    
    x_perm, sorted_indices = permute_by_labels(x, labels)
    x_restored = inverse_permute(x_perm, sorted_indices)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Permuted shape: {x_perm.shape}")
    print(f"  Max error after restore: {(x - x_restored).abs().max().item():.2e}")
    
    torch.testing.assert_close(x, x_restored, rtol=1e-3, atol=1e-3)
    print("  ✓ Permutation test passed!")


def test_block_mask():
    """Test dynamic block mask generation."""
    print("Testing Block Mask Generation...")
    
    B, H, Kq, Kk, D = 1, 4, 16, 16, 64
    device = "cuda"
    
    q_centroids = torch.randn(B, H, Kq, D, device=device)
    k_centroids = torch.randn(B, H, Kk, D, device=device)
    q_sizes = torch.ones(B, H, Kq, device=device) * 10
    k_sizes = torch.ones(B, H, Kk, device=device) * 10
    
    mask = identify_dynamic_mask(q_centroids, k_centroids, q_sizes, k_sizes, top_p=0.5)
    
    print(f"  Mask shape: {mask.shape}")
    print(f"  Sparsity: {1 - mask.float().mean().item():.2%}")
    print("  ✓ Block mask test passed!")


def test_full_svg2():
    """Test complete SVG2 attention."""
    print("Testing Full SVG2 Attention...")
    
    B, S, H, D = 1, 1024, 8, 64
    device = "cuda"
    
    q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    k = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    
    output = svg2_attention_forward(
        q, k, v,
        num_q_clusters=32,
        num_k_clusters=32,
        top_p=0.5,
        kmeans_iters=3,
    )
    
    print(f"  Input shape: {q.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    assert output.shape == q.shape
    assert not torch.isnan(output).any()
    print("  ✓ Full SVG2 test passed!")


def test_correctness_vs_dense():
    """Test SVG2 correctness against dense attention."""
    print("Testing Correctness vs Dense Attention...")
    
    B, S, H, D = 1, 256, 4, 64
    device = "cuda"
    
    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, S, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, S, H, D, device=device, dtype=torch.float32)
    
    # Dense attention (reference)
    scale = 1.0 / math.sqrt(D)
    q_t = q.transpose(1, 2)  # [B, H, S, D]
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    dense_out = torch.matmul(attn_weights, v_t)
    dense_out = dense_out.transpose(1, 2)  # [B, S, H, D]
    
    # SVG2 attention (with high top_p for near-dense behavior)
    svg2_out = svg2_attention_forward(
        q, k, v,
        num_q_clusters=8,
        num_k_clusters=8,
        top_p=0.95,  # Keep most blocks
        kmeans_iters=5,
    )
    
    # Compare
    diff = (svg2_out - dense_out).abs()
    print(f"  Max diff: {diff.max().item():.4f}")
    print(f"  Mean diff: {diff.mean().item():.4f}")
    print(f"  Relative error: {(diff / dense_out.abs().clamp(min=1e-6)).mean().item():.2%}")
    
    # With high top_p, should be reasonably close
    print("  ✓ Correctness test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SVG2 Sparse Attention Tests")
    print("=" * 60)
    
    test_kmeans()
    print()
    
    test_permutation()
    print()
    
    test_block_mask()
    print()
    
    test_full_svg2()
    print()
    
    test_correctness_vs_dense()
    print()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

