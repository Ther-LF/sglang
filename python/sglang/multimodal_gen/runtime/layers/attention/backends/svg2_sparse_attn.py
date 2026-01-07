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
    Labels_ptr,      # [N] - int64
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
    min_idx = tl.zeros((BLOCK_N,), dtype=tl.int64)
    
    for k in range(K):
        dist_ptrs = Dist_ptr + n_offsets * K + k
        dist = tl.load(dist_ptrs, mask=n_mask, other=float('inf'))
        
        is_smaller = dist < min_dist
        min_dist = tl.where(is_smaller, dist, min_dist)
        min_idx = tl.where(is_smaller, tl.cast(k, tl.int64), min_idx)
    
    # Store labels
    tl.store(Labels_ptr + n_offsets, min_idx, mask=n_mask)


def _update_centroids_pytorch(
    x: torch.Tensor,       # [B, N, D]
    labels: torch.Tensor,  # [B, N] int64
    K: int,
    old_centroids: torch.Tensor,  # [B, K, D] for empty cluster fallback
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic centroid update using PyTorch scatter_add.
    
    This matches SVG's triton_centroid_update_sorted_euclid behavior:
    - Uses scatter_add for deterministic accumulation
    - Returns centroids in original dtype
    - Handles empty clusters by keeping old centroids
    
    Returns:
        new_centroids: [B, K, D]
        cluster_sizes: [B, K] int64
    """
    B, N, D = x.shape
    device = x.device
    dtype = x.dtype
    
    # Compute cluster sizes using bincount (deterministic)
    cluster_sizes = torch.zeros(B, K, dtype=torch.int64, device=device)
    for b in range(B):
        counts = torch.bincount(labels[b].long(), minlength=K)
        cluster_sizes[b] = counts
    
    # Accumulate sums using scatter_add (deterministic)
    # Expand labels to [B, N, D] for scatter
    labels_expanded = labels.unsqueeze(-1).expand(-1, -1, D).long()  # [B, N, D]
    
    # Initialize sum buffer
    centroid_sums = torch.zeros(B, K, D, dtype=torch.float32, device=device)
    
    # Scatter add features to their cluster
    centroid_sums.scatter_add_(1, labels_expanded, x.float())
    
    # Compute means, handle empty clusters
    counts_safe = cluster_sizes.clamp(min=1).unsqueeze(-1).float()  # [B, K, 1]
    new_centroids = centroid_sums / counts_safe
    
    # For empty clusters, keep old centroids
    empty_mask = (cluster_sizes == 0).unsqueeze(-1)  # [B, K, 1]
    new_centroids = torch.where(empty_mask, old_centroids.float(), new_centroids)
    
    return new_centroids.to(dtype), cluster_sizes


def triton_kmeans(
    x: torch.Tensor,  # [B, N, D] or [N, D]
    n_clusters: int,
    max_iters: int = 10,
    init_centroids: Optional[torch.Tensor] = None,
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated K-Means clustering matching SVG's batch_kmeans_Euclid.
    
    This implementation uses:
    1. Triton kernels for distance computation and cluster assignment
    2. PyTorch scatter_add for deterministic centroid updates (matching SVG)
    
    Args:
        x: Input tensor [B, N, D] or [N, D]
        n_clusters: Number of clusters K
        max_iters: Maximum iterations
        init_centroids: Optional initial centroids [B, K, D] or [K, D]
        tol: Convergence tolerance for center movement
    
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
    
    # DEBUG: Print K-Means parameters (only once per forward call)
    logger.debug(f"[SVG2 Debug] K-Means: N={N}, K={K}, iters={max_iters}, init={'reuse' if init_centroids is not None else 'random'}")
    
    # Pre-compute squared L2 norm of all points (constant during iterations)
    # This matches SVG's approach
    x_sq = (x.float() ** 2).sum(dim=-1)  # [B, N]
    
    # Initialize centroids (matching SVG's initialization)
    if init_centroids is not None:
        centroids = init_centroids.clone().float()
    else:
        # Randomly select initial centers from x (same as SVG)
        indices = torch.randint(0, N, (B, K), device=device)
        centroids = torch.gather(
            x.float(), dim=1, 
            index=indices.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, K, D]
    
    centroids = centroids.view(B, K, D).float()
    labels = torch.zeros(B, N, dtype=torch.int64, device=device)
    
    # Pre-allocate distance buffer
    dist_buffer = torch.empty(N, K, dtype=torch.float32, device=device)
    
    # Kernel config for distance computation
    BLOCK_N = 128
    BLOCK_K = min(128, K)
    BLOCK_D = min(128, triton.next_power_of_2(D))
    
    grid_n = triton.cdiv(N, BLOCK_N)
    grid_k = triton.cdiv(K, BLOCK_K)
    
    for iteration in range(max_iters):
        # Process each batch
        for b in range(B):
            x_b = x[b].float()  # [N, D]
            c_b = centroids[b]  # [K, D]
            x_sq_b = x_sq[b]    # [N]
            
            # Precompute centroid squared norms
            c_sq_b = (c_b ** 2).sum(dim=-1)  # [K]
            
            # Step 1: Compute distances using Triton
            _pairwise_distance_kernel[(grid_n, grid_k)](
                x_b, c_b, x_sq_b, c_sq_b, dist_buffer,
                N, K, D,
                BLOCK_N, BLOCK_K, BLOCK_D,
            )
            
            # Step 2: Assign clusters using Triton
            _assign_clusters_kernel[(grid_n,)](
                dist_buffer, labels[b],
                N, K, BLOCK_N,
            )
        
        # Step 3: Update centroids using deterministic PyTorch scatter_add
        # This matches SVG's triton_centroid_update_sorted_euclid behavior
        new_centroids, cluster_sizes = _update_centroids_pytorch(
            x.float(), labels, K, centroids
        )
        
        # Check for convergence (matching SVG's logic)
        center_shift = (new_centroids - centroids).norm(dim=-1).max()
        centroids = new_centroids
        
        if center_shift < tol:
            break
    
    # Convert labels to int32 for compatibility
    labels = labels.to(torch.int32)
    cluster_sizes = cluster_sizes.to(torch.int32)
    
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
    labels: Optional[torch.Tensor] = None,  # [B*H, S]
    sorted_indices: Optional[torch.Tensor] = None,  # [B*H, S]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute tensor by cluster labels (sort tokens by cluster).
    Optionally accepts pre-computed sorted_indices to avoid re-sorting.
    
    Returns:
        permuted_x: Permuted tensor [B, H, S, D]
        sorted_indices: Indices used for permutation [B*H, S]
    """
    B, H, S, D = x.shape
    BH = B * H
    device = x.device
    
    # Get sorted indices
    if sorted_indices is None:
        assert labels is not None, "Either labels or sorted_indices must be provided"
        sorted_indices = torch.argsort(labels, dim=-1).to(torch.int32).contiguous()
    else:
        sorted_indices = sorted_indices.to(torch.int32).contiguous()
    
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
    max_k_clusters_per_q: Optional[int] = None,
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
        max_k_clusters_per_q: Optional cap of kept key clusters per query cluster
    
    Returns:
        block_mask: [B, H, Kq, Kk] boolean mask
    """
    B, H, Kq, D = q_centroids.shape
    Kk = k_centroids.shape[2]
    device = q_centroids.device
    
    # 1. Compute attention scores: (Q @ K.T) / sqrt(D)
    scale = 1.0 / math.sqrt(D)
    # [B, H, Kq, D] @ [B, H, Kk, D].T -> [B, H, Kq, Kk]
    scores = torch.einsum('bhqd,bhkd->bhqk', q_centroids.float(), k_centroids.float()) * scale
    
    # 2. Weight scores by Key cluster sizes (Importance of Key)
    # Note: SVG original logic weights by K size, not Q*K size
    k_weights = k_cluster_sizes.unsqueeze(-2).float() # [B, H, 1, Kk]
    
    # 3. Weighted Softmax per Query (Row-wise)
    # This computes probability distribution of attention for each Query cluster
    max_score = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - max_score)
    weighted_exp = k_weights * exp_scores
    weighted_probs = weighted_exp / torch.sum(weighted_exp, dim=-1, keepdim=True).clamp(min=1e-12)
    
    # 4. Optional Top-K cap (Optimization for speed)
    if max_k_clusters_per_q is not None:
        target_k = min(max_k_clusters_per_q, Kk)
        _, topk_idx = torch.topk(weighted_probs, k=target_k, dim=-1)
        block_mask = torch.zeros((B, H, Kq, Kk), device=device, dtype=torch.bool)
        block_mask.scatter_(-1, topk_idx, True)
        return block_mask

    # 5. Sort by probability (Descending)
    sorted_probs, sorted_indices = torch.sort(weighted_probs, dim=-1, descending=True)
    
    # 6. Cumulative Sum to find Top-P cutoff
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 7. Determine which to remove
    # Logic: remove if cumsum > p, but keep the first one that crosses the threshold
    # SVG logic: remove_indices = cumsum > p; shift right; keep first
    remove_indices = cumsum_probs > top_p
    remove_indices[..., 1:] = remove_indices[..., :-1].clone()
    remove_indices[..., 0] = False # Always keep at least the most important one
    
    # 8. Min Ratio Protection
    preserve_length = 0
    if min_kc_ratio > 0:
        preserve_length = int(min_kc_ratio * Kk)
        remove_indices[..., :preserve_length] = False
        
    sorted_clusters_to_keep = ~remove_indices
    
    # DEBUG: Print top-p selection stats
    kept_per_q = sorted_clusters_to_keep.sum(dim=-1).float()
    logger.info(f"[SVG2 Debug] Top-P Selection (p={top_p}, min_kc_ratio={min_kc_ratio}):")
    logger.info(f"  Total K clusters: {Kk}")
    logger.info(f"  Preserve length (min_kc_ratio): {preserve_length}")
    logger.info(f"  Kept K per Q: min={kept_per_q.min().item():.0f}, max={kept_per_q.max().item():.0f}, mean={kept_per_q.mean().item():.1f}")
    logger.info(f"  Avg selection ratio: {kept_per_q.mean().item()/Kk*100:.2f}%")
    
    # 9. Map back to original indices
    block_mask = torch.zeros((B, H, Kq, Kk), device=device, dtype=torch.bool)
    block_mask.scatter_(-1, sorted_indices, sorted_clusters_to_keep)
    
    return block_mask


# ============================================================================
# Part 4: Triton Block Sparse Attention Kernel
# ============================================================================


# ============================================================================
# Part 4: FlashInfer-style Indirect Access Attention Kernels
# ============================================================================


# ============================================================================
# 1. Planning Kernel (Index Expansion)
# ============================================================================
@triton.jit
def _kv_index_expansion_kernel(
    base_id_ptr, length_ptr, write_off_ptr, out_idx_ptr,
    MAX_BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    base = tl.load(base_id_ptr + pid)
    length = tl.load(length_ptr + pid)
    write_off = tl.load(write_off_ptr + pid)
    
    offs = tl.arange(0, MAX_BLOCK_SIZE)
    mask = offs < length
    tl.store(out_idx_ptr + write_off + offs, base + offs, mask=mask)

# ============================================================================
# 2. Phase 1: Split-K Compute Kernel
# ============================================================================
@triton.jit
def _split_k_compute_kernel(
    Q, K, V,
    # Outputs to Intermediate Buffers
    Tmp_Acc, # [Num_Tasks, SPLIT_K, BLOCK_M, D]
    Tmp_M,   # [Num_Tasks, SPLIT_K, BLOCK_M]
    Tmp_L,   # [Num_Tasks, SPLIT_K, BLOCK_M]
    
    # Indirection Lists
    kv_indices_ptr,
    task_k_start_ptr, task_k_end_ptr, # Per-Task K bounds
    
    # Task Metadata
    task_q_base_ptr, task_q_len_ptr,
    
    # Strides
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_tmp_t, stride_tmp_s, stride_tmp_m, stride_tmp_d, # Strides for intermediate buffers
    stride_m_t, stride_m_s, # Strides for Tmp_M/L
    
    # Config
    sm_scale,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (Num_Tasks * SPLIT_K)
    # PID layout: [Task_0_Split_0, Task_0_Split_1, ..., Task_1_Split_0...]
    pid = tl.program_id(0)
    
    task_id = pid // SPLIT_K
    split_id = pid % SPLIT_K
    
    # 1. Load Task Info
    q_len = tl.load(task_q_len_ptr + task_id)
    k_list_start = tl.load(task_k_start_ptr + task_id)
    k_list_end = tl.load(task_k_end_ptr + task_id)
    
    total_k_work = k_list_end - k_list_start
    if total_k_work <= 0:
        # Initialize L=0, M=-inf for safety in reduction
        offs_m = tl.arange(0, BLOCK_M)
        mask_m = offs_m < q_len
        # Pointers for M/L
        m_ptr = Tmp_M + task_id * stride_m_t + split_id * stride_m_s + offs_m
        l_ptr = Tmp_L + task_id * stride_m_t + split_id * stride_m_s + offs_m
        tl.store(m_ptr, -float('inf'), mask=mask_m)
        tl.store(l_ptr, 0.0, mask=mask_m)
        return

    # 2. Determine K range for THIS split
    # Divide total work evenly among splits
    # chunks_total = ceil(total_k_work / BLOCK_N)
    # chunks_per_split = ceil(chunks_total / SPLIT_K)
    # To keep it simple: just divide ranges
    
    work_per_split = (total_k_work + SPLIT_K - 1) // SPLIT_K
    # Ensure aligned to BLOCK_N? Not strictly necessary for correctness, but good for perf.
    # Let's just iterate linearly.
    
    my_k_start = k_list_start + split_id * work_per_split
    my_k_end = min(k_list_end, my_k_start + work_per_split)
    
    if my_k_start >= my_k_end:
        # No work for this split
        offs_m = tl.arange(0, BLOCK_M)
        mask_m = offs_m < q_len
        m_ptr = Tmp_M + task_id * stride_m_t + split_id * stride_m_s + offs_m
        l_ptr = Tmp_L + task_id * stride_m_t + split_id * stride_m_s + offs_m
        tl.store(m_ptr, -float('inf'), mask=mask_m)
        tl.store(l_ptr, 0.0, mask=mask_m)
        return

    # 3. Load Q Tile
    q_phys_base = tl.load(task_q_base_ptr + task_id)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptr = Q + q_phys_base * stride_qs + offs_m[:, None] * stride_qs + offs_d[None, :]
    q_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < BLOCK_D)
    q_tile = tl.load(q_ptr, mask=q_mask, other=0.0)
    
    # 4. Initialize Accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # 5. Compute Loop (Standard FlashAttention)
    offs_n = tl.arange(0, BLOCK_N)
    
    for k_idx in range(my_k_start, my_k_end, BLOCK_N):
        current_block_n = min(BLOCK_N, my_k_end - k_idx)
        
        # Indirect Load K Indices
        k_phys_ids = tl.load(kv_indices_ptr + k_idx + offs_n, mask=offs_n < current_block_n, other=0)
        
        # Indirect Load K/V
        k_ptrs = K + k_phys_ids[:, None] * stride_ks + offs_d[None, :]
        v_ptrs = V + k_phys_ids[:, None] * stride_vs + offs_d[None, :]
        
        load_mask = (offs_n[:, None] < current_block_n) & (offs_d[None, :] < BLOCK_D)
        k_tile = tl.load(k_ptrs, mask=load_mask, other=0.0)
        v_tile = tl.load(v_ptrs, mask=load_mask, other=0.0)
        
        # Attn
        qk = tl.dot(q_tile, tl.trans(k_tile))
        qk *= sm_scale
        qk = tl.where(offs_n[None, :] < current_block_n, qk, float("-inf"))
        
        m_curr = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_curr)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
        
        p = p.to(v_tile.dtype)
        acc = acc * alpha[:, None] + tl.dot(p, v_tile)
    
    # 6. Store Intermediate Results (No Normalization yet)
    # Tmp Buffers are [Num_Tasks, SPLIT_K, BLOCK_M, D]
    
    # Store M
    m_out_ptr = Tmp_M + task_id * stride_m_t + split_id * stride_m_s + offs_m
    tl.store(m_out_ptr, m_i, mask=offs_m < q_len)
    
    # Store L
    l_out_ptr = Tmp_L + task_id * stride_m_t + split_id * stride_m_s + offs_m
    tl.store(l_out_ptr, l_i, mask=offs_m < q_len)
    
    # Store Acc
    # 3D offsets for Acc: [task, split, m, d]
    acc_out_ptr = Tmp_Acc + task_id * stride_tmp_t + split_id * stride_tmp_s + \
                  offs_m[:, None] * stride_tmp_m + offs_d[None, :] * stride_tmp_d
    
    tl.store(acc_out_ptr, acc.to(Tmp_Acc.dtype.element_ty), mask=q_mask)


# ============================================================================
# 3. Phase 2: Reduction Kernel
# ============================================================================
@triton.jit
def _split_k_reduce_kernel(
    Out,
    Tmp_Acc, Tmp_M, Tmp_L,
    
    # Task Metadata to find write location
    task_q_base_ptr, task_q_len_ptr,
    
    # Strides
    stride_os, stride_od,
    stride_tmp_t, stride_tmp_s, stride_tmp_m, stride_tmp_d,
    stride_m_t, stride_m_s,
    
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (Num_Tasks)
    # Each program merges results for one Q Tile
    task_id = tl.program_id(0)
    
    q_len = tl.load(task_q_len_ptr + task_id)
    q_phys_base = tl.load(task_q_base_ptr + task_id)
    
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < q_len
    
    # 1. Find Global Max (M_global) across all splits
    # Logic: M_global = max(M_0, M_1, ... M_k)
    m_global = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    
    for s in range(SPLIT_K):
        m_ptr = Tmp_M + task_id * stride_m_t + s * stride_m_s + offs_m
        m_s = tl.load(m_ptr, mask=mask_m, other=-float("inf"))
        m_global = tl.maximum(m_global, m_s)
        
    # 2. Compute Global Sum (L_global) and Weighted Acc
    # L_global = sum(L_s * exp(M_s - M_global))
    # Acc_global = sum(Acc_s * exp(M_s - M_global))
    
    l_global = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_global = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    for s in range(SPLIT_K):
        # Load M, L for split s
        m_ptr = Tmp_M + task_id * stride_m_t + s * stride_m_s + offs_m
        l_ptr = Tmp_L + task_id * stride_m_t + s * stride_m_s + offs_m
        
        m_s = tl.load(m_ptr, mask=mask_m, other=-float("inf"))
        l_s = tl.load(l_ptr, mask=mask_m, other=0.0)
        
        # Load Acc for split s
        acc_ptr = Tmp_Acc + task_id * stride_tmp_t + s * stride_tmp_s + \
                  offs_m[:, None] * stride_tmp_m + offs_d[None, :] * stride_tmp_d
        
        acc_s = tl.load(acc_ptr, mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_D), other=0.0)
        
        # Rescale factor
        alpha = tl.exp(m_s - m_global)
        
        l_global += l_s * alpha
        acc_global += acc_s * alpha[:, None]
        
    # 3. Normalize and Write
    l_safe = tl.where(l_global == 0, 1.0, l_global)
    out_val = acc_global / l_safe[:, None]
    out_val = tl.where(l_global[:, None] == 0, 0.0, out_val)
    
    out_base = Out + q_phys_base * stride_os
    out_offsets = offs_m[:, None] * stride_os + offs_d[None, :]
    tl.store(out_base + out_offsets, out_val.to(Out.dtype.element_ty), 
             mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_D))



# ============================================================================
# 4. Main Function: Block Sparse Attn with Split-K
# ============================================================================

def block_sparse_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_mask: torch.Tensor,
    q_cluster_sizes: torch.Tensor,
    k_cluster_sizes: torch.Tensor,
    split_k: int = 4 # 默认 Split-K 因子，可调
) -> torch.Tensor:
    
    B, H, S, D = q.shape
    QC = q_cluster_sizes.shape[-1]
    KC = k_cluster_sizes.shape[-1]
    device = q.device
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    
    # === PLANNING PHASE ===
    
    # 1. CSR Indices Generation (Same as before)
    # Use reshape instead of view to handle non-contiguous inputs
    flat_mask = block_mask.reshape(-1, KC)
    sparse_mask = flat_mask.float().to_sparse_csr()
    q_blk_indptr = sparse_mask.crow_indices().int()
    k_blk_indices = sparse_mask.col_indices().int()
    
    # Calculate global offsets for K blocks
    qc_offsets = torch.zeros((B, H, QC + 1), device=device, dtype=torch.int32)
    qc_offsets[..., 1:] = torch.cumsum(q_cluster_sizes, dim=-1)
    kc_offsets = torch.zeros((B, H, KC + 1), device=device, dtype=torch.int32)
    kc_offsets[..., 1:] = torch.cumsum(k_cluster_sizes, dim=-1)
    
    # Expand active K lists (Index Expansion Kernel)
    # Logic: Flatten B/H/KC to find global K offsets
    active_counts = q_blk_indptr[1:] - q_blk_indptr[:-1]
    q_global_ids = torch.repeat_interleave(torch.arange(B*H*QC, device=device, dtype=torch.int32), active_counts)
    batch_ids = q_global_ids // (H * QC)
    head_ids = (q_global_ids // QC) % H
    
    flat_kc_offsets = kc_offsets.reshape(B*H, KC+1)
    bh_ids = batch_ids * H + head_ids
    
    active_k_starts = flat_kc_offsets[bh_ids, k_blk_indices]
    active_k_ends = flat_kc_offsets[bh_ids, k_blk_indices + 1]
    active_k_lengths = active_k_ends - active_k_starts
    global_k_offsets = bh_ids * S + active_k_starts
    
    NNZ = active_k_lengths.numel()
    total_active_k_tokens = active_k_lengths.sum().item()
    
    kv_indices = torch.empty(total_active_k_tokens, device=device, dtype=torch.int64)
    write_offsets = torch.zeros(NNZ + 1, device=device, dtype=torch.int32)
    write_offsets[1:] = torch.cumsum(active_k_lengths, dim=0)
    
    if NNZ > 0:
        max_len = int(active_k_lengths.max().item())
        _kv_index_expansion_kernel[(NNZ,)](
            global_k_offsets, active_k_lengths, write_offsets, kv_indices, 
            triton.next_power_of_2(max_len)
        )
    
    # 2. Scheduling (Q-Tiling + Task Mapping)
    # Use reshape instead of view
    flat_q_sizes = q_cluster_sizes.reshape(-1)
    tiles_per_q_blk = (flat_q_sizes + BLOCK_M - 1) // BLOCK_M
    total_q_tiles = tiles_per_q_blk.sum().item()
    
    # Map Tasks
    task_to_q_map = torch.repeat_interleave(
        torch.arange(B*H*QC, device=device, dtype=torch.int32), tiles_per_q_blk
    )
    
    # Calculate Task Q Bounds
    cum_tiles = torch.zeros(B*H*QC + 1, device=device, dtype=torch.int32)
    cum_tiles[1:] = torch.cumsum(tiles_per_q_blk, dim=0)
    task_start_indices = cum_tiles[task_to_q_map]
    task_local_idx = torch.arange(total_q_tiles, device=device, dtype=torch.int32) - task_start_indices
    offset_in_cluster = task_local_idx * BLOCK_M
    
    flat_q_starts = qc_offsets[..., :-1].reshape(-1)
    q_cluster_base = flat_q_starts[task_to_q_map]
    t_batch = task_to_q_map // (H * QC)
    t_head = (task_to_q_map // QC) % H
    
    task_q_global_base = (t_batch * H + t_head) * S + q_cluster_base + offset_in_cluster
    current_q_sizes = flat_q_sizes[task_to_q_map]
    task_q_lens = torch.clamp(current_q_sizes - offset_in_cluster, max=BLOCK_M)
    
    # Map Task -> K Range
    q_cluster_start_block = q_blk_indptr[task_to_q_map].long()
    q_cluster_end_block = q_blk_indptr[task_to_q_map + 1].long()
    task_k_token_starts = write_offsets[q_cluster_start_block]
    task_k_token_ends = write_offsets[q_cluster_end_block]
    
    # === ALLOCATION FOR SPLIT-K ===
    # Intermediate buffers need to store results for each task and each split
    # Shape: [Total_Tasks, SPLIT_K, BLOCK_M, D]
    tmp_acc = torch.empty((total_q_tiles, split_k, BLOCK_M, D), device=device, dtype=torch.float32)
    tmp_m = torch.empty((total_q_tiles, split_k, BLOCK_M), device=device, dtype=torch.float32)
    tmp_l = torch.empty((total_q_tiles, split_k, BLOCK_M), device=device, dtype=torch.float32)
    
    out = torch.empty_like(q)
    
    # DEBUG: Print sparse attention stats
    total_q_tokens = B * H * S
    total_kv_work = total_active_k_tokens
    dense_kv_work = B * H * QC * S  # If all blocks were active
    sparsity_ratio = 1.0 - (total_kv_work / dense_kv_work) if dense_kv_work > 0 else 0.0
    logger.info(f"[SVG2 Debug] Sparse Attention Compute:")
    logger.info(f"  Total Q tiles: {total_q_tiles}")
    logger.info(f"  Active K tokens: {total_active_k_tokens}/{dense_kv_work} ({100*(1-sparsity_ratio):.2f}%)")
    logger.info(f"  Compute sparsity: {sparsity_ratio*100:.2f}%")
    logger.info(f"  Split-K factor: {split_k}")
    
    # === RUNNING PHASE 1: COMPUTE ===
    # Grid: One thread block per Split per Task
    grid_compute = (total_q_tiles * split_k, )
    
    _split_k_compute_kernel[grid_compute](
        q, k, v,
        tmp_acc, tmp_m, tmp_l,
        kv_indices,
        task_k_token_starts, task_k_token_ends,
        task_q_global_base, task_q_lens,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        tmp_acc.stride(0), tmp_acc.stride(1), tmp_acc.stride(2), tmp_acc.stride(3),
        tmp_m.stride(0), tmp_m.stride(1),
        1.0 / math.sqrt(D),
        SPLIT_K=split_k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
    )
    
    # === RUNNING PHASE 2: REDUCE ===
    # Grid: One thread block per Task (merging all splits)
    grid_reduce = (total_q_tiles, )
    
    _split_k_reduce_kernel[grid_reduce](
        out,
        tmp_acc, tmp_m, tmp_l,
        task_q_global_base, task_q_lens,
        out.stride(2), out.stride(3), # Stride S, Stride D
        tmp_acc.stride(0), tmp_acc.stride(1), tmp_acc.stride(2), tmp_acc.stride(3),
        tmp_m.stride(0), tmp_m.stride(1),
        SPLIT_K=split_k,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D
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
    max_k_clusters_per_q: Optional[int] = None,
    init_q_centroids: Optional[torch.Tensor] = None,
    init_k_centroids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Complete SVG2 (Semantic-Aware Permutation) attention forward pass.

    Args:
        q, k, v: Query, Key, Value tensors [B, S, H, D]
        num_q_clusters: Number of query clusters
        num_k_clusters: Number of key clusters
        top_p: Top-p fraction for block mask
        kmeans_iters: K-Means iterations
        max_k_clusters_per_q: Optional cap of kept key clusters per query cluster
        init_q_centroids: Initial centroids for Query K-Means
        init_k_centroids: Initial centroids for Key K-Means
    
    Returns:
        output: Attention output [B, S, H, D]
        final_q_centroids: Updated Query centroids [B, H, Kq, D]
        final_k_centroids: Updated Key centroids [B, H, Kk, D]
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
    
    # Handle initial centroids if provided
    # They come in as [B, H, K, D], need to reshape to [B*H, K, D] (which triton_kmeans expects for batched init)
    # triton_kmeans expects [B_kmeans, K, D] where B_kmeans = B*H here
    
    q_init = None
    if init_q_centroids is not None:
        q_init = init_q_centroids.reshape(B * H, num_q_clusters, D)
        
    k_init = None
    if init_k_centroids is not None:
        k_init = init_k_centroids.reshape(B * H, num_k_clusters, D)
    
    # Step 1: K-Means clustering
    q_labels, q_centroids, q_cluster_sizes = triton_kmeans(
        q_flat, num_q_clusters, max_iters=kmeans_iters, init_centroids=q_init
    )
    k_labels, k_centroids, k_cluster_sizes = triton_kmeans(
        k_flat, num_k_clusters, max_iters=kmeans_iters, init_centroids=k_init
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
        max_k_clusters_per_q=max_k_clusters_per_q,
    )
    
    # DEBUG: Print block mask statistics
    total_blocks = block_mask.numel()
    active_blocks = block_mask.sum().item()
    sparsity = 1.0 - (active_blocks / total_blocks)
    logger.info(f"[SVG2 Debug] Block Mask Stats:")
    logger.info(f"  Block shape: {block_mask.shape} [B, H, Kq, Kk]")
    logger.info(f"  Active blocks: {active_blocks}/{total_blocks} ({100*(1-sparsity):.2f}%)")
    logger.info(f"  Sparsity: {sparsity*100:.2f}%")
    logger.info(f"  Q cluster sizes: min={q_cluster_sizes.min().item()}, max={q_cluster_sizes.max().item()}, mean={q_cluster_sizes.float().mean().item():.1f}")
    logger.info(f"  K cluster sizes: min={k_cluster_sizes.min().item()}, max={k_cluster_sizes.max().item()}, mean={k_cluster_sizes.float().mean().item():.1f}")
    
    # Step 3: Permute Q, K, V by cluster labels
    q_perm, q_sorted_indices = permute_by_labels(q, labels=q_labels)
    k_perm, k_sorted_indices = permute_by_labels(k, labels=k_labels)
    v_perm, _ = permute_by_labels(v, sorted_indices=k_sorted_indices)
    
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
    
    return output, q_centroids, k_centroids


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
    # Optional cap: for each query cluster, keep at most this many key clusters.
    # This is typically the most important knob for latency.
    max_k_clusters_per_q: Optional[int] = None
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
        max_k_clusters_per_q = kwargs.get("max_k_clusters_per_q", None)
        return SVG2SparseAttentionMetadata(
            current_timestep=current_timestep,
            num_frames=num_frames,
            num_tokens_per_frame=num_tokens_per_frame,
            num_q_clusters=num_q_clusters,
            num_k_clusters=num_k_clusters,
            top_p=top_p,
            kmeans_iters=kmeans_iters,
            max_k_clusters_per_q=max_k_clusters_per_q,
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
    DEFAULT_MAX_K_CLUSTERS_PER_Q: Optional[int] = None
    DEFAULT_FIRST_LAYERS_FP = 0.0  # Ratio of first layers using full attention
    DEFAULT_FIRST_TIMES_FP = 0.0   # Ratio of first timesteps using full attention
    
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
        max_k_clusters_per_q: Optional[int] = None,
        first_layers_fp: float = 0.0,  # Ratio of first layers using full attention
        first_times_fp: float = 0.0,   # Ratio of first timesteps using full attention
        total_layers: int = 40,        # Total number of transformer layers
        total_timesteps: int = 1000,   # Total timesteps in diffusion
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
        self.max_k_clusters_per_q = max_k_clusters_per_q
        
        # Convert ratios to actual thresholds
        self.total_layers = total_layers
        self.total_timesteps = total_timesteps
        self.first_layers_fp = first_layers_fp
        self.first_times_fp = first_times_fp
        
        # first_layers_threshold: layers 0 to threshold-1 use full attention
        self.first_layers_threshold = int(first_layers_fp * total_layers)
        # first_times_threshold: timesteps > threshold use full attention
        # Diffusion timesteps decrease from total_timesteps to 0
        # first_times_fp=0.35 means first 35% of timesteps (high values) use full attention
        # So threshold = total_timesteps * (1 - first_times_fp)
        self.first_times_threshold = int((1.0 - first_times_fp) * total_timesteps)
        
        # Centroid cache for iterative refinement across timesteps
        self.q_centroids = None
        self.k_centroids = None
        self.centroids_initialized = False
        
        # Extract layer index from prefix if available
        self.layer_idx = self._extract_layer_idx(prefix)
        
        # Log configuration on first layer only
        if self.layer_idx == 0:
            logger.info(f"SVG2 Sparse Attention Config:")
            logger.info(f"  num_q_clusters={num_q_clusters}, num_k_clusters={num_k_clusters}")
            logger.info(f"  top_p={top_p}, kmeans_iters={kmeans_iters}")
            logger.info(f"  first_layers_fp={first_layers_fp} -> threshold={self.first_layers_threshold}/{total_layers} layers")
            logger.info(f"  first_times_fp={first_times_fp} -> timestep_threshold={self.first_times_threshold}/{total_timesteps}")
    
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
        - if layer_idx < first_layers_threshold: use full attention
        - if timestep > first_times_threshold: use full attention
        
        Note: timestep decreases from total_timesteps to 0 during inference,
        so timestep > threshold means early inference steps (warm-up phase).
        """
        # First N layers always use full attention
        if self.layer_idx < self.first_layers_threshold:
            return True
        
        # Early timesteps (high values) use full attention
        # timestep > threshold means we're in the early warm-up phase
        if timestep is not None and timestep > self.first_times_threshold:
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
            max_k_clusters_per_q = getattr(attn_metadata, 'max_k_clusters_per_q', self.max_k_clusters_per_q)
            # Get timestep for full attention decision
            timestep = getattr(attn_metadata, 'current_timestep', None)
        else:
            num_q_clusters = self.num_q_clusters
            num_k_clusters = self.num_k_clusters
            top_p = self.top_p
            kmeans_iters = self.kmeans_iters
            max_k_clusters_per_q = self.max_k_clusters_per_q
            timestep = None
        
        # DEBUG: Print actual parameters being used (only for layer 0 to avoid spam)
        if self.layer_idx == 0:
            logger.info(f"[SVG2 Debug Layer {self.layer_idx}] Forward call:")
            logger.info(f"  Query shape: {query.shape}, dtype: {query.dtype}")
            logger.info(f"  num_q_clusters={num_q_clusters}, num_k_clusters={num_k_clusters}")
            logger.info(f"  top_p={top_p}, kmeans_iters={kmeans_iters}")
            logger.info(f"  max_k_clusters_per_q={max_k_clusters_per_q}")
            logger.info(f"  timestep={timestep}, first_times_threshold={self.first_times_threshold}")
            logger.info(f"  first_layers_threshold={self.first_layers_threshold}")
        
        # Check if we should use full attention (early layers or early timesteps)
        use_full_attn = self._should_use_full_attention(timestep)
        if self.layer_idx == 0:
            logger.info(f"  use_full_attention={use_full_attn}")
        
        if use_full_attn:
            # Use standard scaled dot-product attention
            return self._full_attention(query, key, value)
        
        # Use SVG2 sparse attention
        
        # Determine if we can reuse centroids from previous steps
        # If centroids are initialized and we are in a later timestep (or logic permits), reuse them
        # Note: In reverse diffusion, timesteps decrease. 
        # But we simply check if we have cached centroids.
        
        # Default strategy: 
        # - If no cache: init mode (more iters)
        # - If cache: step mode (fewer iters, reuse centroids)
        
        current_kmeans_iters = kmeans_iters
        init_q = None
        init_k = None
        centroid_reuse = False
        
        if self.centroids_initialized and self.q_centroids is not None:
            # Check shape compatibility (batch size might change or be consistent)
            # q_centroids shape: [B, H, K, D]
            if self.q_centroids.shape[0] == query.shape[0] and \
               self.q_centroids.shape[1] == query.shape[2]: # num_heads
                init_q = self.q_centroids
                init_k = self.k_centroids
                # Use fewer iterations for update step (e.g., 1 or 2)
                # Original SVG uses 1 or similar small number for steps
                current_kmeans_iters = 1
                centroid_reuse = True
        
        if self.layer_idx == 0:
            logger.info(f"  centroid_reuse={centroid_reuse}, kmeans_iters={current_kmeans_iters} (config={kmeans_iters})")
        
        output, new_q_centroids, new_k_centroids = svg2_attention_forward(
            query, key, value,
            num_q_clusters=num_q_clusters,
            num_k_clusters=num_k_clusters,
            top_p=top_p,
            kmeans_iters=current_kmeans_iters,
            max_k_clusters_per_q=max_k_clusters_per_q,
            init_q_centroids=init_q,
            init_k_centroids=init_k,
        )
        
        # Update cache
        self.q_centroids = new_q_centroids
        self.k_centroids = new_k_centroids
        self.centroids_initialized = True
        
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



