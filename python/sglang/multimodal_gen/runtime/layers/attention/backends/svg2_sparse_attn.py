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
# Part 1: Triton Kernels for K-Means Clustering (ported from SVG)
# ============================================================================

# -----------------------------------------------------------------------------
# Triton kernel: compute nearest-centroid IDs (Euclidean distance)
# This is ported from SVG's kmeans_utils.py for exact numerical consistency
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=4, num_warps=wp)
        for BN in [32, 64, 128]
        for BK in [32, 64, 128]
        for wp in [4, 8]
        if not (BN * BK < 32 * 32 and wp > 4)  # Prune unbalanced configs
    ],
    key=["N", "K"],
)
@triton.jit
def _svg_euclid_assign_kernel(
    x_ptr,       # [B, N, D]
    c_ptr,       # [B, K, D]
    x_sq_ptr,    # [B, N] - precomputed ||x||^2
    out_ptr,     # [B, N] - output cluster IDs
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each program handles a tile of BLOCK_N points for a given batch element.
    Iterates over centroids in chunks of BLOCK_K and maintains running minimum.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Load x tile (BLOCK_N, D)
    offs_d = tl.arange(0, D)
    x_ptrs = x_ptr + pid_b * stride_x_b + n_offsets[:, None] * stride_x_n + offs_d[None, :] * stride_x_d
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)

    # Pre-load x_sq for the tile (BLOCK_N,)
    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    # Init best distance / index
    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # Iterate over centroids in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load centroid tile (D, BLOCK_K)
        c_ptrs = c_ptr + pid_b * stride_c_b + k_offsets[None, :] * stride_c_k + offs_d[:, None] * stride_c_d
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)

        # Compute centroid squared norms (BLOCK_K,)
        cent_sq = tl.sum(c_tile * c_tile, axis=0).to(tl.float32)

        # Compute cross term (BLOCK_N, BLOCK_K) = x_tile @ c_tile
        cross = tl.dot(x_tile, c_tile).to(tl.float32)

        # Squared Euclidean distance
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)

        # Mask out invalid centroid columns
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # Write results
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


def _svg_euclid_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
) -> torch.Tensor:
    """
    Return nearest-centroid indices using Triton kernel.
    Ported from SVG's euclid_assign_triton.
    """
    B, N, D = x.shape
    K = centroids.shape[1]
    
    out = torch.empty((B, N), device=x.device, dtype=torch.int64)
    
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_out_b, stride_out_n = out.stride()
    
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)
    
    _svg_euclid_assign_kernel[grid](
        x, centroids, x_sq, out,
        B, N, K, D,
        stride_x_b, stride_x_n, stride_x_d,
        stride_c_b, stride_c_k, stride_c_d,
        stride_xsq_b, stride_xsq_n,
        stride_out_b, stride_out_n,
    )
    return out


# -----------------------------------------------------------------------------
# Triton kernel: chunk-wise centroid update (sorted IDs)
# This is ported from SVG's _centroid_update_chunk_kernel
# -----------------------------------------------------------------------------

@triton.jit
def _svg_centroid_update_chunk_kernel(
    x_ptr,               # [B, N, D] - original features
    sorted_idx_ptr,      # [B, N] - indices after sort
    sorted_cluster_ptr,  # [B, N] - cluster ids in sorted order
    sum_ptr,             # [B, K, D] - output sums
    count_ptr,           # [B, K] - output counts
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Each program processes BLOCK_N consecutive, already-sorted tokens.
    Accumulates local sum/count for each cluster run and performs atomic update.
    """
    pid_chunk = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    b = pid_b
    chunk_start = pid_chunk * BLOCK_N

    if chunk_start >= N:
        return

    # Base pointers for this batch
    idx_batch_base = sorted_idx_ptr + b * N
    cid_batch_base = sorted_cluster_ptr + b * N
    x_batch_base = x_ptr + b * N * D

    # Helper aranges
    offs_token = tl.arange(0, BLOCK_N)
    offs_dim = tl.arange(0, D)

    # Token index & validity mask
    token_idx = chunk_start + offs_token
    valid_tok = token_idx < N
    first_token_idx = chunk_start
    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

    # Load cluster IDs
    first_id = tl.load(cid_batch_base + first_token_idx)
    last_id = tl.load(cid_batch_base + last_token_idx)
    all_ids = tl.load(cid_batch_base + token_idx, mask=valid_tok, other=-1)

    all_tokens_idxs = tl.load(idx_batch_base + token_idx, mask=valid_tok, other=-1)
    load_mask = all_tokens_idxs[:, None] * D + offs_dim[None, :]

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            cluster_feats = tl.load(x_batch_base + load_mask, mask=cluster_mask[:, None], other=0.0)
            cluster_feats = cluster_feats.to(tl.float32)
            sum_feats = tl.sum(cluster_feats, axis=0)
            dest_ptr = sum_ptr + (b * K + cid) * D + offs_dim
            tl.atomic_add(dest_ptr, sum_feats)
            tl.atomic_add(count_ptr + b * K + cid, cluster_size)


def _svg_centroid_update_sorted(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
    BLOCK_N: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast centroid update for Euclidean K-Means.
    Ported from SVG's triton_centroid_update_sorted_euclid.
    """
    B, N, D = x.shape
    K = old_centroids.shape[1]
    
    # Batch-wise sort of cluster assignments
    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)
    
    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    centroid_cnts = torch.zeros((B, K), device=x.device, dtype=torch.int32)
    
    grid = (triton.cdiv(N, BLOCK_N), B)
    _svg_centroid_update_chunk_kernel[grid](
        x,
        sorted_idx_int,
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_cnts,
        B, N, D, K,
        BLOCK_N=BLOCK_N,
    )
    
    # Convert sums to means; replace empty clusters with old centroids
    counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f
    empty_mask = (centroid_cnts == 0).unsqueeze(-1)
    centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
    
    return centroids.to(x.dtype), centroid_cnts


# -----------------------------------------------------------------------------
# Legacy Triton kernels (kept for reference, but not used by triton_kmeans)
# -----------------------------------------------------------------------------

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
    K-Means clustering using Triton kernels (ported from SVG's batch_kmeans_Euclid).
    
    This implementation uses the same Triton kernels as SVG for exact numerical
    consistency:
    1. _svg_euclid_assign_kernel: Distance computation + cluster assignment
    2. _svg_centroid_update_chunk_kernel: Sorted centroid update
    
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
    
    # K-Means logging at debug level
    # logger.debug(f"[SVG2] K-Means: N={N}, K={K}, iters={max_iters}, init={'reuse' if init_centroids is not None else 'random'}")
    
    # Pre-compute squared L2 norm of all points (constant during iterations)
    x_sq = (x ** 2).sum(dim=-1)  # [B, N]
    
    # Initialize centroids (matching SVG's initialization)
    if init_centroids is not None:
        centroids = init_centroids.clone()
    else:
        indices = torch.randint(0, N, (B, K), device=device)
        centroids = torch.gather(
            x, dim=1, 
            index=indices.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, K, D]
    
    centroids = centroids.view(B, K, D)
    
    for iteration in range(max_iters):
        # ============================================================
        # Step 1: Cluster assignment using Triton kernel (matching SVG)
        # ============================================================
        labels = _svg_euclid_assign(x, centroids, x_sq)
        
        # ============================================================
        # Step 2: Centroid update using sorted Triton kernel (matching SVG)
        # ============================================================
        new_centroids, cluster_sizes = _svg_centroid_update_sorted(
            x, labels, centroids
        )
        
        # ============================================================
        # Step 3: Convergence check
        # ============================================================
        center_shift = (new_centroids - centroids).norm(dim=-1).max()
        centroids = new_centroids
        
        if center_shift < tol:
            break
    
    # Convert to expected dtypes
    labels = labels.to(torch.int64)
    cluster_sizes = cluster_sizes.to(torch.int64)
    
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
    *,
    match_sparse_videogen_numerics: bool = True,
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
    #
    # NOTE on numerical parity with Sparse-VideoGen:
    # Sparse-VideoGen's `identify_dynamic_map()` computes matmul in the incoming dtype
    # (fp16/bf16 typically), then runs softmax in fp32 and *casts the probabilities back*
    # to the original dtype before sort/cumsum/top-p selection. That cast can change the
    # ordering for near-ties at the top-p boundary and lead to a few-bit mask mismatch.
    #
    # SGLang's default path computes probabilities in fp32 for better stability.
    # For exact parity tests, set `match_sparse_videogen_numerics=True`.
    if match_sparse_videogen_numerics:
        # [B, H, Kq, D] @ [B, H, Kk, D].T -> [B, H, Kq, Kk] in input dtype
        scores = torch.matmul(q_centroids, k_centroids.transpose(-2, -1)) * scale
    else:
        scores = torch.einsum("bhqd,bhkd->bhqk", q_centroids.float(), k_centroids.float()) * scale
    
    # 2. Weight scores by Key cluster sizes (Importance of Key)
    # Note: SVG original logic weights by K size, not Q*K size
    k_weights = k_cluster_sizes.unsqueeze(-2).float() # [B, H, 1, Kk]
    
    # 3. Weighted Softmax per Query (Row-wise)
    # This computes probability distribution of attention for each Query cluster
    out_dtype = scores.dtype if match_sparse_videogen_numerics else None
    max_score = torch.max(scores.float(), dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores.float() - max_score)
    weighted_exp = k_weights * exp_scores
    weighted_probs = weighted_exp / torch.sum(weighted_exp, dim=-1, keepdim=True).clamp(min=1e-12)
    if out_dtype is not None:
        weighted_probs = weighted_probs.to(out_dtype)
    
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
    
    # DEBUG: Print top-p selection stats (debug level to reduce spam)
    kept_per_q = sorted_clusters_to_keep.sum(dim=-1).float()
    logger.debug(f"[SVG2] Top-P Selection (p={top_p}, min_kc_ratio={min_kc_ratio}):")
    logger.debug(f"  Kept K per Q: min={kept_per_q.min().item():.0f}, max={kept_per_q.max().item():.0f}, mean={kept_per_q.mean().item():.1f}")
    logger.debug(f"  Avg selection ratio: {kept_per_q.mean().item()/Kk*100:.2f}%")
    
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
    
    # DEBUG: Print sparse attention stats (debug level to reduce spam)
    total_q_tokens = B * H * S
    total_kv_work = total_active_k_tokens
    dense_kv_work = B * H * QC * S  # If all blocks were active
    sparsity_ratio = 1.0 - (total_kv_work / dense_kv_work) if dense_kv_work > 0 else 0.0
    logger.debug(f"[SVG2] Sparse Attn: Q_tiles={total_q_tiles}, Active_K={total_active_k_tokens}/{dense_kv_work} ({100*(1-sparsity_ratio):.1f}%), sparsity={sparsity_ratio*100:.1f}%")
    
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
    min_kc_ratio: float = 0.0,
    init_q_centroids: Optional[torch.Tensor] = None,
    init_k_centroids: Optional[torch.Tensor] = None,
    enable_profiling: bool = False,
    layer_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict]]:
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
        enable_profiling: Whether to collect timing statistics
        layer_idx: Layer index for logging
    
    Returns:
        output: Attention output [B, S, H, D]
        final_q_centroids: Updated Query centroids [B, H, Kq, D]
        final_k_centroids: Updated Key centroids [B, H, Kk, D]
        profile_stats: Optional dict with timing statistics
    """
    import time
    
    B, S, H, D = q.shape
    device = q.device
    dtype = q.dtype
    
    profile_stats = {} if enable_profiling else None
    
    # Transpose to [B, H, S, D] for easier processing
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    # ---------------------------------------------------------------------
    # SGLang compute flow using local Triton/PyTorch implementations
    # ---------------------------------------------------------------------

    # Flatten batch and head dimensions for K-Means (SVG expects [Bk, N, D])
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
    
    # Step 1: K-Means clustering (local Triton kernels)
    if enable_profiling:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    
    q_labels, q_centroids, q_cluster_sizes = triton_kmeans(
        q_flat, num_q_clusters, max_iters=kmeans_iters, init_centroids=q_init
    )
    k_labels, k_centroids, k_cluster_sizes = triton_kmeans(
        k_flat, num_k_clusters, max_iters=kmeans_iters, init_centroids=k_init
    )
    
    if enable_profiling:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        profile_stats['kmeans_ms'] = (t1 - t0) * 1000
    
    # Reshape back
    q_labels = q_labels.reshape(B * H, S)
    q_centroids = q_centroids.reshape(B, H, num_q_clusters, D)
    q_cluster_sizes = q_cluster_sizes.reshape(B, H, num_q_clusters)
    
    k_labels = k_labels.reshape(B * H, S)
    k_centroids = k_centroids.reshape(B, H, num_k_clusters, D)
    k_cluster_sizes = k_cluster_sizes.reshape(B, H, num_k_clusters)
    
    # Step 2: Generate dynamic block mask (local implementation; match SVG numerics)
    if enable_profiling:
        torch.cuda.synchronize()
        t2 = time.perf_counter()
    
    block_mask = identify_dynamic_mask(
        q_centroids,
        k_centroids,
        q_cluster_sizes,
        k_cluster_sizes,
        top_p=top_p,
        min_kc_ratio=min_kc_ratio,
        max_k_clusters_per_q=max_k_clusters_per_q,
        match_sparse_videogen_numerics=True,
    )
    
    if enable_profiling:
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        profile_stats['mask_gen_ms'] = (t3 - t2) * 1000
    
    # Compute block mask statistics
    total_blocks = block_mask.numel()
    active_blocks = block_mask.sum().item()
    block_retention_ratio = active_blocks / total_blocks
    sparsity = 1.0 - block_retention_ratio
    
    if enable_profiling:
        profile_stats['total_blocks'] = total_blocks
        profile_stats['active_blocks'] = active_blocks
        profile_stats['block_retention_pct'] = block_retention_ratio * 100
        profile_stats['sparsity_pct'] = sparsity * 100
    
    logger.debug(f"[SVG2] Block Mask: {active_blocks}/{total_blocks} active ({100*block_retention_ratio:.1f}%), sparsity={sparsity*100:.1f}%")
    
    # Step 3: Permute Q, K, V by cluster labels (local Triton kernels)
    if enable_profiling:
        torch.cuda.synchronize()
        t4 = time.perf_counter()
    
    q_perm, q_sorted_indices = permute_by_labels(q, labels=q_labels)
    k_perm, k_sorted_indices = permute_by_labels(k, labels=k_labels)
    v_perm, _ = permute_by_labels(v, sorted_indices=k_sorted_indices)
    
    if enable_profiling:
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        profile_stats['permute_ms'] = (t5 - t4) * 1000
    
    # Step 4: Block sparse attention (local Triton split-k)
    if enable_profiling:
        torch.cuda.synchronize()
        t6 = time.perf_counter()
    
    # Ensure expected dtype for offset computations inside Triton kernel
    q_cluster_sizes_i32 = q_cluster_sizes.to(torch.int32)
    k_cluster_sizes_i32 = k_cluster_sizes.to(torch.int32)
    out_perm = block_sparse_attention(
        q_perm,
        k_perm,
        v_perm,
        block_mask,
        q_cluster_sizes_i32,
        k_cluster_sizes_i32,
    )
    
    if enable_profiling:
        torch.cuda.synchronize()
        t7 = time.perf_counter()
        profile_stats['sparse_attn_ms'] = (t7 - t6) * 1000
    
    # Step 5: Inverse permutation (local Triton kernel)
    if enable_profiling:
        torch.cuda.synchronize()
        t8 = time.perf_counter()
    
    output = inverse_permute(out_perm, q_sorted_indices)
    
    if enable_profiling:
        torch.cuda.synchronize()
        t9 = time.perf_counter()
        profile_stats['inv_permute_ms'] = (t9 - t8) * 1000
        profile_stats['total_ms'] = (t9 - t0) * 1000
    
    # Transpose back to [B, S, H, D]
    output = output.transpose(1, 2).contiguous()
    
    return output, q_centroids, k_centroids, profile_stats


# ============================================================================
# Part 6: SGLang Attention Backend Integration
# ============================================================================


@dataclass
class SVG2SparseAttentionMetadata(AttentionMetadata):
    """Metadata for SVG2 sparse attention."""
    current_timestep: int  # Step index (0, 1, 2, ..., num_inference_steps-1)
    num_frames: int
    num_tokens_per_frame: int
    num_q_clusters: int = 64
    num_k_clusters: int = 300
    top_p: float = 0.9
    # KMeans iteration schedule (match Sparse-VideoGen SAP defaults)
    kmeans_iter_init: int = 5
    kmeans_iter_step: int = 1
    # Back-compat alias; if provided by older callers, it is treated as the "step" iters.
    kmeans_iters: int = 1
    # Minimum ratio of key clusters to keep (Sparse-VideoGen uses 0.10 in released scripts)
    min_kc_ratio: float = 0.0
    # Optional cap: for each query cluster, keep at most this many key clusters.
    # This is typically the most important knob for latency.
    max_k_clusters_per_q: Optional[int] = None
    # Cache for centroids (for iterative refinement)
    q_centroids_cache: Optional[torch.Tensor] = None
    k_centroids_cache: Optional[torch.Tensor] = None
    # Total number of inference steps (needed for first_times_fp calculation)
    num_inference_steps: int = 40


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
        kmeans_iter_init: int = 5,
        kmeans_iter_step: int = 1,
        min_kc_ratio: float = 0.0,
        num_inference_steps: int = 40,
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
            kmeans_iter_init=kmeans_iter_init,
            kmeans_iter_step=kmeans_iter_step,
            kmeans_iters=kmeans_iter_step,
            min_kc_ratio=min_kc_ratio,
            max_k_clusters_per_q=max_k_clusters_per_q,
            num_inference_steps=num_inference_steps,
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
        # first_times_fp: fraction of initial steps using full attention
        # first_times_fp=0.35 means first 35% of steps (low indices) use full attention
        # Threshold is calculated dynamically in forward() based on actual num_inference_steps
        
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
            logger.info(f"  first_layers_fp={first_layers_fp} -> first {self.first_layers_threshold}/{total_layers} layers use full attention")
            logger.info(f"  first_times_fp={first_times_fp} -> first {first_times_fp*100:.0f}% of inference steps use full attention")
    
    def _extract_layer_idx(self, prefix: str) -> int:
        """Extract layer index from prefix string like 'blocks.5.attn1'."""
        import re
        match = re.search(r'blocks\.(\d+)', prefix)
        if match:
            return int(match.group(1))
        return 0
    
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
        
        Note: The decision of when to use full attention vs sparse attention is now
        handled by the model layer (USPAttention_SVG2), not here. This method always
        performs SVG2 sparse attention.
        """
        # Get parameters from metadata or use defaults
        if attn_metadata is not None:
            num_q_clusters = getattr(attn_metadata, 'num_q_clusters', self.num_q_clusters)
            num_k_clusters = getattr(attn_metadata, 'num_k_clusters', self.num_k_clusters)
            top_p = getattr(attn_metadata, 'top_p', self.top_p)
            # KMeans schedule
            kmeans_iter_init = getattr(attn_metadata, 'kmeans_iter_init', getattr(attn_metadata, 'kmeans_iters', self.kmeans_iters))
            kmeans_iter_step = getattr(attn_metadata, 'kmeans_iter_step', getattr(attn_metadata, 'kmeans_iters', self.kmeans_iters))
            min_kc_ratio = getattr(attn_metadata, 'min_kc_ratio', 0.0)
            max_k_clusters_per_q = getattr(attn_metadata, 'max_k_clusters_per_q', self.max_k_clusters_per_q)
        else:
            num_q_clusters = self.num_q_clusters
            num_k_clusters = self.num_k_clusters
            top_p = self.top_p
            kmeans_iter_init = self.kmeans_iters
            kmeans_iter_step = self.kmeans_iters
            min_kc_ratio = 0.0
            max_k_clusters_per_q = self.max_k_clusters_per_q
        
        # Determine if we can reuse centroids from previous steps
        # Default strategy: 
        # - If no cache: init mode (more iters)
        # - If cache: step mode (fewer iters, reuse centroids)
        
        current_kmeans_iters = kmeans_iter_init
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
                # Match Sparse-VideoGen SAP: do a small refinement each diffusion step.
                current_kmeans_iters = kmeans_iter_step
                centroid_reuse = True
        
        # Enable profiling only for layer 0 to reduce logging overhead
        enable_profiling = (self.layer_idx == 0)
        
        output, new_q_centroids, new_k_centroids, profile_stats = svg2_attention_forward(
            query, key, value,
            num_q_clusters=num_q_clusters,
            num_k_clusters=num_k_clusters,
            top_p=top_p,
            kmeans_iters=current_kmeans_iters,
            max_k_clusters_per_q=max_k_clusters_per_q,
            min_kc_ratio=min_kc_ratio,
            init_q_centroids=init_q,
            init_k_centroids=init_k,
            enable_profiling=enable_profiling,
            layer_idx=self.layer_idx,
        )
        
        # Log profiling stats for layer 0
        if enable_profiling and profile_stats is not None:
            B, S, H, D = query.shape
            logger.info(
                f"[SVG2 Profile] Layer {self.layer_idx} | "
                f"Shape: B={B},S={S},H={H},D={D} | "
                f"Qc={num_q_clusters},Kc={num_k_clusters},top_p={top_p} | "
                f"Blocks: {profile_stats.get('active_blocks', 0)}/{profile_stats.get('total_blocks', 0)} "
                f"({profile_stats.get('block_retention_pct', 0):.1f}% kept) | "
                f"centroid_reuse={centroid_reuse}"
            )
            logger.info(
                f"[SVG2 Timing] "
                f"KMeans: {profile_stats.get('kmeans_ms', 0):.1f}ms | "
                f"MaskGen: {profile_stats.get('mask_gen_ms', 0):.1f}ms | "
                f"Permute: {profile_stats.get('permute_ms', 0):.1f}ms | "
                f"SparseAttn: {profile_stats.get('sparse_attn_ms', 0):.1f}ms | "
                f"InvPermute: {profile_stats.get('inv_permute_ms', 0):.1f}ms | "
                f"Total: {profile_stats.get('total_ms', 0):.1f}ms"
            )
        
        # Update cache
        self.q_centroids = new_q_centroids
        self.k_centroids = new_k_centroids
        self.centroids_initialized = True
        
        return output
        