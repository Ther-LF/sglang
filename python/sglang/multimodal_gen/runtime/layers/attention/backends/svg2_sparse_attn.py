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
    Optimized to minimize memory allocations.
    
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
    
    # Flatten batch dimension for simpler kernel dispatch if possible, 
    # but here we process batch-by-batch or use a batched kernel.
    # Current kernels are unbatched (process one set of points).
    # To reduce python overhead, ideally we would make kernels batched, 
    # but for now let's optimize memory allocation.
    
    x_flat = x.reshape(B * N, D).contiguous()
    
    # Initialize centroids
    if init_centroids is not None:
        centroids = init_centroids.reshape(B * K, D).clone().float()
    else:
        # Random initialization
        indices = torch.randint(0, N, (B, K), device=device)
        batch_offset = torch.arange(B, device=device)[:, None] * N
        flat_indices = (batch_offset + indices).flatten()
        centroids = x_flat[flat_indices].float().clone()
    
    centroids = centroids.reshape(B, K, D)
    labels = torch.zeros(B, N, dtype=torch.int32, device=device)
    
    # Pre-allocate buffers for the loop to avoid malloc overhead
    dist_buffer = torch.empty(N, K, dtype=torch.float32, device=device)
    centroid_sum_buffer = torch.empty(K, D, dtype=torch.float32, device=device)
    centroid_count_buffer = torch.empty(K, dtype=torch.int32, device=device)
    x_sqnorm_buffer = torch.empty(N, dtype=torch.float32, device=device)
    c_sqnorm_buffer = torch.empty(K, dtype=torch.float32, device=device)
    
    # Kernel config
    BLOCK_N = 128 # Increased from 32 for better occupancy
    BLOCK_K = min(128, K) # Increased
    BLOCK_D = min(128, triton.next_power_of_2(D))
    
    grid_n = triton.cdiv(N, BLOCK_N)
    grid_k = triton.cdiv(K, BLOCK_K)
    
    for iteration in range(max_iters):
        # Process each batch
        # TODO: Ideally fuse batch dimension into kernel grid to avoid Python loop
        for b in range(B):
            x_b = x[b] # [N, D]
            c_b = centroids[b] # [K, D]
            
            # Precompute squared norms
            # torch.sum is fast and optimized
            torch.sum(x_b * x_b, dim=-1, out=x_sqnorm_buffer)
            torch.sum(c_b * c_b, dim=-1, out=c_sqnorm_buffer)
            
            # Step 1: Compute distances
            _pairwise_distance_kernel[(grid_n, grid_k)](
                x_b, c_b, x_sqnorm_buffer, c_sqnorm_buffer, dist_buffer,
                N, K, D,
                BLOCK_N, BLOCK_K, BLOCK_D,
            )
            
            # Step 2: Assign clusters
            _assign_clusters_kernel[(grid_n,)](
                dist_buffer, labels[b],
                N, K, BLOCK_N,
            )
            
            # Step 3: Update centroids
            # Zero out buffers
            centroid_sum_buffer.zero_()
            centroid_count_buffer.zero_()
            
            _update_centroids_kernel[(N,)](
                x_b, labels[b], centroid_sum_buffer, centroid_count_buffer,
                N, D, K, BLOCK_D,
            )
            
            # Update centroids
            # Avoid division by zero
            centroid_count_safe = centroid_count_buffer.clamp(min=1)
            centroids[b] = centroid_sum_buffer / centroid_count_safe[:, None]
    
    # Compute final cluster sizes
    cluster_sizes = torch.zeros(B, K, dtype=torch.int32, device=device)
    for b in range(B):
        # Using bincount is faster than python loop
        # labels[b] is [N], values in [0, K-1]
        counts = torch.bincount(labels[b].long(), minlength=K)
        cluster_sizes[b] = counts.to(torch.int32)
    
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
    if min_kc_ratio > 0:
        preserve_length = int(min_kc_ratio * Kk)
        remove_indices[..., :preserve_length] = False
        
    sorted_clusters_to_keep = ~remove_indices
    
    # 9. Map back to original indices
    block_mask = torch.zeros((B, H, Kq, Kk), device=device, dtype=torch.bool)
    block_mask.scatter_(-1, sorted_indices, sorted_clusters_to_keep)
    
    return block_mask


# ============================================================================
# Part 4: Triton Block Sparse Attention Kernel
# ============================================================================


@triton.jit
def _sparse_attn_csr_kernel(
    Q, K, V, Out,
    # CSR Indices
    kv_indices_ptr, # [NNZ] int32
    q_indptr_ptr,   # [B*H*QC + 1] int32
    # Cumulative Sizes
    qc_cum_size, kc_cum_size,
    # Strides
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_qcs_b, stride_qcs_h, stride_qcs_qc,
    stride_kcs_b, stride_kcs_h, stride_kcs_kc,
    # Meta
    H, QC_NUM, 
    sm_scale,
    # Block Constants
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # pid 对应一个具体的 Q 簇 (展平后的视图: batch * head * q_cluster)
    pid = tl.program_id(0)
    
    # 反解坐标
    q_block_idx = pid % QC_NUM
    pid_bh = pid // QC_NUM
    h = pid_bh % H
    b = pid_bh // H
    
    # --- 1. 获取任务范围 (CSR Planning 的结果) ---
    # q_indptr 告诉我们：在这个 CSR 数组里，属于我这个 pid 的 K 块是从哪到哪
    csr_start = tl.load(q_indptr_ptr + pid)
    csr_end = tl.load(q_indptr_ptr + pid + 1)
    
    # 如果没有 K 块要处理，直接返回 (Zero-overhead for empty rows)
    if csr_end == csr_start:
        return

    # --- 2. Q Block 定位 ---
    qcs_offset = b * stride_qcs_b + h * stride_qcs_h
    q_start = tl.load(qc_cum_size + qcs_offset + q_block_idx * stride_qcs_qc)
    q_end = tl.load(qc_cum_size + qcs_offset + (q_block_idx + 1) * stride_qcs_qc)
    q_len = q_end - q_start
    
    if q_len == 0: return

    # 指针初始化
    q_base = Q + b * stride_qb + h * stride_qh + q_start * stride_qs
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    out_base = Out + b * stride_ob + h * stride_oh + q_start * stride_os
    
    kcs_base_ptr = kc_cum_size + b * stride_kcs_b + h * stride_kcs_h

    # --- 3. Q 切分循环 ---
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    for q_chunk_start in range(0, q_len, BLOCK_M):
        q_rows = q_chunk_start + offs_m
        q_mask = q_rows < q_len
        
        # 加载 Q Chunk
        q_ptr = q_base + q_rows[:, None] * stride_qs + offs_d[None, :]
        q_chunk = tl.load(q_ptr, mask=q_mask[:, None] & (offs_d[None, :] < BLOCK_D), other=0.0)
        
        # Online Softmax 累加器
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        
        # --- 4. K 块遍历 (基于 CSR 索引) ---
        # 这里的循环次数 = 活跃 K 块的数量，没有浪费！
        for csr_idx in range(csr_start, csr_end):
            # 获取真实的 K Block ID
            k_block_idx = tl.load(kv_indices_ptr + csr_idx)
            
            # 获取 K Block 长度信息
            k_start = tl.load(kcs_base_ptr + k_block_idx * stride_kcs_kc)
            k_end = tl.load(kcs_base_ptr + (k_block_idx + 1) * stride_kcs_kc)
            k_len = k_end - k_start
            
            if k_len > 0:
                k_chunk_ptr = k_base + k_start * stride_ks
                v_chunk_ptr = v_base + k_start * stride_vs
                
                # --- 5. K 切分循环 (FlashAttention Core) ---
                offs_n = tl.arange(0, BLOCK_N)
                for k_chunk_start in range(0, k_len, BLOCK_N):
                    k_rows = k_chunk_start + offs_n
                    k_mask = k_rows < k_len
                    
                    k_ptr = k_chunk_ptr + k_rows[:, None] * stride_ks + offs_d[None, :]
                    v_ptr = v_chunk_ptr + k_rows[:, None] * stride_vs + offs_d[None, :]
                    
                    # Ensure we don't read out of bounds for D dim
                    load_mask = k_mask[:, None] & (offs_d[None, :] < BLOCK_D)
                    k_chunk = tl.load(k_ptr, mask=load_mask, other=0.0)
                    v_chunk = tl.load(v_ptr, mask=load_mask, other=0.0)
                    
                    # QK^T
                    qk = tl.dot(q_chunk, tl.trans(k_chunk))
                    qk *= sm_scale
                    
                    # Masking (inf) - critical for padded K rows
                    qk = tl.where(q_mask[:, None] & k_mask[None, :], qk, float("-inf"))
                    
                    # Online Softmax
                    m_curr = tl.max(qk, 1)
                    m_new = tl.maximum(m_i, m_curr)
                    
                    # p = exp(qk - m_new)
                    p = tl.exp(qk - m_new[:, None])
                    
                    # alpha = exp(m_prev - m_new)
                    alpha = tl.exp(m_i - m_new)
                    
                    # Update l_i
                    l_i = l_i * alpha + tl.sum(p, 1)
                    m_i = m_new
                    
                    # Update accumulator
                    # Check dtype for p @ v
                    p = p.to(v_chunk.dtype)
                    acc = acc * alpha[:, None] + tl.dot(p, v_chunk)

        # --- 6. 写回 Output ---
        # Normalize: out = acc / l_i
        # Avoid div by zero
        l_i_safe = tl.where(l_i == 0, 1.0, l_i)
        acc = acc / l_i_safe[:, None]
        # If l_i was 0, result should be 0
        acc = tl.where(l_i[:, None] == 0, 0.0, acc)
        
        out_ptr = out_base + q_rows[:, None] * stride_os + offs_d[None, :]
        tl.store(out_ptr, acc.to(Out.dtype.element_ty), mask=q_mask[:, None] & (offs_d[None, :] < BLOCK_D))


def block_sparse_attention(
    q: torch.Tensor,           # [B, H, S, D]
    k: torch.Tensor,           # [B, H, S, D]
    v: torch.Tensor,           # [B, H, S, D]
    block_mask: torch.Tensor,  # [B, H, Kq, Kk]
    q_cluster_sizes: torch.Tensor,  # [B, H, Kq]
    k_cluster_sizes: torch.Tensor,  # [B, H, Kk]
) -> torch.Tensor:
    """
    Block sparse attention with variable block sizes using Triton CSR kernel.
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
    # Add a 0 at the start for offsets
    qc_cum_size = torch.zeros((B, H, qc_num + 1), device=q.device, dtype=torch.int32)
    qc_cum_size[..., 1:] = torch.cumsum(q_cluster_sizes, dim=-1)
    
    kc_cum_size = torch.zeros((B, H, kc_num + 1), device=q.device, dtype=torch.int32)
    kc_cum_size[..., 1:] = torch.cumsum(k_cluster_sizes, dim=-1)
    
    # --- CSR Planning ---
    # Convert dense mask [B, H, Qc, Kc] to CSR format
    # Flatten to [Rows, Cols] where Rows = B*H*Qc
    flat_mask = block_mask.view(-1, kc_num)
    
    # Convert to Sparse CSR Tensor
    # Note: Input to to_sparse_csr must be float/int/complex, bool not always supported directly
    sparse_mask = flat_mask.float().to_sparse_csr()
    
    # Extract CSR components
    q_indptr = sparse_mask.crow_indices().int()
    kv_indices = sparse_mask.col_indices().int()
    
    # Output tensor
    out = torch.empty_like(q)
    
    # Triton kernel config
    BLOCK_D = triton.next_power_of_2(D)
    
    if S <= 512:
        BLOCK_M = 32
        BLOCK_N = 32
    elif S <= 1024:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 64 # Tuned down from 128 for safety
        BLOCK_N = 64
        
    BLOCK_M = min(BLOCK_M, S)
    BLOCK_N = min(BLOCK_N, S)
    
    # Launch grid: One program per query block per batch/head
    # Note: Using flattened QC_NUM dimension
    grid = (B * H * qc_num,)
    
    _sparse_attn_csr_kernel[grid](
        q, k, v, out,
        kv_indices, q_indptr,
        qc_cum_size, kc_cum_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        qc_cum_size.stride(0), qc_cum_size.stride(1), qc_cum_size.stride(2),
        kc_cum_size.stride(0), kc_cum_size.stride(1), kc_cum_size.stride(2),
        H, qc_num,
        scale,
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
        max_k_clusters_per_q: Optional[int] = None,
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
        self.max_k_clusters_per_q = max_k_clusters_per_q
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
        
        # Check if we should use full attention (early layers or early timesteps)
        if self._should_use_full_attention(timestep):
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



