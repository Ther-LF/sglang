#!/usr/bin/env python3
"""
SVG2 组件精度测试 - 逐步对比每个组件与标准实现的差异

这个脚本将逐一测试：
1. K-Means 聚类
2. 动态块掩码生成 (identify_dynamic_map/mask)
3. Permutation (排列)
4. Inverse Permutation (逆排列)
5. Block Sparse Attention (块稀疏注意力)

Usage:
    cd /Users/luofan/Desktop/sglang
    python test_svg2_components.py
"""

import sys
import os
import torch
import torch.nn.functional as F

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
sys.path.insert(0, "/Users/luofan/Desktop/Sparse-VideoGen")

# ============================================================================
# 导入两边的实现
# ============================================================================

# SGLang 实现 (你的)
from sglang.multimodal_gen.runtime.layers.attention.backends.svg2_sparse_attn import (
    triton_kmeans as sglang_kmeans,
    identify_dynamic_mask as sglang_identify_mask,
    permute_by_labels as sglang_permute,
    inverse_permute as sglang_inverse_permute,
    block_sparse_attention as sglang_block_sparse_attn,
    svg2_attention_forward as sglang_svg2_forward,
)

# 标准 SVG2 实现
from svg.kmeans_utils import (
    batch_kmeans_Euclid as svg_kmeans,
    identify_dynamic_map as svg_identify_map,
    dynamic_block_sparse_fwd_torch as svg_block_sparse_torch,  # PyTorch 参考版本
)
from svg.kernels.triton.permute import (
    permute_tensor_by_labels_triton as svg_permute,
    apply_inverse_permutation_triton as svg_inverse_permute,
)


def compute_errors(output, reference, name=""):
    """计算误差指标"""
    output = output.float()
    reference = reference.float()
    
    abs_diff = torch.abs(output - reference)
    
    # L2 相对误差
    ref_norm = torch.norm(reference).item()
    error_norm = torch.norm(abs_diff).item()
    l2_rel_pct = (error_norm / ref_norm) * 100 if ref_norm > 0 else 0
    
    # 余弦相似度
    cos_sim = F.cosine_similarity(
        output.flatten(), reference.flatten(), dim=0
    ).item()
    
    # 最大绝对误差
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    
    print(f"  [{name}] Max Abs: {max_abs:.6f}, Mean Abs: {mean_abs:.6f}, "
          f"L2 Rel: {l2_rel_pct:.2f}%, Cosine: {cos_sim:.6f}")
    
    return {
        'max_abs': max_abs,
        'mean_abs': mean_abs,
        'l2_rel_pct': l2_rel_pct,
        'cosine_sim': cos_sim,
    }


def dense_attention(q, k, v):
    """参考 dense attention，输入 [B, H, S, D]"""
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return out


# ============================================================================
# 测试 1: K-Means 聚类
# ============================================================================
def test_kmeans(
    batch_size=1,
    num_heads=4,
    seq_len=1024,
    dim=64,
    num_clusters=64,
    max_iters=10,
    dtype=torch.bfloat16,
):
    """对比两边的 K-Means 实现"""
    print("\n" + "="*80)
    print("TEST 1: K-Means Clustering")
    print("="*80)
    
    device = 'cuda'
    torch.manual_seed(42)
    
    # 生成测试数据 - 标准 SVG 使用 [B*H, S, D] 格式
    x = torch.randn(batch_size * num_heads, seq_len, dim, dtype=dtype, device=device)
    
    print(f"Input shape: {x.shape}")
    print(f"Clusters: {num_clusters}, Max iters: {max_iters}")
    
    # 标准 SVG K-Means
    print("\n[SVG Standard] Running K-Means...")
    svg_labels, svg_centroids, svg_sizes, svg_iters = svg_kmeans(
        x, n_clusters=num_clusters, max_iters=max_iters
    )
    print(f"  Labels shape: {svg_labels.shape}")
    print(f"  Centroids shape: {svg_centroids.shape}")
    print(f"  Cluster sizes: min={svg_sizes.min().item()}, max={svg_sizes.max().item()}")
    
    # SGLang K-Means (你的实现)
    print("\n[SGLang] Running K-Means...")
    sglang_labels, sglang_centroids, sglang_sizes = sglang_kmeans(
        x, n_clusters=num_clusters, max_iters=max_iters
    )
    print(f"  Labels shape: {sglang_labels.shape}")
    print(f"  Centroids shape: {sglang_centroids.shape}")
    print(f"  Cluster sizes: min={sglang_sizes.min().item()}, max={sglang_sizes.max().item()}")
    
    # 对比 - 注意：K-Means 结果可能不完全一致（随机初始化等），
    # 但聚类质量应该相近
    print("\n[Comparison]")
    
    # 对比 cluster size 分布
    svg_sizes_sorted = svg_sizes.sort(dim=-1)[0].float()
    sglang_sizes_sorted = sglang_sizes.sort(dim=-1)[0].float()
    size_diff = (svg_sizes_sorted - sglang_sizes_sorted).abs().mean().item()
    print(f"  Cluster size distribution diff (sorted): {size_diff:.4f}")
    
    # 对比 centroids 的统计特性
    svg_cent_norm = torch.norm(svg_centroids.float(), dim=-1).mean().item()
    sglang_cent_norm = torch.norm(sglang_centroids.float(), dim=-1).mean().item()
    print(f"  SVG centroid norm mean: {svg_cent_norm:.4f}")
    print(f"  SGLang centroid norm mean: {sglang_cent_norm:.4f}")
    
    return svg_labels, sglang_labels, svg_centroids, sglang_centroids, svg_sizes, sglang_sizes


# ============================================================================
# 测试 2: 动态块掩码生成
# ============================================================================
def test_identify_dynamic_mask(
    batch_size=1,
    num_heads=4,
    num_q_clusters=64,
    num_k_clusters=64,
    dim=64,
    top_p=0.5,
    min_kc_ratio=0.0,
    dtype=torch.bfloat16,
):
    """对比动态块掩码生成"""
    print("\n" + "="*80)
    print("TEST 2: Dynamic Block Mask Generation (identify_dynamic_map/mask)")
    print("="*80)
    
    device = 'cuda'
    torch.manual_seed(42)
    
    # 生成测试 centroids 和 cluster sizes
    # 使用相同的输入来确保可比性
    q_centroids = torch.randn(batch_size, num_heads, num_q_clusters, dim, dtype=dtype, device=device)
    k_centroids = torch.randn(batch_size, num_heads, num_k_clusters, dim, dtype=dtype, device=device)
    
    # Cluster sizes - 随机但总和一致
    seq_len = 1024
    q_sizes = torch.ones(batch_size, num_heads, num_q_clusters, dtype=torch.int64, device=device)
    q_sizes[..., :seq_len % num_q_clusters] += seq_len // num_q_clusters
    q_sizes[..., seq_len % num_q_clusters:] = seq_len // num_q_clusters
    
    k_sizes = torch.ones(batch_size, num_heads, num_k_clusters, dtype=torch.int64, device=device)
    k_sizes[..., :seq_len % num_k_clusters] += seq_len // num_k_clusters
    k_sizes[..., seq_len % num_k_clusters:] = seq_len // num_k_clusters
    
    print(f"Q centroids shape: {q_centroids.shape}")
    print(f"K centroids shape: {k_centroids.shape}")
    print(f"top_p: {top_p}, min_kc_ratio: {min_kc_ratio}")
    
    # 标准 SVG
    print("\n[SVG Standard] identify_dynamic_map...")
    svg_mask = svg_identify_map(
        q_centroids, k_centroids,
        q_sizes, k_sizes,
        p=top_p,
        min_kc_ratio=min_kc_ratio,
    )
    svg_active = svg_mask.sum().item()
    svg_sparsity = 1.0 - svg_active / svg_mask.numel()
    print(f"  Mask shape: {svg_mask.shape}")
    print(f"  Active blocks: {svg_active}/{svg_mask.numel()} ({100*(1-svg_sparsity):.2f}%)")
    
    # SGLang (你的实现)
    print("\n[SGLang] identify_dynamic_mask...")
    sglang_mask = sglang_identify_mask(
        q_centroids, k_centroids,
        q_sizes, k_sizes,
        top_p=top_p,
        min_kc_ratio=min_kc_ratio,
    )
    sglang_active = sglang_mask.sum().item()
    sglang_sparsity = 1.0 - sglang_active / sglang_mask.numel()
    print(f"  Mask shape: {sglang_mask.shape}")
    print(f"  Active blocks: {sglang_active}/{sglang_mask.numel()} ({100*(1-sglang_sparsity):.2f}%)")
    
    # 对比
    print("\n[Comparison]")
    mask_match = (svg_mask == sglang_mask).sum().item()
    mask_total = svg_mask.numel()
    match_pct = 100.0 * mask_match / mask_total
    print(f"  Mask agreement: {mask_match}/{mask_total} ({match_pct:.2f}%)")
    
    # 检查哪些位置不同
    diff_mask = svg_mask != sglang_mask
    if diff_mask.any():
        num_diff = diff_mask.sum().item()
        print(f"  ⚠️  {num_diff} positions differ!")
        # 分析差异
        svg_only = (svg_mask & ~sglang_mask).sum().item()
        sglang_only = (~svg_mask & sglang_mask).sum().item()
        print(f"    SVG active but SGLang inactive: {svg_only}")
        print(f"    SGLang active but SVG inactive: {sglang_only}")
    else:
        print(f"  ✓ Masks are identical!")
    
    return svg_mask, sglang_mask


# ============================================================================
# 测试 3: Permutation
# ============================================================================
def test_permutation(
    batch_size=1,
    num_heads=4,
    seq_len=1024,
    dim=64,
    num_clusters=64,
    dtype=torch.bfloat16,
):
    """对比 Permutation 实现"""
    print("\n" + "="*80)
    print("TEST 3: Permutation")
    print("="*80)
    
    device = 'cuda'
    torch.manual_seed(42)
    
    # 生成测试数据 - [B, H, S, D] 格式
    x = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    
    # 生成随机 labels - [B*H, S] 格式
    labels = torch.randint(0, num_clusters, (batch_size * num_heads, seq_len), device=device)
    
    print(f"Input shape: {x.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 标准 SVG permute - 注意它期望 [B, H, S, D] 和 [B*H, S]
    # 但 svg_permute 的 dim 参数是针对 [cfg, num_heads, seq_len, dim] 的
    print("\n[SVG Standard] permute_tensor_by_labels_triton...")
    # SVG 使用 dim=2 表示在 seq_len 维度上排列
    svg_out, svg_indices = svg_permute(x, labels, dim=2)
    print(f"  Output shape: {svg_out.shape}")
    print(f"  Indices shape: {svg_indices.shape}")
    
    # SGLang permute
    print("\n[SGLang] permute_by_labels...")
    sglang_out, sglang_indices = sglang_permute(x, labels=labels)
    print(f"  Output shape: {sglang_out.shape}")
    print(f"  Indices shape: {sglang_indices.shape}")
    
    # 对比
    print("\n[Comparison]")
    compute_errors(sglang_out, svg_out, "Permuted Output")
    
    # 检查 indices 是否一致
    indices_match = (svg_indices == sglang_indices).all().item()
    print(f"  Indices match: {'✓' if indices_match else '✗'}")
    
    return svg_out, sglang_out, svg_indices, sglang_indices


# ============================================================================
# 测试 4: Inverse Permutation
# ============================================================================
def test_inverse_permutation(
    batch_size=1,
    num_heads=4,
    seq_len=1024,
    dim=64,
    num_clusters=64,
    dtype=torch.bfloat16,
):
    """对比 Inverse Permutation 实现"""
    print("\n" + "="*80)
    print("TEST 4: Inverse Permutation")
    print("="*80)
    
    device = 'cuda'
    torch.manual_seed(42)
    
    # 生成测试数据
    x = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    labels = torch.randint(0, num_clusters, (batch_size * num_heads, seq_len), device=device)
    
    print(f"Input shape: {x.shape}")
    
    # 先做 permutation
    svg_perm, svg_indices = svg_permute(x, labels, dim=2)
    sglang_perm, sglang_indices = sglang_permute(x, labels=labels)
    
    print("After permutation:")
    print(f"  SVG permuted shape: {svg_perm.shape}")
    print(f"  SGLang permuted shape: {sglang_perm.shape}")
    
    # 做 inverse permutation
    print("\n[SVG Standard] apply_inverse_permutation_triton...")
    svg_inv = svg_inverse_permute(svg_perm, svg_indices, dim=2)
    print(f"  Output shape: {svg_inv.shape}")
    
    print("\n[SGLang] inverse_permute...")
    sglang_inv = sglang_inverse_permute(sglang_perm, sglang_indices)
    print(f"  Output shape: {sglang_inv.shape}")
    
    # 对比
    print("\n[Comparison]")
    compute_errors(sglang_inv, svg_inv, "Inverse Permuted Output")
    
    # 检查是否恢复原始输入
    svg_recover = compute_errors(svg_inv, x, "SVG Recovery (vs original)")
    sglang_recover = compute_errors(sglang_inv, x, "SGLang Recovery (vs original)")
    
    return svg_inv, sglang_inv


# ============================================================================
# 测试 5: Block Sparse Attention (使用 PyTorch 参考实现)
# ============================================================================
def test_block_sparse_attention(
    batch_size=1,
    num_heads=4,
    seq_len=1024,
    dim=64,
    num_q_clusters=64,
    num_k_clusters=64,
    top_p=0.5,
    dtype=torch.bfloat16,
):
    """对比 Block Sparse Attention 实现"""
    print("\n" + "="*80)
    print("TEST 5: Block Sparse Attention")
    print("="*80)
    
    device = 'cuda'
    torch.manual_seed(42)
    
    # 生成已排列的 Q, K, V (模拟 K-Means 后的结果)
    q = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    
    # 生成 cluster sizes - 均匀分布
    tokens_per_q = seq_len // num_q_clusters
    tokens_per_k = seq_len // num_k_clusters
    
    q_sizes = torch.full((batch_size, num_heads, num_q_clusters), tokens_per_q, 
                         dtype=torch.int64, device=device)
    k_sizes = torch.full((batch_size, num_heads, num_k_clusters), tokens_per_k,
                         dtype=torch.int64, device=device)
    
    # 处理余数
    remainder_q = seq_len % num_q_clusters
    remainder_k = seq_len % num_k_clusters
    q_sizes[..., :remainder_q] += 1
    k_sizes[..., :remainder_k] += 1
    
    print(f"Q/K/V shape: {q.shape}")
    print(f"Q clusters: {num_q_clusters}, K clusters: {num_k_clusters}")
    print(f"Q cluster sizes: {q_sizes[0, 0, :5].tolist()}... (sum={q_sizes[0,0].sum().item()})")
    print(f"K cluster sizes: {k_sizes[0, 0, :5].tolist()}... (sum={k_sizes[0,0].sum().item()})")
    
    # 生成 block mask
    torch.manual_seed(42)  # 确保一致
    q_centroids = torch.randn(batch_size, num_heads, num_q_clusters, dim, dtype=dtype, device=device)
    k_centroids = torch.randn(batch_size, num_heads, num_k_clusters, dim, dtype=dtype, device=device)
    
    block_mask = svg_identify_map(
        q_centroids, k_centroids,
        q_sizes, k_sizes,
        p=top_p,
        min_kc_ratio=0.0,
    )
    
    active_blocks = block_mask.sum().item()
    total_blocks = block_mask.numel()
    print(f"Block mask: {active_blocks}/{total_blocks} active ({100*active_blocks/total_blocks:.2f}%)")
    
    # 计算 dense attention 作为参考
    print("\n[Reference] Dense Attention...")
    with torch.no_grad():
        dense_out = dense_attention(q, k, v)
    print(f"  Output shape: {dense_out.shape}")
    
    # SVG PyTorch 参考实现 (慢但正确)
    print("\n[SVG Standard] PyTorch Block Sparse Attention...")
    with torch.no_grad():
        svg_out = svg_block_sparse_torch(q, k, v, block_mask, q_sizes, k_sizes)
    print(f"  Output shape: {svg_out.shape}")
    
    # SGLang 实现 (你的 Triton 版本)
    print("\n[SGLang] Triton Block Sparse Attention...")
    with torch.no_grad():
        sglang_out = sglang_block_sparse_attn(q, k, v, block_mask, q_sizes, k_sizes)
    print(f"  Output shape: {sglang_out.shape}")
    
    # 对比
    print("\n[Comparison]")
    print("SVG vs Dense:")
    compute_errors(svg_out, dense_out, "SVG Sparse vs Dense")
    
    print("\nSGLang vs Dense:")
    compute_errors(sglang_out, dense_out, "SGLang Sparse vs Dense")
    
    print("\nSGLang vs SVG (should be very close if implementations match):")
    compute_errors(sglang_out, svg_out, "SGLang vs SVG Sparse")
    
    return dense_out, svg_out, sglang_out


# ============================================================================
# 测试 6: 完整 SVG2 流程 (不含 FlashInfer)
# ============================================================================
def test_full_svg2_pipeline(
    batch_size=1,
    seq_len=2048,
    num_heads=8,
    head_dim=64,
    num_q_clusters=64,
    num_k_clusters=64,
    top_p=0.5,
    kmeans_iters=10,
    dtype=torch.bfloat16,
):
    """测试完整的 SVG2 流程，使用 PyTorch 参考版本做对比"""
    print("\n" + "="*80)
    print("TEST 6: Full SVG2 Pipeline (with PyTorch reference)")
    print("="*80)
    
    device = 'cuda'
    torch.manual_seed(42)
    
    # 生成测试数据 [B, S, H, D] (SGLang 格式)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    
    print(f"Input shape: {q.shape}")
    print(f"Clusters: Qc={num_q_clusters}, Kc={num_k_clusters}")
    print(f"top_p: {top_p}, kmeans_iters: {kmeans_iters}")
    
    # Dense reference
    print("\n[Reference] Dense Attention...")
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    with torch.no_grad():
        dense_out = dense_attention(q_t, k_t, v_t)
        dense_out = dense_out.transpose(1, 2)  # 回到 [B, S, H, D]
    print(f"  Output shape: {dense_out.shape}")
    
    # SGLang SVG2
    print("\n[SGLang] SVG2 Forward...")
    with torch.no_grad():
        sglang_out, _, _ = sglang_svg2_forward(
            q, k, v,
            num_q_clusters=num_q_clusters,
            num_k_clusters=num_k_clusters,
            top_p=top_p,
            kmeans_iters=kmeans_iters,
        )
    print(f"  Output shape: {sglang_out.shape}")
    
    # 手动实现 SVG2 流程（用 SVG 组件 + PyTorch sparse attention）
    print("\n[SVG Components + PyTorch Sparse Attn]...")
    with torch.no_grad():
        # 转换为 [B, H, S, D]
        q_bhsd = q.transpose(1, 2).contiguous()
        k_bhsd = k.transpose(1, 2).contiguous()
        v_bhsd = v.transpose(1, 2).contiguous()
        
        # K-Means
        q_flat = q_bhsd.reshape(batch_size * num_heads, seq_len, head_dim)
        k_flat = k_bhsd.reshape(batch_size * num_heads, seq_len, head_dim)
        
        q_labels, q_centroids, q_sizes, _ = svg_kmeans(q_flat, n_clusters=num_q_clusters, max_iters=kmeans_iters)
        k_labels, k_centroids, k_sizes, _ = svg_kmeans(k_flat, n_clusters=num_k_clusters, max_iters=kmeans_iters)
        
        # Reshape
        q_centroids = q_centroids.reshape(batch_size, num_heads, num_q_clusters, head_dim)
        k_centroids = k_centroids.reshape(batch_size, num_heads, num_k_clusters, head_dim)
        q_sizes = q_sizes.reshape(batch_size, num_heads, num_q_clusters)
        k_sizes = k_sizes.reshape(batch_size, num_heads, num_k_clusters)
        
        # Block mask
        block_mask = svg_identify_map(q_centroids, k_centroids, q_sizes, k_sizes, p=top_p)
        
        # Permutation
        q_perm, q_idx = svg_permute(q_bhsd, q_labels, dim=2)
        k_perm, k_idx = svg_permute(k_bhsd, k_labels, dim=2)
        v_perm, _ = svg_permute(v_bhsd, k_labels, dim=2, sorted_indices=k_idx)
        
        # Block sparse attention (PyTorch reference)
        out_perm = svg_block_sparse_torch(q_perm, k_perm, v_perm, block_mask, q_sizes, k_sizes)
        
        # Inverse permutation
        svg_ref_out = svg_inverse_permute(out_perm, q_idx, dim=2)
        svg_ref_out = svg_ref_out.transpose(1, 2)  # 回到 [B, S, H, D]
    
    print(f"  Output shape: {svg_ref_out.shape}")
    
    # 对比
    print("\n[Comparison]")
    print("SGLang SVG2 vs Dense:")
    compute_errors(sglang_out, dense_out, "SGLang vs Dense")
    
    print("\nSVG Reference vs Dense:")
    compute_errors(svg_ref_out, dense_out, "SVG Ref vs Dense")
    
    print("\nSGLang vs SVG Reference (implementations should match):")
    compute_errors(sglang_out, svg_ref_out, "SGLang vs SVG Ref")
    
    return dense_out, sglang_out, svg_ref_out


# ============================================================================
# Main
# ============================================================================
def main():
    print("="*80)
    print("SVG2 Component-by-Component Precision Test")
    print("="*80)
    
    # 基本配置
    batch_size = 1
    num_heads = 4
    seq_len = 1024
    dim = 64
    num_q_clusters = 32
    num_k_clusters = 32
    top_p = 0.5
    dtype = torch.bfloat16
    
    results = {}
    
    # 测试 1: K-Means
    try:
        test_kmeans(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            dim=dim,
            num_clusters=num_q_clusters,
            dtype=dtype,
        )
        results['kmeans'] = 'PASS'
    except Exception as e:
        print(f"ERROR in test_kmeans: {e}")
        results['kmeans'] = f'FAIL: {e}'
        import traceback
        traceback.print_exc()
    
    # 测试 2: Dynamic Mask
    try:
        test_identify_dynamic_mask(
            batch_size=batch_size,
            num_heads=num_heads,
            num_q_clusters=num_q_clusters,
            num_k_clusters=num_k_clusters,
            dim=dim,
            top_p=top_p,
            dtype=dtype,
        )
        results['dynamic_mask'] = 'PASS'
    except Exception as e:
        print(f"ERROR in test_identify_dynamic_mask: {e}")
        results['dynamic_mask'] = f'FAIL: {e}'
        import traceback
        traceback.print_exc()
    
    # 测试 3: Permutation
    try:
        test_permutation(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            dim=dim,
            num_clusters=num_q_clusters,
            dtype=dtype,
        )
        results['permutation'] = 'PASS'
    except Exception as e:
        print(f"ERROR in test_permutation: {e}")
        results['permutation'] = f'FAIL: {e}'
        import traceback
        traceback.print_exc()
    
    # 测试 4: Inverse Permutation
    try:
        test_inverse_permutation(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            dim=dim,
            num_clusters=num_q_clusters,
            dtype=dtype,
        )
        results['inverse_permutation'] = 'PASS'
    except Exception as e:
        print(f"ERROR in test_inverse_permutation: {e}")
        results['inverse_permutation'] = f'FAIL: {e}'
        import traceback
        traceback.print_exc()
    
    # 测试 5: Block Sparse Attention
    try:
        test_block_sparse_attention(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            dim=dim,
            num_q_clusters=num_q_clusters,
            num_k_clusters=num_k_clusters,
            top_p=top_p,
            dtype=dtype,
        )
        results['block_sparse_attn'] = 'PASS'
    except Exception as e:
        print(f"ERROR in test_block_sparse_attention: {e}")
        results['block_sparse_attn'] = f'FAIL: {e}'
        import traceback
        traceback.print_exc()
    
    # 测试 6: Full Pipeline
    try:
        test_full_svg2_pipeline(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=dim,
            num_q_clusters=num_q_clusters,
            num_k_clusters=num_k_clusters,
            top_p=top_p,
            dtype=dtype,
        )
        results['full_pipeline'] = 'PASS'
    except Exception as e:
        print(f"ERROR in test_full_svg2_pipeline: {e}")
        results['full_pipeline'] = f'FAIL: {e}'
        import traceback
        traceback.print_exc()
    
    # 总结
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for test_name, result in results.items():
        status = "✓" if result == 'PASS' else "✗"
        print(f"  {status} {test_name}: {result}")
    print("="*80)


if __name__ == '__main__':
    main()

