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

# 自动检测 Sparse-VideoGen 路径
# 优先级: 环境变量 > 同级目录 > 常用路径
SVG_PATHS = [
    os.environ.get("SVG_PATH", ""),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Sparse-VideoGen")),  # 同级目录
    os.path.expanduser("~/Sparse-VideoGen"),  # Home 目录
    "/root/Sparse-VideoGen",  # 服务器常用路径
    "/Users/luofan/Desktop/Sparse-VideoGen",  # Mac 路径
]
SVG_PATH = None
for p in SVG_PATHS:
    if p and os.path.exists(os.path.join(p, "svg")):
        SVG_PATH = os.path.abspath(p)  # 规范化路径
        break

if SVG_PATH is None:
    print("ERROR: Could not find Sparse-VideoGen directory!")
    print("Please set SVG_PATH environment variable, e.g.:")
    print("  export SVG_PATH=/path/to/Sparse-VideoGen")
    print("Checked paths:")
    for p in SVG_PATHS:
        if p:
            print(f"  - {p}")
    sys.exit(1)

print(f"Using Sparse-VideoGen from: {SVG_PATH}")
sys.path.insert(0, SVG_PATH)

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

# ============================================================================
# PyTorch 参考实现作为 Ground Truth（绝对正确的实现）
# 用于验证 SGLang Triton 实现的正确性
# ============================================================================

def pytorch_ref_permute(tensor, labels, dim, sorted_indices=None):
    """
    PyTorch 参考实现：按 labels 排序 permute tensor。
    这是 ground truth，用于验证 Triton 实现的正确性。
    
    tensor: [B, H, S, D]
    labels: [B*H, S]
    dim: 必须是 2
    """
    assert dim == 2, "Only dim=2 is supported"
    B, H, S, D = tensor.shape
    BH = B * H
    
    if sorted_indices is None:
        sorted_indices = torch.argsort(labels, dim=-1)
    
    # Flatten and permute
    tensor_flat = tensor.reshape(BH, S, D)
    
    # Expand indices for gather
    gather_idx = sorted_indices.unsqueeze(-1).expand(BH, S, D).long()
    permuted_flat = torch.gather(tensor_flat, 1, gather_idx)
    
    return permuted_flat.reshape(B, H, S, D), sorted_indices

def pytorch_ref_inverse_permute(permuted_tensor, sorted_indices, dim):
    """
    PyTorch 参考实现：逆 permute。
    这是 ground truth。
    """
    assert dim == 2, "Only dim=2 is supported"
    B, H, S, D = permuted_tensor.shape
    BH = B * H
    
    # Compute inverse indices
    inverse_indices = torch.argsort(sorted_indices.long(), dim=-1)
    
    # Flatten and inverse permute
    tensor_flat = permuted_tensor.reshape(BH, S, D)
    gather_idx = inverse_indices.unsqueeze(-1).expand(BH, S, D)
    original_flat = torch.gather(tensor_flat, 1, gather_idx)
    
    return original_flat.reshape(B, H, S, D)

print("Using PyTorch reference implementation as ground truth for permute functions")
print("Testing SGLang Triton implementations from svg2_sparse_attn.py")


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
# 测试 1: K-Means 聚类 (使用相同初始化)
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
    """对比两边的 K-Means 实现 - 使用相同的初始 centroids"""
    print("\n" + "="*80)
    print("TEST 1: K-Means Clustering (with identical initialization)")
    print("="*80)
    
    device = 'cuda'
    torch.manual_seed(42)
    
    # 生成测试数据 - 标准 SVG 使用 [B*H, S, D] 格式
    BH = batch_size * num_heads
    x = torch.randn(BH, seq_len, dim, dtype=dtype, device=device)
    
    print(f"Input shape: {x.shape}")
    print(f"Clusters: {num_clusters}, Max iters: {max_iters}")
    
    # 生成固定的初始 centroids (两边共用)
    torch.manual_seed(42)  # 确保可复现
    init_indices = torch.randint(0, seq_len, (BH, num_clusters), device=device)
    init_centroids = torch.gather(
        x, dim=1, 
        index=init_indices.unsqueeze(-1).expand(-1, -1, dim)
    )  # [BH, K, D]
    print(f"Initial centroids shape: {init_centroids.shape}")
    
    # 标准 SVG K-Means (使用相同初始化)
    print("\n[SVG Standard] Running K-Means with shared init...")
    svg_labels, svg_centroids, svg_sizes, svg_iters = svg_kmeans(
        x, n_clusters=num_clusters, max_iters=max_iters, init_centroids=init_centroids.clone()
    )
    print(f"  Labels shape: {svg_labels.shape}")
    print(f"  Centroids shape: {svg_centroids.shape}")
    print(f"  Cluster sizes: min={svg_sizes.min().item()}, max={svg_sizes.max().item()}")
    print(f"  Iterations: {svg_iters}")
    
    # SGLang K-Means (使用相同初始化)
    print("\n[SGLang] Running K-Means with shared init...")
    sglang_labels, sglang_centroids, sglang_sizes = sglang_kmeans(
        x, n_clusters=num_clusters, max_iters=max_iters, init_centroids=init_centroids.clone()
    )
    print(f"  Labels shape: {sglang_labels.shape}")
    print(f"  Centroids shape: {sglang_centroids.shape}")
    print(f"  Cluster sizes: min={sglang_sizes.min().item()}, max={sglang_sizes.max().item()}")
    
    # 对比 - 使用相同初始化，结果应该非常接近
    print("\n[Comparison]")
    
    # 对比 labels
    labels_match = (svg_labels == sglang_labels).float().mean().item() * 100
    print(f"  Labels match: {labels_match:.2f}%")
    
    # 对比 centroids
    compute_errors(sglang_centroids, svg_centroids, "Centroids")
    
    # 对比 cluster sizes
    sizes_match = (svg_sizes == sglang_sizes).float().mean().item() * 100
    print(f"  Cluster sizes match: {sizes_match:.2f}%")
    
    # 对比 cluster size 分布
    svg_sizes_sorted = svg_sizes.sort(dim=-1)[0].float()
    sglang_sizes_sorted = sglang_sizes.sort(dim=-1)[0].float()
    size_diff = (svg_sizes_sorted - sglang_sizes_sorted).abs().mean().item()
    print(f"  Cluster size distribution diff (sorted): {size_diff:.4f}")
    
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
    """
    对比 Permutation 实现
    - Ground Truth: PyTorch 参考实现（gather 操作，保证正确）
    - 被测试: SGLang Triton 实现
    """
    print("\n" + "="*80)
    print("TEST 3: Permutation (SGLang Triton vs PyTorch Reference)")
    print("="*80)
    
    device = 'cuda'
    torch.manual_seed(42)
    
    # 生成测试数据 - [B, H, S, D] 格式
    x = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    
    # 生成随机 labels - [B*H, S] 格式
    labels = torch.randint(0, num_clusters, (batch_size * num_heads, seq_len), device=device)
    
    print(f"Input shape: {x.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # PyTorch 参考实现 (Ground Truth)
    print("\n[Ground Truth] PyTorch Reference permute...")
    ref_out, ref_indices = pytorch_ref_permute(x, labels, dim=2)
    print(f"  Output shape: {ref_out.shape}")
    print(f"  Indices shape: {ref_indices.shape}")
    
    # SGLang Triton 实现 (被测试)
    print("\n[SGLang Triton] permute_by_labels...")
    sglang_out, sglang_indices = sglang_permute(x, labels=labels)
    print(f"  Output shape: {sglang_out.shape}")
    print(f"  Indices shape: {sglang_indices.shape}")
    
    # 对比
    print("\n[Comparison: SGLang vs Ground Truth]")
    compute_errors(sglang_out, ref_out, "Permuted Output")
    
    # 检查 indices 是否一致
    indices_match = (ref_indices.long() == sglang_indices.long()).all().item()
    print(f"  Indices match: {'✓' if indices_match else '✗'}")
    
    return ref_out, sglang_out, ref_indices, sglang_indices


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
    """
    对比 Inverse Permutation 实现
    - Ground Truth: PyTorch 参考实现
    - 被测试: SGLang Triton 实现
    """
    print("\n" + "="*80)
    print("TEST 4: Inverse Permutation (SGLang Triton vs PyTorch Reference)")
    print("="*80)
    
    device = 'cuda'
    torch.manual_seed(42)
    
    # 生成测试数据
    x = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    labels = torch.randint(0, num_clusters, (batch_size * num_heads, seq_len), device=device)
    
    print(f"Input shape: {x.shape}")
    
    # 先做 permutation (两者使用各自的实现)
    ref_perm, ref_indices = pytorch_ref_permute(x, labels, dim=2)
    sglang_perm, sglang_indices = sglang_permute(x, labels=labels)
    
    print("After permutation:")
    print(f"  Reference permuted shape: {ref_perm.shape}")
    print(f"  SGLang permuted shape: {sglang_perm.shape}")
    
    # 做 inverse permutation
    print("\n[Ground Truth] PyTorch Reference inverse_permute...")
    ref_inv = pytorch_ref_inverse_permute(ref_perm, ref_indices, dim=2)
    print(f"  Output shape: {ref_inv.shape}")
    
    print("\n[SGLang Triton] inverse_permute...")
    sglang_inv = sglang_inverse_permute(sglang_perm, sglang_indices)
    print(f"  Output shape: {sglang_inv.shape}")
    
    # 对比
    print("\n[Comparison: SGLang vs Ground Truth]")
    compute_errors(sglang_inv, ref_inv, "Inverse Permuted Output")
    
    # 检查是否恢复原始输入
    print("\n[Recovery Check]")
    compute_errors(ref_inv, x, "Reference Recovery (vs original)")
    compute_errors(sglang_inv, x, "SGLang Recovery (vs original)")
    
    return ref_inv, sglang_inv


# ============================================================================
# 测试 5: Block Sparse Attention
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
    """
    对比 Block Sparse Attention 实现
    - Dense Attention: 完整注意力计算 (最终精度参考)
    - SVG Standard: svg.kmeans_utils.dynamic_block_sparse_fwd_torch (标准 SVG 实现)
    - SGLang Triton: 你的 block_sparse_attention 实现 (被测试)
    """
    print("\n" + "="*80)
    print("TEST 5: Block Sparse Attention (SGLang Triton vs SVG Standard)")
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
# 测试 6: 完整 SVG2 流程 (使用相同的 K-Means 初始化)
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
    """测试完整的 SVG2 流程，使用相同的 K-Means 初始化确保公平对比"""
    print("\n" + "="*80)
    print("TEST 6: Full SVG2 Pipeline (with identical K-Means initialization)")
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
    
    # 转换为 [B, H, S, D] 格式
    q_bhsd = q.transpose(1, 2).contiguous()
    k_bhsd = k.transpose(1, 2).contiguous()
    v_bhsd = v.transpose(1, 2).contiguous()
    
    # 生成共享的初始 centroids
    BH = batch_size * num_heads
    q_flat = q_bhsd.reshape(BH, seq_len, head_dim)
    k_flat = k_bhsd.reshape(BH, seq_len, head_dim)
    
    torch.manual_seed(42)  # 确保可复现
    q_init_indices = torch.randint(0, seq_len, (BH, num_q_clusters), device=device)
    k_init_indices = torch.randint(0, seq_len, (BH, num_k_clusters), device=device)
    
    q_init_centroids = torch.gather(
        q_flat, dim=1, 
        index=q_init_indices.unsqueeze(-1).expand(-1, -1, head_dim)
    )  # [BH, Kq, D]
    k_init_centroids = torch.gather(
        k_flat, dim=1,
        index=k_init_indices.unsqueeze(-1).expand(-1, -1, head_dim)
    )  # [BH, Kk, D]
    
    print(f"Shared Q init centroids shape: {q_init_centroids.shape}")
    print(f"Shared K init centroids shape: {k_init_centroids.shape}")
    
    # Dense reference
    print("\n[Reference] Dense Attention...")
    with torch.no_grad():
        dense_out = dense_attention(q_bhsd, k_bhsd, v_bhsd)
        dense_out = dense_out.transpose(1, 2)  # 回到 [B, S, H, D]
    print(f"  Output shape: {dense_out.shape}")
    
    # SGLang SVG2 (使用共享的初始化)
    print("\n[SGLang] SVG2 Forward (with shared init)...")
    # 转换 init_centroids 到 [B, H, K, D] 格式给 SGLang
    q_init_bhkd = q_init_centroids.reshape(batch_size, num_heads, num_q_clusters, head_dim)
    k_init_bhkd = k_init_centroids.reshape(batch_size, num_heads, num_k_clusters, head_dim)
    
    with torch.no_grad():
        sglang_out, sglang_q_cent, sglang_k_cent = sglang_svg2_forward(
            q, k, v,
            num_q_clusters=num_q_clusters,
            num_k_clusters=num_k_clusters,
            top_p=top_p,
            kmeans_iters=kmeans_iters,
            init_q_centroids=q_init_bhkd.clone(),
            init_k_centroids=k_init_bhkd.clone(),
        )
    print(f"  Output shape: {sglang_out.shape}")
    
    # SVG Components 流程 (使用相同的初始化)
    print("\n[SVG Components + PyTorch Sparse Attn] (with shared init)...")
    with torch.no_grad():
        # K-Means (使用共享初始化)
        q_labels, q_centroids, q_sizes, _ = svg_kmeans(
            q_flat, n_clusters=num_q_clusters, max_iters=kmeans_iters, 
            init_centroids=q_init_centroids.clone()
        )
        k_labels, k_centroids, k_sizes, _ = svg_kmeans(
            k_flat, n_clusters=num_k_clusters, max_iters=kmeans_iters,
            init_centroids=k_init_centroids.clone()
        )
        
        # Reshape
        q_centroids = q_centroids.reshape(batch_size, num_heads, num_q_clusters, head_dim)
        k_centroids = k_centroids.reshape(batch_size, num_heads, num_k_clusters, head_dim)
        q_sizes = q_sizes.reshape(batch_size, num_heads, num_q_clusters)
        k_sizes = k_sizes.reshape(batch_size, num_heads, num_k_clusters)
        
        # Block mask
        block_mask = svg_identify_map(q_centroids, k_centroids, q_sizes, k_sizes, p=top_p)
        
        # Permutation (使用 PyTorch 参考实现)
        q_perm, q_idx = pytorch_ref_permute(q_bhsd, q_labels, dim=2)
        k_perm, k_idx = pytorch_ref_permute(k_bhsd, k_labels, dim=2)
        v_perm, _ = pytorch_ref_permute(v_bhsd, k_labels, dim=2, sorted_indices=k_idx)
        
        # Block sparse attention (标准 SVG PyTorch 参考实现)
        out_perm = svg_block_sparse_torch(q_perm, k_perm, v_perm, block_mask, q_sizes, k_sizes)
        
        # Inverse permutation (使用 PyTorch 参考实现)
        svg_ref_out = pytorch_ref_inverse_permute(out_perm, q_idx, dim=2)
        svg_ref_out = svg_ref_out.transpose(1, 2)  # 回到 [B, S, H, D]
    
    print(f"  Output shape: {svg_ref_out.shape}")
    
    # 对比 K-Means 结果
    print("\n[K-Means Comparison]")
    compute_errors(sglang_q_cent, q_centroids, "Q Centroids")
    compute_errors(sglang_k_cent, k_centroids, "K Centroids")
    
    # 对比最终结果
    print("\n[Final Output Comparison]")
    print("SGLang SVG2 vs Dense:")
    compute_errors(sglang_out, dense_out, "SGLang vs Dense")
    
    print("\nSVG Reference vs Dense:")
    compute_errors(svg_ref_out, dense_out, "SVG Ref vs Dense")
    
    print("\nSGLang vs SVG Reference (should be very close now):")
    compute_errors(sglang_out, svg_ref_out, "SGLang vs SVG Ref")
    
    return dense_out, sglang_out, svg_ref_out


# ============================================================================
# 测试 7: 多 top_p 值对比测试
# ============================================================================
def test_multi_top_p(
    batch_size=1,
    seq_len=1024,
    num_heads=4,
    head_dim=64,
    num_q_clusters=32,
    num_k_clusters=32,
    top_p_values=[0.3, 0.5, 0.7, 0.9],
    kmeans_iters=10,
    dtype=torch.bfloat16,
):
    """测试不同 top_p 值下 SGLang 和 SVG Reference 的对比"""
    print("\n" + "="*80)
    print("TEST 7: Multi Top-P Comparison (SGLang vs SVG Reference)")
    print("="*80)
    
    device = 'cuda'
    torch.manual_seed(42)
    
    # 生成测试数据
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    
    # 转换格式
    q_bhsd = q.transpose(1, 2).contiguous()
    k_bhsd = k.transpose(1, 2).contiguous()
    v_bhsd = v.transpose(1, 2).contiguous()
    
    # 生成共享的初始 centroids
    BH = batch_size * num_heads
    q_flat = q_bhsd.reshape(BH, seq_len, head_dim)
    k_flat = k_bhsd.reshape(BH, seq_len, head_dim)
    
    torch.manual_seed(42)
    q_init_indices = torch.randint(0, seq_len, (BH, num_q_clusters), device=device)
    k_init_indices = torch.randint(0, seq_len, (BH, num_k_clusters), device=device)
    
    q_init_centroids = torch.gather(
        q_flat, dim=1, 
        index=q_init_indices.unsqueeze(-1).expand(-1, -1, head_dim)
    )
    k_init_centroids = torch.gather(
        k_flat, dim=1,
        index=k_init_indices.unsqueeze(-1).expand(-1, -1, head_dim)
    )
    
    # 先运行 K-Means 一次，两边使用相同结果
    print("Running shared K-Means...")
    with torch.no_grad():
        # SVG K-Means
        q_labels, q_centroids, q_sizes, _ = svg_kmeans(
            q_flat, n_clusters=num_q_clusters, max_iters=kmeans_iters, 
            init_centroids=q_init_centroids.clone()
        )
        k_labels, k_centroids, k_sizes, _ = svg_kmeans(
            k_flat, n_clusters=num_k_clusters, max_iters=kmeans_iters,
            init_centroids=k_init_centroids.clone()
        )
        
        # SGLang K-Means (验证一致性)
        sglang_q_labels, sglang_q_centroids, sglang_q_sizes = sglang_kmeans(
            q_flat, n_clusters=num_q_clusters, max_iters=kmeans_iters, 
            init_centroids=q_init_centroids.clone()
        )
        sglang_k_labels, sglang_k_centroids, sglang_k_sizes = sglang_kmeans(
            k_flat, n_clusters=num_k_clusters, max_iters=kmeans_iters,
            init_centroids=k_init_centroids.clone()
        )
    
    # 检查 K-Means 结果一致性
    q_labels_match = (q_labels == sglang_q_labels).float().mean().item() * 100
    k_labels_match = (k_labels == sglang_k_labels).float().mean().item() * 100
    print(f"  Q labels match: {q_labels_match:.2f}%")
    print(f"  K labels match: {k_labels_match:.2f}%")
    
    # Reshape centroids
    q_centroids_bhkd = q_centroids.reshape(batch_size, num_heads, num_q_clusters, head_dim)
    k_centroids_bhkd = k_centroids.reshape(batch_size, num_heads, num_k_clusters, head_dim)
    q_sizes_bh = q_sizes.reshape(batch_size, num_heads, num_q_clusters)
    k_sizes_bh = k_sizes.reshape(batch_size, num_heads, num_k_clusters)
    
    sglang_q_centroids_bhkd = sglang_q_centroids.reshape(batch_size, num_heads, num_q_clusters, head_dim)
    sglang_k_centroids_bhkd = sglang_k_centroids.reshape(batch_size, num_heads, num_k_clusters, head_dim)
    sglang_q_sizes_bh = sglang_q_sizes.reshape(batch_size, num_heads, num_q_clusters)
    sglang_k_sizes_bh = sglang_k_sizes.reshape(batch_size, num_heads, num_k_clusters)
    
    # Permutation (共用一次)
    with torch.no_grad():
        # PyTorch 参考 permute
        q_perm_ref, q_idx = pytorch_ref_permute(q_bhsd, q_labels, dim=2)
        k_perm_ref, k_idx = pytorch_ref_permute(k_bhsd, k_labels, dim=2)
        v_perm_ref, _ = pytorch_ref_permute(v_bhsd, k_labels, dim=2, sorted_indices=k_idx)
        
        # SGLang Triton permute
        q_perm_sg, _ = sglang_permute(q_bhsd, labels=sglang_q_labels)
        k_perm_sg, sglang_k_idx = sglang_permute(k_bhsd, labels=sglang_k_labels)
        v_perm_sg, _ = sglang_permute(v_bhsd, sorted_indices=sglang_k_idx)
    
    # Dense reference
    with torch.no_grad():
        dense_out = dense_attention(q_bhsd, k_bhsd, v_bhsd)
    
    print(f"\nInput: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
    print(f"Clusters: Qc={num_q_clusters}, Kc={num_k_clusters}")
    print("-" * 100)
    print(f"{'top_p':>6} | {'Active%':>8} | {'SGLang vs SVG':>20} | {'SGLang vs Dense':>15} | {'SVG vs Dense':>15}")
    print(f"{'':>6} | {'':>8} | {'Cosine':>10} {'MaxAbs':>9} | {'Cosine':>15} | {'Cosine':>15}")
    print("-" * 100)
    
    for top_p in top_p_values:
        with torch.no_grad():
            # 生成 block mask (两边使用相同的 centroids)
            svg_mask = svg_identify_map(q_centroids_bhkd, k_centroids_bhkd, q_sizes_bh, k_sizes_bh, p=top_p)
            sglang_mask = sglang_identify_mask(
                sglang_q_centroids_bhkd, sglang_k_centroids_bhkd, 
                sglang_q_sizes_bh, sglang_k_sizes_bh, 
                top_p=top_p
            )
            
            active_pct = svg_mask.float().mean().item() * 100
            
            # SVG block sparse attention
            svg_out_perm = svg_block_sparse_torch(
                q_perm_ref, k_perm_ref, v_perm_ref, 
                svg_mask, q_sizes_bh, k_sizes_bh
            )
            svg_out = pytorch_ref_inverse_permute(svg_out_perm, q_idx, dim=2)
            
            # SGLang block sparse attention
            sglang_out_perm = sglang_block_sparse_attn(
                q_perm_sg, k_perm_sg, v_perm_sg,
                sglang_mask, sglang_q_sizes_bh, sglang_k_sizes_bh
            )
            sglang_out = sglang_inverse_permute(sglang_out_perm, sglang_k_idx)
            # 注意：SGLang inverse permute 需要使用 q 的 sorted_indices
            # 重新计算
            _, q_sorted_idx = sglang_permute(q_bhsd, labels=sglang_q_labels)
            sglang_out = sglang_inverse_permute(sglang_out_perm, q_sorted_idx)
            
            # 计算误差
            sglang_f = sglang_out.float()
            svg_f = svg_out.float()
            dense_f = dense_out.float()
            
            # SGLang vs SVG
            cos_sim = F.cosine_similarity(sglang_f.flatten(), svg_f.flatten(), dim=0).item()
            max_abs = (sglang_f - svg_f).abs().max().item()
            
            # SGLang vs Dense
            cos_sglang_dense = F.cosine_similarity(sglang_f.flatten(), dense_f.flatten(), dim=0).item()
            
            # SVG vs Dense
            cos_svg_dense = F.cosine_similarity(svg_f.flatten(), dense_f.flatten(), dim=0).item()
            
            print(f"{top_p:>6.1f} | {active_pct:>7.2f}% | {cos_sim:>10.6f} {max_abs:>9.6f} | {cos_sglang_dense:>15.6f} | {cos_svg_dense:>15.6f}")
    
    print("-" * 100)
    print("Note: 'SGLang vs SVG' should be very close (Cosine ≈ 1.0) for all top_p values")
    print("      'SGLang vs Dense' and 'SVG vs Dense' should be nearly identical")
    print("      Both will vary based on sparsity level (higher top_p = closer to dense)")


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
    # 使用 float32 进行精度测试，避免 bfloat16 的数值差异
    # bfloat16 在距离计算中会有舍入误差，导致 K-Means 结果不一致
    dtype = torch.float32  # 改为 float32 以确保精度一致性
    
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
    
    # 测试 7: Multi Top-P Comparison
    try:
        test_multi_top_p(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=dim,
            num_q_clusters=num_q_clusters,
            num_k_clusters=num_k_clusters,
            top_p_values=[0.3, 0.5, 0.7, 0.9],
            dtype=dtype,
        )
        results['multi_top_p'] = 'PASS'
    except Exception as e:
        print(f"ERROR in test_multi_top_p: {e}")
        results['multi_top_p'] = f'FAIL: {e}'
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

