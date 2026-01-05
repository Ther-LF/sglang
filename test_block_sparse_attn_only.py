#!/usr/bin/env python3
"""
专门测试 Block Sparse Attention 内核的精度

这个脚本会：
1. 使用固定的 block mask 测试你的 Triton 实现
2. 与 PyTorch 参考实现对比
3. 与 Dense Attention 对比

Usage:
    cd /Users/luofan/Desktop/sglang
    python test_block_sparse_attn_only.py
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

# 自动检测 Sparse-VideoGen 路径
SVG_PATHS = [
    os.environ.get("SVG_PATH", ""),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Sparse-VideoGen")),
    os.path.expanduser("~/Sparse-VideoGen"),
    "/root/Sparse-VideoGen",
    "/Users/luofan/Desktop/Sparse-VideoGen",
]
SVG_PATH = None
for p in SVG_PATHS:
    if p and os.path.exists(os.path.join(p, "svg")):
        SVG_PATH = os.path.abspath(p)
        break

if SVG_PATH is None:
    print("ERROR: Could not find Sparse-VideoGen directory!")
    print("Please set SVG_PATH environment variable")
    print("Checked paths:")
    for p in SVG_PATHS:
        if p:
            print(f"  - {p}")
    sys.exit(1)

print(f"Using Sparse-VideoGen from: {SVG_PATH}")
sys.path.insert(0, SVG_PATH)

# 你的实现
from sglang.multimodal_gen.runtime.layers.attention.backends.svg2_sparse_attn import (
    block_sparse_attention as sglang_block_sparse_attn,
)

# 标准 SVG PyTorch 参考实现
from svg.kmeans_utils import dynamic_block_sparse_fwd_torch as svg_block_sparse_torch


def dense_attention(q, k, v):
    """参考 Dense Attention"""
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def compute_errors(output, reference, name=""):
    """计算误差"""
    output = output.float()
    reference = reference.float()
    
    abs_diff = torch.abs(output - reference)
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    
    ref_norm = torch.norm(reference).item()
    error_norm = torch.norm(abs_diff).item()
    l2_rel_pct = (error_norm / ref_norm) * 100 if ref_norm > 0 else 0
    
    cos_sim = F.cosine_similarity(output.flatten(), reference.flatten(), dim=0).item()
    
    print(f"  [{name}]")
    print(f"    Max Abs: {max_abs:.6f}, Mean Abs: {mean_abs:.6f}")
    print(f"    L2 Rel: {l2_rel_pct:.4f}%, Cosine: {cos_sim:.8f}")
    
    return {'max_abs': max_abs, 'mean_abs': mean_abs, 'l2_rel_pct': l2_rel_pct, 'cosine_sim': cos_sim}


def test_with_full_mask():
    """测试 1: 全 1 的 block mask (应该等同于 dense attention)"""
    print("\n" + "="*80)
    print("TEST 1: Full Block Mask (all ones) - Should match Dense Attention")
    print("="*80)
    
    device = 'cuda'
    dtype = torch.bfloat16
    torch.manual_seed(42)
    
    batch_size = 1
    num_heads = 4
    seq_len = 512
    dim = 64
    num_q_clusters = 8
    num_k_clusters = 8
    
    # 生成 Q, K, V
    q = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    
    # 均匀的 cluster sizes
    tokens_per_cluster = seq_len // num_q_clusters
    q_sizes = torch.full((batch_size, num_heads, num_q_clusters), tokens_per_cluster, 
                         dtype=torch.int64, device=device)
    k_sizes = torch.full((batch_size, num_heads, num_k_clusters), tokens_per_cluster,
                         dtype=torch.int64, device=device)
    
    # 全 1 的 block mask
    block_mask = torch.ones((batch_size, num_heads, num_q_clusters, num_k_clusters), 
                            dtype=torch.bool, device=device)
    
    print(f"Config: B={batch_size}, H={num_heads}, S={seq_len}, D={dim}")
    print(f"Clusters: Qc={num_q_clusters}, Kc={num_k_clusters}")
    print(f"Block mask: all ones (should be identical to dense)")
    
    # Dense reference
    with torch.no_grad():
        dense_out = dense_attention(q, k, v)
    
    # SVG PyTorch reference
    with torch.no_grad():
        svg_out = svg_block_sparse_torch(q, k, v, block_mask, q_sizes, k_sizes)
    
    # Your implementation
    with torch.no_grad():
        sglang_out = sglang_block_sparse_attn(q, k, v, block_mask, q_sizes, k_sizes)
    
    print("\n[Results]")
    print("SVG PyTorch vs Dense:")
    svg_err = compute_errors(svg_out, dense_out, "SVG vs Dense")
    
    print("\nSGLang Triton vs Dense:")
    sglang_err = compute_errors(sglang_out, dense_out, "SGLang vs Dense")
    
    print("\nSGLang vs SVG PyTorch:")
    compare_err = compute_errors(sglang_out, svg_out, "SGLang vs SVG")
    
    # 检查
    if svg_err['cosine_sim'] < 0.999:
        print("\n⚠️  WARNING: SVG PyTorch reference differs from Dense!")
    if sglang_err['cosine_sim'] < 0.99:
        print("\n❌ FAIL: SGLang implementation has significant error!")
    else:
        print("\n✓ SGLang implementation looks correct for full mask")
    
    return sglang_err


def test_with_sparse_mask():
    """测试 2: 稀疏的 block mask"""
    print("\n" + "="*80)
    print("TEST 2: Sparse Block Mask (50% sparsity)")
    print("="*80)
    
    device = 'cuda'
    dtype = torch.bfloat16
    torch.manual_seed(42)
    
    batch_size = 1
    num_heads = 4
    seq_len = 512
    dim = 64
    num_q_clusters = 8
    num_k_clusters = 8
    
    # 生成 Q, K, V
    q = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    
    # 均匀的 cluster sizes
    tokens_per_cluster = seq_len // num_q_clusters
    q_sizes = torch.full((batch_size, num_heads, num_q_clusters), tokens_per_cluster, 
                         dtype=torch.int64, device=device)
    k_sizes = torch.full((batch_size, num_heads, num_k_clusters), tokens_per_cluster,
                         dtype=torch.int64, device=device)
    
    # 稀疏的 block mask (50% 稀疏)
    torch.manual_seed(123)
    block_mask = torch.rand((batch_size, num_heads, num_q_clusters, num_k_clusters), 
                            device=device) > 0.5
    # 确保每行至少有一个 active block
    block_mask[..., 0] = True
    
    active = block_mask.sum().item()
    total = block_mask.numel()
    print(f"Config: B={batch_size}, H={num_heads}, S={seq_len}, D={dim}")
    print(f"Block mask: {active}/{total} active ({100*active/total:.1f}%)")
    
    # SVG PyTorch reference
    with torch.no_grad():
        svg_out = svg_block_sparse_torch(q, k, v, block_mask, q_sizes, k_sizes)
    
    # Your implementation
    with torch.no_grad():
        sglang_out = sglang_block_sparse_attn(q, k, v, block_mask, q_sizes, k_sizes)
    
    print("\n[Results]")
    print("SGLang Triton vs SVG PyTorch:")
    compare_err = compute_errors(sglang_out, svg_out, "SGLang vs SVG")
    
    if compare_err['cosine_sim'] < 0.99:
        print("\n❌ FAIL: SGLang implementation differs significantly from SVG PyTorch!")
    else:
        print("\n✓ SGLang implementation matches SVG PyTorch for sparse mask")
    
    return compare_err


def test_with_uneven_clusters():
    """测试 3: 不均匀的 cluster sizes"""
    print("\n" + "="*80)
    print("TEST 3: Uneven Cluster Sizes")
    print("="*80)
    
    device = 'cuda'
    dtype = torch.bfloat16
    torch.manual_seed(42)
    
    batch_size = 1
    num_heads = 4
    seq_len = 512
    dim = 64
    num_q_clusters = 8
    num_k_clusters = 8
    
    # 生成 Q, K, V
    q = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    
    # 不均匀的 cluster sizes (模拟真实 K-Means 结果)
    # 生成随机分布，但确保总和等于 seq_len
    torch.manual_seed(456)
    q_sizes_raw = torch.randint(10, 100, (batch_size, num_heads, num_q_clusters), device=device).float()
    q_sizes_normalized = (q_sizes_raw / q_sizes_raw.sum(dim=-1, keepdim=True) * seq_len).long()
    # 修正余数
    remainder = seq_len - q_sizes_normalized.sum(dim=-1, keepdim=True)
    q_sizes_normalized[..., 0] += remainder.squeeze(-1)
    q_sizes = q_sizes_normalized
    
    k_sizes_raw = torch.randint(10, 100, (batch_size, num_heads, num_k_clusters), device=device).float()
    k_sizes_normalized = (k_sizes_raw / k_sizes_raw.sum(dim=-1, keepdim=True) * seq_len).long()
    remainder = seq_len - k_sizes_normalized.sum(dim=-1, keepdim=True)
    k_sizes_normalized[..., 0] += remainder.squeeze(-1)
    k_sizes = k_sizes_normalized
    
    print(f"Config: B={batch_size}, H={num_heads}, S={seq_len}, D={dim}")
    print(f"Q cluster sizes: min={q_sizes.min().item()}, max={q_sizes.max().item()}, sum={q_sizes[0,0].sum().item()}")
    print(f"K cluster sizes: min={k_sizes.min().item()}, max={k_sizes.max().item()}, sum={k_sizes[0,0].sum().item()}")
    
    # 全 1 block mask
    block_mask = torch.ones((batch_size, num_heads, num_q_clusters, num_k_clusters), 
                            dtype=torch.bool, device=device)
    
    # SVG PyTorch reference
    with torch.no_grad():
        svg_out = svg_block_sparse_torch(q, k, v, block_mask, q_sizes, k_sizes)
    
    # Your implementation
    with torch.no_grad():
        sglang_out = sglang_block_sparse_attn(q, k, v, block_mask, q_sizes, k_sizes)
    
    # Dense reference
    with torch.no_grad():
        dense_out = dense_attention(q, k, v)
    
    print("\n[Results]")
    print("SVG PyTorch vs Dense:")
    compute_errors(svg_out, dense_out, "SVG vs Dense")
    
    print("\nSGLang Triton vs Dense:")
    compute_errors(sglang_out, dense_out, "SGLang vs Dense")
    
    print("\nSGLang vs SVG PyTorch:")
    compare_err = compute_errors(sglang_out, svg_out, "SGLang vs SVG")
    
    if compare_err['cosine_sim'] < 0.99:
        print("\n❌ FAIL: SGLang implementation differs with uneven cluster sizes!")
    else:
        print("\n✓ SGLang implementation handles uneven clusters correctly")
    
    return compare_err


def test_larger_scale():
    """测试 4: 更大规模的测试"""
    print("\n" + "="*80)
    print("TEST 4: Larger Scale (closer to real video generation)")
    print("="*80)
    
    device = 'cuda'
    dtype = torch.bfloat16
    torch.manual_seed(42)
    
    batch_size = 1
    num_heads = 16
    seq_len = 4096
    dim = 64
    num_q_clusters = 64
    num_k_clusters = 64
    top_p_sparsity = 0.5  # 50% 的 blocks 被选中
    
    # 生成 Q, K, V
    q = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, dim, dtype=dtype, device=device)
    
    # 均匀的 cluster sizes
    tokens_per_cluster = seq_len // num_q_clusters
    q_sizes = torch.full((batch_size, num_heads, num_q_clusters), tokens_per_cluster, 
                         dtype=torch.int64, device=device)
    k_sizes = torch.full((batch_size, num_heads, num_k_clusters), tokens_per_cluster,
                         dtype=torch.int64, device=device)
    
    # 稀疏 block mask
    torch.manual_seed(789)
    block_mask = torch.rand((batch_size, num_heads, num_q_clusters, num_k_clusters), 
                            device=device) < top_p_sparsity
    block_mask[..., 0] = True  # 确保每行至少有一个
    
    active = block_mask.sum().item()
    total = block_mask.numel()
    print(f"Config: B={batch_size}, H={num_heads}, S={seq_len}, D={dim}")
    print(f"Clusters: Qc={num_q_clusters}, Kc={num_k_clusters}")
    print(f"Block mask: {active}/{total} active ({100*active/total:.1f}%)")
    
    # SVG PyTorch reference (会比较慢)
    print("\nRunning SVG PyTorch reference (may take a while)...")
    with torch.no_grad():
        svg_out = svg_block_sparse_torch(q, k, v, block_mask, q_sizes, k_sizes)
    
    # Your implementation
    print("Running SGLang Triton...")
    with torch.no_grad():
        sglang_out = sglang_block_sparse_attn(q, k, v, block_mask, q_sizes, k_sizes)
    
    print("\n[Results]")
    print("SGLang Triton vs SVG PyTorch:")
    compare_err = compute_errors(sglang_out, svg_out, "SGLang vs SVG")
    
    if compare_err['cosine_sim'] < 0.99:
        print("\n❌ FAIL: SGLang implementation has issues at larger scale!")
    else:
        print("\n✓ SGLang implementation works at larger scale")
    
    return compare_err


def main():
    print("="*80)
    print("Block Sparse Attention Kernel Precision Test")
    print("="*80)
    
    results = {}
    
    # Test 1: Full mask
    try:
        results['full_mask'] = test_with_full_mask()
    except Exception as e:
        print(f"\n❌ ERROR in test_with_full_mask: {e}")
        import traceback
        traceback.print_exc()
        results['full_mask'] = {'error': str(e)}
    
    # Test 2: Sparse mask
    try:
        results['sparse_mask'] = test_with_sparse_mask()
    except Exception as e:
        print(f"\n❌ ERROR in test_with_sparse_mask: {e}")
        import traceback
        traceback.print_exc()
        results['sparse_mask'] = {'error': str(e)}
    
    # Test 3: Uneven clusters
    try:
        results['uneven_clusters'] = test_with_uneven_clusters()
    except Exception as e:
        print(f"\n❌ ERROR in test_with_uneven_clusters: {e}")
        import traceback
        traceback.print_exc()
        results['uneven_clusters'] = {'error': str(e)}
    
    # Test 4: Larger scale
    try:
        results['larger_scale'] = test_larger_scale()
    except Exception as e:
        print(f"\n❌ ERROR in test_larger_scale: {e}")
        import traceback
        traceback.print_exc()
        results['larger_scale'] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        if 'error' in result:
            print(f"  ❌ {test_name}: FAILED - {result['error']}")
        elif result.get('cosine_sim', 0) >= 0.99:
            print(f"  ✓ {test_name}: PASS (cosine={result['cosine_sim']:.6f})")
        else:
            print(f"  ⚠️  {test_name}: LOW PRECISION (cosine={result['cosine_sim']:.6f})")
    
    print("="*80)
    print("\n如果上面的测试显示 SGLang vs SVG PyTorch 的 cosine similarity 很低，")
    print("那么问题出在你的 block_sparse_attention Triton 内核。")
    print("\n如果 cosine similarity 很高（>0.99），但完整 SVG2 测试失败，")
    print("那么问题可能出在 K-Means 或 Permutation 环节。")


if __name__ == '__main__':
    main()

