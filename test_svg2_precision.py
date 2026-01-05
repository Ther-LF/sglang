#!/usr/bin/env python3
"""
SVG2 Precision Test - Compare different top_p values against dense attention

Usage:
    cd /Users/luofan/Desktop/sglang
    python test_svg2_precision.py
"""

import torch
import sys
import os

# Add sglang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from sglang.multimodal_gen.runtime.layers.attention.backends.svg2_sparse_attn import svg2_attention_forward


def dense_attention(q, k, v):
    """Reference dense attention using PyTorch SDPA."""
    # Input: [B, S, H, D], need [B, H, S, D] for SDPA
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2) 
    v_t = v.transpose(1, 2)
    
    out = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, dropout_p=0.0, is_causal=False
    )
    
    return out.transpose(1, 2)  # Back to [B, S, H, D]


def compute_errors(output, reference):
    """Compute various error metrics."""
    output = output.float()
    reference = reference.float()
    
    abs_diff = torch.abs(output - reference)
    
    # Method 1: Relative to reference norm (more stable)
    ref_norm = torch.norm(reference).item()
    error_norm = torch.norm(abs_diff).item()
    l2_rel_pct = (error_norm / ref_norm) * 100 if ref_norm > 0 else 0
    
    # Method 2: Relative to value range (robust to near-zero values)
    ref_range = reference.max().item() - reference.min().item()
    range_rel_pct = (abs_diff.mean().item() / ref_range) * 100 if ref_range > 0 else 0
    
    # Method 3: Element-wise relative (only for non-tiny values)
    threshold = 0.01  # Only compute relative error where |ref| > threshold
    mask = torch.abs(reference) > threshold
    if mask.sum() > 0:
        rel_diff_masked = abs_diff[mask] / torch.abs(reference[mask])
        mean_rel_masked = rel_diff_masked.mean().item() * 100
        max_rel_masked = rel_diff_masked.max().item() * 100
    else:
        mean_rel_masked = 0
        max_rel_masked = 0
    
    # Cosine similarity (1.0 = perfect)
    cos_sim = torch.nn.functional.cosine_similarity(
        output.flatten(), reference.flatten(), dim=0
    ).item()
    
    # RMSE
    rmse = torch.sqrt((abs_diff ** 2).mean()).item()
    
    return {
        'max_abs': abs_diff.max().item(),
        'mean_abs': abs_diff.mean().item(),
        'rmse': rmse,
        'l2_rel_pct': l2_rel_pct,  # Most reliable
        'range_rel_pct': range_rel_pct,  # Relative to output range
        'mean_rel_pct': mean_rel_masked,  # Only for |ref| > 0.01
        'max_rel_pct': max_rel_masked,
        'cosine_sim': cos_sim,
    }


def test_top_p_precision(
    batch_size=1,
    seq_len=8192,
    num_heads=16, 
    head_dim=64,
    num_q_clusters=300,
    num_k_clusters=1000,
    top_p_values=[0.3, 0.5, 0.7, 0.9],
    kmeans_iters=50,
    dtype=torch.bfloat16,
):
    """Test SVG2 precision across different top_p values."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*80)
    print("SVG2 Sparse Attention Precision Test")
    print("="*80)
    print(f"Configuration:")
    print(f"  Shape: B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}")
    print(f"  Clusters: Qc={num_q_clusters}, Kc={num_k_clusters}")
    print(f"  K-Means iters: {kmeans_iters}")
    print(f"  Dtype: {dtype}")
    print("="*80 + "\n")
    
    # Generate random test data
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    
    # Compute dense reference
    print("Computing DENSE attention (reference)...")
    with torch.no_grad():
        dense_out = dense_attention(q, k, v)
    print(f"  Output shape: {dense_out.shape}")
    print(f"  Output range: [{dense_out.min().item():.4f}, {dense_out.max().item():.4f}]")
    print()
    
    # Test each top_p value
    results = {}
    for top_p in top_p_values:
        print(f"Testing SVG2 with top_p={top_p}...")
        
        with torch.no_grad():
            svg2_out, _, _ = svg2_attention_forward(
                q, k, v,
                num_q_clusters=num_q_clusters,
                num_k_clusters=num_k_clusters,
                top_p=top_p,
                kmeans_iters=kmeans_iters,
                max_k_clusters_per_q=None,
            )
        
        errors = compute_errors(svg2_out, dense_out)
        results[top_p] = errors
        
        print(f"  Max absolute error: {errors['max_abs']:.6f}")
        print(f"  Mean absolute error: {errors['mean_abs']:.6f}")
        print(f"  RMSE: {errors['rmse']:.6f}")
        print(f"  L2 relative error: {errors['l2_rel_pct']:.2f}%")
        print(f"  Range relative error: {errors['range_rel_pct']:.2f}%")
        print(f"  Cosine similarity: {errors['cosine_sim']:.6f}")
        print()
        
        # Clear cache
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Summary table
    print("="*80)
    print("SUMMARY: Precision vs Top-P")
    print("="*80)
    print(f"{'Top-P':<10} {'Mean Abs':<12} {'RMSE':<12} {'L2 Rel%':<12} {'Cosine Sim':<12}")
    print("-"*80)
    
    for top_p in top_p_values:
        e = results[top_p]
        print(f"{top_p:<10.1f} {e['mean_abs']:<12.6f} {e['rmse']:<12.6f} "
              f"{e['l2_rel_pct']:<12.2f} {e['cosine_sim']:<12.6f}")
    
    print("="*80)
    print("\nInterpretation Guide:")
    print("  • Cosine Similarity: 1.0 = perfect, >0.99 = excellent, >0.95 = good, >0.9 = acceptable")
    print("  • L2 Relative Error: <10% = excellent, <30% = good, <50% = acceptable")
    print("  • RMSE & Mean Abs: Lower is better (compare to output range)")
    print("  • Higher top_p → less pruning → better precision (but slower)")
    print("="*80 + "\n")
    
    return results


def test_realistic_video_config():
    """Test with realistic video generation parameters."""
    
    print("\n" + "#"*80)
    print("# Realistic Video Config Test (720p, 81 frames)")
    print("#"*80 + "\n")
    
    # 720p video: 60x45 = 2700 tokens per frame, 81 frames = 218700 tokens
    # Plus ~300 text tokens = ~219000 tokens total
    # This is close to your actual usage
    
    batch_size = 2  # Conditional + unconditional for CFG
    seq_len = 8192  # Simplified for testing (full would be ~220k)
    num_heads = 24  # Typical for large models
    head_dim = 128  # Typical for large models
    
    results = test_top_p_precision(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        num_q_clusters=300,
        num_k_clusters=1000,
        top_p_values=[0.3, 0.5, 0.7, 0.9],
        kmeans_iters=50,
        dtype=torch.bfloat16,
    )
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SVG2 precision')
    parser.add_argument('--quick', action='store_true', help='Quick test with small config')
    parser.add_argument('--realistic', action='store_true', help='Test with realistic video config')
    parser.add_argument('--seq-len', type=int, default=8192, help='Sequence length')
    parser.add_argument('--num-heads', type=int, default=16, help='Number of heads')
    parser.add_argument('--head-dim', type=int, default=64, help='Head dimension')
    
    args = parser.parse_args()
    
    if args.realistic:
        test_realistic_video_config()
    elif args.quick:
        print("Quick test mode...")
        test_top_p_precision(
            batch_size=1,
            seq_len=2048,  # Smaller for speed
            num_heads=8,
            head_dim=64,
            num_q_clusters=64,
            num_k_clusters=128,
            top_p_values=[0.5, 0.9],
            kmeans_iters=10,
            dtype=torch.bfloat16,
        )
    else:
        # Default test
        test_top_p_precision(
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
        )

