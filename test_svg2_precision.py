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
    rel_diff = abs_diff / (torch.abs(reference) + 1e-8)
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        output.flatten(), reference.flatten(), dim=0
    ).item()
    
    return {
        'max_abs': abs_diff.max().item(),
        'mean_abs': abs_diff.mean().item(),
        'max_rel': rel_diff.max().item() * 100,  # as percentage
        'mean_rel': rel_diff.mean().item() * 100,  # as percentage
        'l2_rel': (torch.norm(abs_diff) / torch.norm(reference)).item() * 100,
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
        print(f"  Max relative error: {errors['max_rel']:.2f}%")
        print(f"  Mean relative error: {errors['mean_rel']:.2f}%")
        print(f"  L2 relative error: {errors['l2_rel']:.2f}%")
        print(f"  Cosine similarity: {errors['cosine_sim']:.6f}")
        print()
        
        # Clear cache
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Summary table
    print("="*80)
    print("SUMMARY: Precision vs Top-P")
    print("="*80)
    print(f"{'Top-P':<10} {'Mean Rel%':<12} {'Max Rel%':<12} {'L2 Rel%':<12} {'Cosine Sim':<12}")
    print("-"*80)
    
    for top_p in top_p_values:
        e = results[top_p]
        print(f"{top_p:<10.1f} {e['mean_rel']:<12.4f} {e['max_rel']:<12.2f} "
              f"{e['l2_rel']:<12.4f} {e['cosine_sim']:<12.6f}")
    
    print("="*80)
    print("\nInterpretation Guide:")
    print("  • Cosine Similarity: 1.0 = perfect, >0.99 = excellent, >0.95 = good")
    print("  • Mean Relative Error: <1% = excellent, <5% = acceptable, <10% = usable")
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

