#!/usr/bin/env python3
"""
Benchmark script for SVG2 Sparse Attention.

Compares SVG2 implementation with PyTorch dense attention across:
1. Individual components (K-Means, Permutation, Block Sparse Attention)
2. Full attention forward pass
3. Different sequence lengths and sparsity levels

Usage:
    python benchmark_svg2_sparse_attn.py
    python benchmark_svg2_sparse_attn.py --seq-lengths 1024 4096 8192
    python benchmark_svg2_sparse_attn.py --top-p 0.3 0.5 0.7
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

# ============================================================================
# Utilities
# ============================================================================


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    time_ms: float
    memory_mb: float = 0.0


def benchmark_fn(
    fn: Callable,
    warmup: int = 3,
    repeat: int = 10,
    sync: bool = True,
) -> float:
    """Benchmark a function and return average time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / repeat * 1000


def get_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_comparison(name: str, svg2_ms: float, torch_ms: float):
    """Print a comparison result."""
    speedup = torch_ms / svg2_ms if svg2_ms > 0 else 0
    faster = "SVG2" if speedup > 1 else "Torch"
    ratio = speedup if speedup > 1 else 1 / speedup if speedup > 0 else 0
    
    print(f"  {name:40s} | SVG2: {svg2_ms:8.2f}ms | Torch: {torch_ms:8.2f}ms | {faster} {ratio:.2f}x faster")


# ============================================================================
# Import SVG2 Components
# ============================================================================


def import_svg2_components():
    """Import SVG2 components."""
    from sglang.multimodal_gen.runtime.layers.attention.backends.svg2_sparse_attn import (
        block_sparse_attention,
        identify_dynamic_mask,
        inverse_permute,
        permute_by_labels,
        svg2_attention_forward,
        triton_kmeans,
    )
    return {
        'triton_kmeans': triton_kmeans,
        'permute_by_labels': permute_by_labels,
        'inverse_permute': inverse_permute,
        'identify_dynamic_mask': identify_dynamic_mask,
        'block_sparse_attention': block_sparse_attention,
        'svg2_attention_forward': svg2_attention_forward,
    }


# ============================================================================
# Torch Reference Implementations
# ============================================================================


def torch_kmeans(x: torch.Tensor, n_clusters: int, max_iters: int = 10):
    """Simple PyTorch K-Means implementation."""
    B, N, D = x.shape
    device = x.device
    dtype = x.dtype
    
    # Random initialization
    indices = torch.randint(0, N, (B, n_clusters), device=device)
    batch_offset = torch.arange(B, device=device)[:, None] * N
    flat_indices = (batch_offset + indices).flatten()
    centroids = x.reshape(B * N, D)[flat_indices].reshape(B, n_clusters, D).float()
    
    for _ in range(max_iters):
        # Compute distances [B, N, K]
        x_expanded = x.float().unsqueeze(2)  # [B, N, 1, D]
        c_expanded = centroids.unsqueeze(1)  # [B, 1, K, D]
        dist = ((x_expanded - c_expanded) ** 2).sum(dim=-1)  # [B, N, K]
        
        # Assign clusters
        labels = dist.argmin(dim=-1)  # [B, N]
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(B, n_clusters, device=device)
        
        for b in range(B):
            for k in range(n_clusters):
                mask = labels[b] == k
                if mask.sum() > 0:
                    new_centroids[b, k] = x[b, mask].float().mean(dim=0)
                    counts[b, k] = mask.sum()
        
        centroids = new_centroids
    
    # Compute cluster sizes
    cluster_sizes = torch.zeros(B, n_clusters, dtype=torch.int32, device=device)
    for b in range(B):
        for k in range(n_clusters):
            cluster_sizes[b, k] = (labels[b] == k).sum()
    
    return labels, centroids.to(dtype), cluster_sizes


def torch_dense_attention(q, k, v, scale=None):
    """PyTorch dense attention."""
    B, S, H, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Transpose to [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Compute attention
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v.float())
    
    # Transpose back
    return output.transpose(1, 2).to(q.dtype)


def torch_sdpa_attention(q, k, v, scale=None):
    """PyTorch scaled_dot_product_attention."""
    B, S, H, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Transpose to [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Use SDPA
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale
    )
    
    return output.transpose(1, 2)


# ============================================================================
# Component Benchmarks
# ============================================================================


def benchmark_kmeans(svg2_components: dict, device: str = "cuda"):
    """Benchmark K-Means clustering."""
    print_header("K-Means Clustering Benchmark")
    
    configs = [
        # (B, N, D, K)
        (1, 1024, 64, 16),
        (1, 4096, 64, 32),
        (1, 8192, 64, 64),
        (2, 4096, 128, 64),
    ]
    
    for B, N, D, K in configs:
        x = torch.randn(B, N, D, device=device, dtype=torch.float16)
        
        # SVG2 K-Means
        svg2_time = benchmark_fn(
            lambda: svg2_components['triton_kmeans'](x, K, max_iters=5)
        )
        
        # Torch K-Means
        torch_time = benchmark_fn(
            lambda: torch_kmeans(x, K, max_iters=5)
        )
        
        name = f"B={B}, N={N}, D={D}, K={K}"
        print_comparison(name, svg2_time, torch_time)


def benchmark_permutation(svg2_components: dict, device: str = "cuda"):
    """Benchmark permutation operations."""
    print_header("Permutation Benchmark")
    
    configs = [
        # (B, H, S, D)
        (1, 8, 1024, 64),
        (1, 16, 4096, 64),
        (1, 24, 8192, 128),
    ]
    
    for B, H, S, D in configs:
        x = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        labels = torch.randint(0, 32, (B * H, S), device=device)
        
        # SVG2 Permutation
        svg2_time = benchmark_fn(
            lambda: svg2_components['permute_by_labels'](x, labels)
        )
        
        # Torch Permutation (argsort + gather)
        def torch_permute():
            sorted_indices = torch.argsort(labels, dim=-1)
            x_flat = x.reshape(B * H, S, D)
            expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, D)
            return torch.gather(x_flat, 1, expanded_indices).reshape(B, H, S, D)
        
        torch_time = benchmark_fn(torch_permute)
        
        name = f"B={B}, H={H}, S={S}, D={D}"
        print_comparison(name, svg2_time, torch_time)


def benchmark_block_sparse_attention(svg2_components: dict, device: str = "cuda"):
    """Benchmark block sparse attention."""
    print_header("Block Sparse Attention Benchmark")
    
    configs = [
        # (B, H, S, D, Kq, Kk, sparsity)
        (1, 8, 1024, 64, 16, 16, 0.5),
        (1, 16, 4096, 64, 32, 32, 0.5),
        (1, 16, 4096, 64, 32, 32, 0.7),
    ]
    
    for B, H, S, D, Kq, Kk, sparsity in configs:
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        
        # Create block sizes and mask
        block_size = S // Kq
        q_cluster_sizes = torch.full((B, H, Kq), block_size, dtype=torch.int32, device=device)
        k_cluster_sizes = torch.full((B, H, Kk), block_size, dtype=torch.int32, device=device)
        
        # Random mask with specified sparsity
        block_mask = torch.rand(B, H, Kq, Kk, device=device) > sparsity
        # Ensure at least diagonal is kept
        for i in range(min(Kq, Kk)):
            block_mask[:, :, i, i] = True
        
        # SVG2 Block Sparse Attention
        svg2_time = benchmark_fn(
            lambda: svg2_components['block_sparse_attention'](
                q, k, v, block_mask, q_cluster_sizes, k_cluster_sizes
            )
        )
        
        # Torch Dense Attention (for comparison)
        q_t = q.transpose(1, 2).contiguous()  # [B, S, H, D]
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()
        
        torch_time = benchmark_fn(
            lambda: torch_dense_attention(q_t, k_t, v_t)
        )
        
        actual_sparsity = 1 - block_mask.float().mean().item()
        name = f"S={S}, K={Kq}, sparsity={actual_sparsity:.0%}"
        print_comparison(name, svg2_time, torch_time)


# ============================================================================
# Full Attention Benchmark
# ============================================================================


def benchmark_full_attention(
    svg2_components: dict,
    device: str = "cuda",
    seq_lengths: Optional[List[int]] = None,
    top_p_values: Optional[List[float]] = None,
):
    """Benchmark full SVG2 attention vs Torch attention."""
    print_header("Full Attention Benchmark: SVG2 vs Torch")
    
    if seq_lengths is None:
        seq_lengths = [1024, 2048, 4096, 8192]
    
    if top_p_values is None:
        top_p_values = [0.3, 0.5, 0.7]
    
    B, H, D = 1, 16, 64
    num_clusters = 64
    
    print(f"\n  Config: B={B}, H={H}, D={D}, num_clusters={num_clusters}")
    print("-" * 70)
    
    for S in seq_lengths:
        print(f"\n  Sequence Length: {S}")
        print("-" * 50)
        
        q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        
        # Torch Dense Attention
        torch_dense_time = benchmark_fn(
            lambda: torch_dense_attention(q, k, v)
        )
        
        # Torch SDPA
        torch_sdpa_time = benchmark_fn(
            lambda: torch_sdpa_attention(q, k, v)
        )
        
        print(f"    {'Method':35s} | {'Time (ms)':>12s} | {'vs Dense':>12s} | {'vs SDPA':>12s}")
        print("    " + "-" * 75)
        print(f"    {'Torch Dense':35s} | {torch_dense_time:12.2f} | {'1.00x':>12s} | {torch_dense_time/torch_sdpa_time:.2f}x slower")
        print(f"    {'Torch SDPA':35s} | {torch_sdpa_time:12.2f} | {torch_sdpa_time/torch_dense_time:.2f}x faster | {'1.00x':>12s}")
        
        # SVG2 with different sparsity levels
        for top_p in top_p_values:
            svg2_time = benchmark_fn(
                lambda tp=top_p: svg2_components['svg2_attention_forward'](
                    q, k, v,
                    num_q_clusters=num_clusters,
                    num_k_clusters=num_clusters,
                    top_p=tp,
                    kmeans_iters=3,
                )
            )
            
            vs_dense = torch_dense_time / svg2_time if svg2_time > 0 else 0
            vs_sdpa = torch_sdpa_time / svg2_time if svg2_time > 0 else 0
            
            vs_dense_str = f"{vs_dense:.2f}x faster" if vs_dense > 1 else f"{1/vs_dense:.2f}x slower"
            vs_sdpa_str = f"{vs_sdpa:.2f}x faster" if vs_sdpa > 1 else f"{1/vs_sdpa:.2f}x slower"
            
            sparsity = (1 - top_p) * 100
            print(f"    {'SVG2 (top_p=' + str(top_p) + f', ~{sparsity:.0f}% sparse)':35s} | {svg2_time:12.2f} | {vs_dense_str:>12s} | {vs_sdpa_str:>12s}")


# ============================================================================
# Memory Benchmark
# ============================================================================


def benchmark_memory(svg2_components: dict, device: str = "cuda"):
    """Benchmark memory usage."""
    print_header("Memory Usage Benchmark")
    
    configs = [
        # (B, S, H, D)
        (1, 4096, 16, 64),
        (1, 8192, 16, 64),
        (1, 16384, 16, 64),
    ]
    
    print(f"  {'Config':30s} | {'Dense (MB)':>12s} | {'SVG2 (MB)':>12s} | {'Savings':>12s}")
    print("-" * 70)
    
    for B, S, H, D in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        
        # Measure dense attention memory
        torch.cuda.reset_peak_memory_stats()
        _ = torch_dense_attention(q, k, v)
        torch.cuda.synchronize()
        dense_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure SVG2 memory
        _ = svg2_components['svg2_attention_forward'](
            q, k, v,
            num_q_clusters=64,
            num_k_clusters=64,
            top_p=0.5,
            kmeans_iters=3,
        )
        torch.cuda.synchronize()
        svg2_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        savings = (1 - svg2_memory / dense_memory) * 100 if dense_memory > 0 else 0
        
        name = f"B={B}, S={S}, H={H}, D={D}"
        print(f"  {name:30s} | {dense_memory:12.1f} | {svg2_memory:12.1f} | {savings:11.1f}%")


# ============================================================================
# Scalability Benchmark
# ============================================================================


def benchmark_scalability(svg2_components: dict, device: str = "cuda"):
    """Benchmark scalability with sequence length."""
    print_header("Scalability Benchmark (Time vs Sequence Length)")
    
    B, H, D = 1, 16, 64
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    
    print(f"\n  Config: B={B}, H={H}, D={D}")
    print(f"  {'Seq Len':>10s} | {'Dense (ms)':>12s} | {'SDPA (ms)':>12s} | {'SVG2 (ms)':>12s} | {'SVG2 Speedup':>14s}")
    print("-" * 70)
    
    for S in seq_lengths:
        try:
            q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
            k = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
            v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
            
            # Torch Dense
            try:
                dense_time = benchmark_fn(lambda: torch_dense_attention(q, k, v), repeat=5)
            except RuntimeError:  # OOM
                dense_time = float('inf')
            
            # Torch SDPA
            sdpa_time = benchmark_fn(lambda: torch_sdpa_attention(q, k, v), repeat=5)
            
            # SVG2
            svg2_time = benchmark_fn(
                lambda: svg2_components['svg2_attention_forward'](
                    q, k, v,
                    num_q_clusters=min(64, S // 16),
                    num_k_clusters=min(64, S // 16),
                    top_p=0.5,
                    kmeans_iters=3,
                ),
                repeat=5
            )
            
            speedup = sdpa_time / svg2_time if svg2_time > 0 else 0
            speedup_str = f"{speedup:.2f}x vs SDPA"
            
            dense_str = f"{dense_time:.2f}" if dense_time != float('inf') else "OOM"
            
            print(f"  {S:10d} | {dense_str:>12s} | {sdpa_time:12.2f} | {svg2_time:12.2f} | {speedup_str:>14s}")
            
        except RuntimeError as e:
            print(f"  {S:10d} | Error: {e}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Benchmark SVG2 Sparse Attention")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=None, 
                        help="Sequence lengths to benchmark")
    parser.add_argument("--top-p", type=float, nargs="+", default=None,
                        help="Top-p values to benchmark")
    parser.add_argument("--component", type=str, default="all",
                        choices=["all", "kmeans", "permutation", "block_attn", "full", "memory", "scalability"],
                        help="Which component to benchmark")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for benchmarking!")
        return
    
    print("=" * 70)
    print(" SVG2 Sparse Attention Benchmark Suite")
    print("=" * 70)
    print(f"\n  Device: {torch.cuda.get_device_name()}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    
    components = import_svg2_components()
    
    if args.component == "all" or args.component == "kmeans":
        benchmark_kmeans(components, args.device)
    
    if args.component == "all" or args.component == "permutation":
        benchmark_permutation(components, args.device)
    
    if args.component == "all" or args.component == "block_attn":
        benchmark_block_sparse_attention(components, args.device)
    
    if args.component == "all" or args.component == "full":
        benchmark_full_attention(
            components, 
            args.device,
            seq_lengths=args.seq_lengths,
            top_p_values=args.top_p,
        )
    
    if args.component == "all" or args.component == "memory":
        benchmark_memory(components, args.device)
    
    if args.component == "all" or args.component == "scalability":
        benchmark_scalability(components, args.device)
    
    print("\n" + "=" * 70)
    print(" Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

