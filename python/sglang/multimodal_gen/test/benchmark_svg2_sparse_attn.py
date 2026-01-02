#!/usr/bin/env python3
"""
Benchmark script for SVG2 Sparse Attention (Real-world Video Gen Workloads).

Target Models:
1. Wan2.1-I2V/T2V-14B: 21 frames * 3600 tokens = ~75,600 tokens. (H=40, D=128)
2. HunyuanVideo-13B: 33 frames * 3600 tokens = ~118,800 tokens. (H=48, D=128)

Usage:
    python benchmark_real_world.py
    python benchmark_real_world.py --scenario wan
    python benchmark_real_world.py --scenario hunyuan
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

# ============================================================================
# Utilities
# ============================================================================

def benchmark_fn(fn: Callable, warmup: int = 2, repeat: int = 5, sync: bool = True) -> float:
    """Benchmark a function and return average time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        try:
            fn()
        except RuntimeError:
            return float('inf') # Fail fast on OOM during warmup
            
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / repeat * 1000

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

# ============================================================================
# Import Components
# ============================================================================

def import_svg2_components():
    from sglang.multimodal_gen.runtime.layers.attention.backends.svg2_sparse_attn import (
        svg2_attention_forward,
    )
    return {'svg2_attention_forward': svg2_attention_forward}

# ============================================================================
# Baselines
# ============================================================================

def torch_dense_attention(q, k, v, scale=None):
    if scale is None: scale = 1.0 / math.sqrt(q.shape[-1])
    # [B, S, H, D] -> [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2)

def torch_sdpa_attention(q, k, v, scale=None):
    if scale is None: scale = 1.0 / math.sqrt(q.shape[-1])
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    return out.transpose(1, 2)

# ============================================================================
# Core Benchmark Logic
# ============================================================================

def run_model_benchmark(
    name: str,
    B: int, H: int, D: int, S: int,
    svg2_components: dict,
    device: str = "cuda"
):
    print_header(f"Scenario: {name}")
    print(f"  Configuration: Batch={B}, Heads={H}, Dim={D}, SeqLen={S}")
    print(f"  Total Tokens : {B*S:,}")
    print(f"  Memory (Est) : ~{B*S*H*D*2*3 / 1024**3:.2f} GB (KV Cache + Q)")
    print("-" * 80)

    # 1. Prepare Data
    try:
        q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    except RuntimeError as e:
        print(f"  Skipping: OOM during tensor allocation. {e}")
        return

    # 2. Run Baselines
    # Dense
    print(f"  Running Torch Dense (Reference)... ", end="", flush=True)
    if S > 16384:
        dense_ms = float('inf')
        print("Skipped (Predict OOM)")
    else:
        try:
            dense_ms = benchmark_fn(lambda: torch_dense_attention(q, k, v))
            print(f"{dense_ms:.2f} ms")
        except RuntimeError:
            dense_ms = float('inf')
            print("OOM")

    # SDPA (FlashAttention)
    print(f"  Running Torch SDPA (FlashAttn)...  ", end="", flush=True)
    try:
        sdpa_ms = benchmark_fn(lambda: torch_sdpa_attention(q, k, v))
        print(f"{sdpa_ms:.2f} ms")
    except RuntimeError:
        sdpa_ms = float('inf')
        print("OOM")

    # 3. Run SVG2 Configurations
    # Settings from Paper: Cq=100, Ck=500 (Section D)
    configs = [
        # (Top-P, QC, KC, Max_K_per_Q)
        (0.5, 100, 500, None), 
        (0.7, 100, 500, None),
        (0.9, 100, 500, None), 
        # Add a "Turbo" setting with cap
        (0.9, 100, 500, 32), 
    ]

    print("\n  SVG2 Results:")
    print(f"    {'Setting':40s} | {'Time (ms)':>12s} | {'vs SDPA':>12s} | {'vs Dense':>12s}")
    print("    " + "-" * 76)

    for top_p, qc, kc, cap in configs:
        desc = f"P={top_p}, Qc={qc}, Kc={kc}"
        if cap: desc += f", Cap={cap}"
        
        try:
            # We use iter=1 for step to simulate "next step" performance (using cache)
            # But here we assume cold start or average step. 
            # Paper uses kmeans_iters=1 for steps.
            svg2_ms = benchmark_fn(
                lambda: svg2_components['svg2_attention_forward'](
                    q, k, v,
                    num_q_clusters=qc,
                    num_k_clusters=kc,
                    top_p=top_p,
                    kmeans_iters=2, # Conservative estimate
                    max_k_clusters_per_q=cap
                )
            )
            
            # Formatting
            speedup_sdpa = sdpa_ms / svg2_ms if svg2_ms > 0 and sdpa_ms != float('inf') else 0
            if speedup_sdpa > 1: vs_sdpa_str = f"{speedup_sdpa:.2f}x faster"
            elif speedup_sdpa > 0: vs_sdpa_str = f"{1/speedup_sdpa:.2f}x slower"
            else: vs_sdpa_str = "N/A"

            speedup_dense = dense_ms / svg2_ms if svg2_ms > 0 and dense_ms != float('inf') else 0
            if speedup_dense > 1: vs_dense_str = f"{speedup_dense:.2f}x faster"
            elif dense_ms == float('inf'): vs_dense_str = "        Inf"
            else: vs_dense_str = f"{1/speedup_dense:.2f}x slower"

            print(f"    {desc:40s} | {svg2_ms:12.2f} | {vs_sdpa_str:>12s} | {vs_dense_str:>12s}")

        except RuntimeError as e:
            print(f"    {desc:40s} |        Error | {str(e)}")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "wan", "hunyuan", "scaling"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        return

    print("Initializing SVG2 Components...")
    comps = import_svg2_components()
    
    # --- 1. Wan2.1 Scenario ---
    if args.scenario in ["all", "wan"]:
        # Wan2.1-14B: 21 frames * 3600 tokens
        # H=40, D=128
        run_model_benchmark(
            "Wan2.1-I2V-14B (720p)",
            B=1, H=40, D=128, S=75600,
            svg2_components=comps
        )

    # --- 2. HunyuanVideo Scenario ---
    if args.scenario in ["all", "hunyuan"]:
        # HunyuanVideo-13B: 33 frames * 3600 tokens
        # H=48, D=128
        run_model_benchmark(
            "HunyuanVideo-T2V-13B (720p)",
            B=1, H=48, D=128, S=118800,
            svg2_components=comps
        )

    # --- 3. Scaling Curve (Standard) ---
    if args.scenario in ["all", "scaling"]:
        print_header("Standard Scaling Curve (H=24, D=128)")
        lengths = [16384, 32768, 65536, 96000]
        for S in lengths:
            run_model_benchmark(
                f"Scaling S={S}",
                B=1, H=24, D=128, S=S,
                svg2_components=comps
            )

if __name__ == "__main__":
    main()