#!/usr/bin/env python3
"""
SVG2 Sparse Attention WAN I2V Inference Script

This script demonstrates how to use SVG2 sparse attention with WAN I2V model
in sglang-diffusion.

Usage:
    python svg2_wan_i2v_inference.py \
        --model-path "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers" \
        --prompt "A dog running in the park" \
        --image-path "input_image.jpg" \
        --attention-backend svg2_sparse_attn
"""

import argparse
import os
import sys

# Add sglang to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

from sglang.multimodal_gen import DiffGenerator, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="SVG2 WAN I2V Inference")
    
    # Model arguments
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        help="Path to the model or HuggingFace model ID"
    )
    
    # Input arguments
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--image-path", 
        type=str, 
        required=True,
        help="Path to the input image for I2V"
    )
    # Default negative prompt from Sparse-VideoGen
    DEFAULT_NEGATIVE_PROMPT = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
        "paintings, images, static, overall gray, worst quality, low quality, "
        "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
        "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
        "still picture, messy background, three legs, many people in the background, walking backwards"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt to guide generation"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num-inference-steps", 
        type=int, 
        default=40,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720p",
        choices=["480p", "720p"],
        help="Output resolution"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second for output video"
    )
    
    # SVG2 specific parameters (matching original SVG)
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="svg2_sparse_attn",
        help="Attention backend to use"
    )
    parser.add_argument(
        "--num-q-clusters",
        type=int,
        default=300,
        help="Number of query clusters for K-Means (qc_kmeans in SVG)"
    )
    parser.add_argument(
        "--num-k-clusters",
        type=int,
        default=1000,
        help="Number of key clusters for K-Means (kc_kmeans in SVG)"
    )
    parser.add_argument(
        "--top-p-kmeans",
        type=str,
        default="0.9",
        help="Top-p for block mask selection. Supports single value or comma-separated list (e.g., '0.5,0.7,0.9')"
    )
    parser.add_argument(
        "--min-kc-ratio",
        type=float,
        default=0.10,
        help="Minimum ratio of key clusters to keep"
    )
    parser.add_argument(
        "--kmeans-iter-init",
        type=int,
        default=50,
        help="K-Means iterations for initialization"
    )
    parser.add_argument(
        "--kmeans-iter-step",
        type=int,
        default=2,
        help="K-Means iterations per step"
    )
    parser.add_argument(
        "--first-times-fp",
        type=float,
        default=0.35,
        help="Fraction of initial timesteps using full attention"
    )
    parser.add_argument(
        "--first-layers-fp",
        type=float,
        default=0.03,
        help="Fraction of initial layers using full attention"
    )
    
    # Hardware settings
    parser.add_argument(
        "--num-gpus", 
        type=int, 
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Ulysses parallelism degree"
    )
    parser.add_argument(
        "--ring-degree",
        type=int,
        default=1,
        help="Ring attention degree"
    )
    
    # Offload settings
    parser.add_argument(
        "--text-encoder-cpu-offload",
        action="store_true",
        help="Offload text encoder to CPU"
    )
    parser.add_argument(
        "--pin-cpu-memory",
        action="store_true",
        help="Pin CPU memory for faster transfers"
    )
    
    # Comparison
    parser.add_argument(
        "--compare-with-dense",
        action="store_true",
        help="Run comparison with dense attention (flash_attn_2)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-path", 
        type=str, 
        default="outputs/svg2",
        help="Directory to save output videos"
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="Output filename (default: auto-generated)"
    )
    
    return parser.parse_args()


def get_resolution_dims(resolution: str):
    """Get height and width for a given resolution."""
    if resolution == "480p":
        return 480, 832
    elif resolution == "720p":
        return 720, 1280
    else:
        raise ValueError(f"Unknown resolution: {resolution}")


def main():
    args = parse_args()
    
    # Get resolution dimensions
    height, width = get_resolution_dims(args.resolution)
    
    # Parse top-p values (support comma-separated list)
    top_p_values = [float(x.strip()) for x in args.top_p_kmeans.split(',')]
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Generate base output filename if not specified
    if args.output_filename is None:
        import hashlib
        prompt_hash = hashlib.md5(args.prompt.encode()).hexdigest()[:8]
        args.output_filename = f"svg2_i2v_{args.resolution}_{prompt_hash}.mp4"
    
    print("=" * 70)
    print("SVG2 WAN I2V Inference - Multi-Config Test")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Resolution: {args.resolution} ({height}x{width})")
    print(f"Attention Backend: {args.attention_backend}")
    print(f"Inference Steps: {args.num_inference_steps}")
    print(f"SVG2 Base Config:")
    print(f"  - Q Clusters: {args.num_q_clusters}")
    print(f"  - K Clusters: {args.num_k_clusters}")
    print(f"  - Top-p Values to Test: {top_p_values}")
    print(f"  - First Times FP: {args.first_times_fp}")
    print(f"  - First Layers FP: {args.first_layers_fp}")
    print(f"Output Dir: {args.output_path}")
    print("=" * 70)
    
    # Define experiments
    # (backend_name, display_name, filename_suffix, top_p_value)
    experiments = []
    
    # Add SVG2 experiments for each top-p value
    for top_p in top_p_values:
        label = f"SVG2 (p={top_p})"
        suffix = f"_p{top_p:.1f}".replace('.', '')  # e.g., _p09 for 0.9
        experiments.append((args.attention_backend, label, suffix, top_p))
    
    # Optionally add dense baseline
    if args.compare_with_dense:
        experiments.append(("fa2", "Dense (FlashAttn2)", "_dense", None))
    
    benchmark_results = {}

    for backend, label, suffix, top_p in experiments:
        print(f"\n{'='*50}")
        print(f" Running Inference: {label}")
        print(f"{'='*50}")
        if top_p is not None:
            print(f" Top-p: {top_p}")
        
        # Determine output filename for this run
        current_filename = args.output_filename
        if suffix:
            name, ext = os.path.splitext(current_filename)
            current_filename = f"{name}{suffix}{ext}"
        
        current_output_path = os.path.join(args.output_path, current_filename)

        # Prepare generator kwargs
        gen_kwargs = dict(
            model_path=args.model_path,
            num_gpus=args.num_gpus,
            ulysses_degree=args.ulysses_degree,
            ring_degree=args.ring_degree,
            attention_backend=backend,
            text_encoder_cpu_offload=args.text_encoder_cpu_offload,
            pin_cpu_memory=args.pin_cpu_memory,
        )
        
        # Add SVG2-specific parameters if using sparse attention
        if backend == args.attention_backend and top_p is not None:
            gen_kwargs.update({
                'num_q_clusters': args.num_q_clusters,
                'num_k_clusters': args.num_k_clusters,
                'top_p_kmeans': top_p,
                'min_kc_ratio': args.min_kc_ratio,
                'kmeans_iter_init': args.kmeans_iter_init,
                'kmeans_iter_step': args.kmeans_iter_step,
                'first_times_fp': args.first_times_fp,
                'first_layers_fp': args.first_layers_fp,
            })

        # Create generator
        try:
            generator = DiffGenerator.from_pretrained(**gen_kwargs)
        except Exception as e:
            print(f"Failed to initialize generator: {e}")
            continue

        # Generate video
        print("\nStarting generation...")
        import time
        import gc
        import torch
        
        # Clear cache before generation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        try:
            result = generator.generate(
                sampling_params_kwargs=dict(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    image_path=args.image_path,  # Use image_path, not image
                    height=height,
                    width=width,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                    fps=args.fps,
                    output_path=args.output_path,
                    output_file_name=current_filename,
                    save_output=True,
                    return_frames=False,
                )
            )
            
            # Sync for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"\n{label} Generation completed in {elapsed:.2f} seconds")
            print(f"Output saved to: {current_output_path}")
            
            benchmark_results[label] = elapsed
            
        except Exception as e:
            print(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        if 'generator' in locals():
            generator.shutdown()
            del generator
        
        # Aggressive cleanup to prevent OOM
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Optional: wait a bit for memory to settle
        time.sleep(2)
        
        # Print memory status
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"Memory after cleanup: {free_mem/1024**3:.2f}GB / {total_mem/1024**3:.2f}GB free")

    # Print Comparison Summary
    if len(benchmark_results) > 1:
        print("\n" + "=" * 80)
        print(" Performance Comparison Summary")
        print("=" * 80)
        print(f" {'Configuration':<30} | {'Time (s)':<12} | {'Speedup vs Dense':<18}")
        print("-" * 80)
        
        # Get dense baseline time if available
        dense_time = benchmark_results.get("Dense (FlashAttn2)", None)
        
        # Sort results: Dense first (if exists), then SVG2 configs by top-p
        sorted_results = []
        if "Dense (FlashAttn2)" in benchmark_results:
            sorted_results.append(("Dense (FlashAttn2)", benchmark_results["Dense (FlashAttn2)"]))
        
        # Add SVG2 results sorted by top-p value
        svg2_results = [(k, v) for k, v in benchmark_results.items() if k != "Dense (FlashAttn2)"]
        svg2_results.sort(key=lambda x: float(x[0].split('p=')[1].rstrip(')')))  # Sort by top-p value
        sorted_results.extend(svg2_results)
        
        for label, duration in sorted_results:
            if label == "Dense (FlashAttn2)":
                speedup_str = "Baseline (1.00x)"
            elif dense_time is not None:
                speedup = dense_time / duration if duration > 0 else 0
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"
            
            print(f" {label:<30} | {duration:<12.2f} | {speedup_str:<18}")
        
        print("=" * 80)
        
        # Additional analysis for SVG2 configs
        if len(svg2_results) > 1:
            print("\n SVG2 Top-p Trade-off Analysis:")
            print(" " + "-" * 78)
            fastest_svg2 = min(svg2_results, key=lambda x: x[1])
            slowest_svg2 = max(svg2_results, key=lambda x: x[1])
            print(f"   Fastest SVG2 Config: {fastest_svg2[0]} ({fastest_svg2[1]:.2f}s)")
            print(f"   Slowest SVG2 Config: {slowest_svg2[0]} ({slowest_svg2[1]:.2f}s)")
            variance = (slowest_svg2[1] - fastest_svg2[1]) / fastest_svg2[1] * 100
            print(f"   Performance Variance: {variance:.1f}%")
            print(" " + "-" * 78)
    elif len(benchmark_results) == 1:
        print("\n" + "=" * 80)
        print(" Generation Summary")
        print("=" * 80)
        for label, duration in benchmark_results.items():
            print(f" {label}: {duration:.2f} seconds")
        print("=" * 80)


if __name__ == "__main__":
    main()

