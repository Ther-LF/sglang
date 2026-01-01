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
        type=float,
        default=0.9,
        help="Top-p for block mask selection"
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
    
    # Calculate actual layer/timestep thresholds
    # These will be passed to the model through attention metadata
    # For now, we'll use the default mechanism in SVG2SparseAttentionImpl
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Generate output filename if not specified
    if args.output_filename is None:
        import hashlib
        prompt_hash = hashlib.md5(args.prompt.encode()).hexdigest()[:8]
        args.output_filename = f"svg2_i2v_{args.resolution}_{prompt_hash}.mp4"
    
    output_file = os.path.join(args.output_path, args.output_filename)
    
    print("=" * 70)
    print("SVG2 WAN I2V Inference")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Resolution: {args.resolution} ({height}x{width})")
    print(f"Attention Backend: {args.attention_backend}")
    print(f"Inference Steps: {args.num_inference_steps}")
    print(f"SVG2 Config:")
    print(f"  - Q Clusters: {args.num_q_clusters}")
    print(f"  - K Clusters: {args.num_k_clusters}")
    print(f"  - Top-p: {args.top_p_kmeans}")
    print(f"  - First Times FP: {args.first_times_fp}")
    print(f"  - First Layers FP: {args.first_layers_fp}")
    print(f"Output: {output_file}")
    print("=" * 70)
    
    # Create generator with SVG2 attention
    generator = DiffGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        attention_backend=args.attention_backend,
        text_encoder_cpu_offload=args.text_encoder_cpu_offload,
        pin_cpu_memory=args.pin_cpu_memory,
    )
    
    # Generate video
    print("\nStarting generation...")
    import time
    start_time = time.time()
    
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
            output_file_name=args.output_filename,
            save_output=True,
            return_frames=False,
        )
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nGeneration completed in {elapsed:.2f} seconds")
    print(f"Output saved to: {output_file}")
    
    # Cleanup
    generator.shutdown()


if __name__ == "__main__":
    main()

