#!/bin/bash
# SVG2 WAN T2V 720p Inference Script using sglang CLI
# 
# This script runs WAN T2V inference with SVG2 sparse attention.
#
# Usage:
#   ./svg2_wan_t2v_720p.sh
#
# Or with custom prompt:
#   PROMPT="Your custom prompt" ./svg2_wan_t2v_720p.sh

set -e

# ===== Configuration =====
resolution="720p"
infer_step=40

# Model
model_id=${MODEL_ID:-"Wan-AI/Wan2.1-T2V-14B-720P-Diffusers"}

# Prompt
prompt=${PROMPT:-"A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."}

# Output
output_dir="outputs/svg2/wan_t2v"
output_file="output.mp4"

# Hardware
num_gpus=${NUM_GPUS:-1}
ulysses_degree=1
ring_degree=1

if [ $num_gpus -gt 1 ]; then
    ulysses_degree=$num_gpus
fi

# ===== Run =====
echo "========================================"
echo " SVG2 WAN T2V 720p Inference"
echo "========================================"
echo "Model: $model_id"
echo "Attention: svg2_sparse_attn"
echo "GPUs: $num_gpus"
echo "Prompt: $prompt"
echo "========================================"

mkdir -p "$output_dir"

# Server arguments
SERVER_ARGS=(
    --model-path "$model_id"
    --attention-backend svg2_sparse_attn
    --text-encoder-cpu-offload
    --pin-cpu-memory
    --num-gpus $num_gpus
    --ulysses-degree $ulysses_degree
    --ring-degree $ring_degree
)

# Sampling arguments
SAMPLING_ARGS=(
    --prompt "$prompt"
    --num-inference-steps $infer_step
    --save-output
    --output-path "$output_dir"
    --output-file-name "$output_file"
)

# Run with sglang CLI
sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"

echo ""
echo "Output saved to: ${output_dir}/${output_file}"

