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
model_id=${MODEL_ID:-"/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_libra/user_spanaluo/opensource_model/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"}

# Sparse-VideoGen base path (for reading example prompts)
SVG_BASE=${SVG_BASE:-"/root/Sparse-VideoGen"}

# Default prompt ID
prompt_id=${PROMPT_ID:-2}

# Try to read prompt from file, or use default
if [ -f "${SVG_BASE}/examples/${prompt_id}/prompt.txt" ]; then
    prompt=$(cat "${SVG_BASE}/examples/${prompt_id}/prompt.txt")
else
    prompt=${PROMPT:-"A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."}
fi

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

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run with sglang CLI
sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"

echo ""
echo "Output saved to: ${output_dir}/${output_file}"

