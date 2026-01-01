#!/bin/bash
# SVG2 WAN I2V 720p Inference Script
# 
# This script runs WAN I2V inference with SVG2 sparse attention,
# matching the configuration from Sparse-VideoGen/scripts/wan/wan_i2v_720p_sap.sh
#
# Usage:
#   ./svg2_wan_i2v_720p.sh
#
# Or with custom prompt:
#   PROMPT="Your custom prompt" IMAGE_PATH="your_image.jpg" ./svg2_wan_i2v_720p.sh

set -e

# ===== Configuration (matching SVG original) =====

# Video settings
resolution="720p"
infer_step=40
fps=16

# Dense attention warm-up (fraction of layers/timesteps using full attention)
first_times_fp=0.35
first_layers_fp=0.03

# K-Means clustering parameters
qc_kmeans=300      # Number of query clusters
kc_kmeans=1000     # Number of key clusters
top_p_k=0.9        # Top-p for block mask selection
min_kc_ratio=0.10  # Minimum ratio of key clusters to keep

# K-Means iteration settings
kmeans_iter_init=50
kmeans_iter_step=2

# ===== Input Settings =====

# Sparse-VideoGen base path
SVG_BASE=${SVG_BASE:-"/root/Sparse-VideoGen"}

# Default prompt (can be overridden by environment variable)
prompt_id=${PROMPT_ID:-1}

# Try to read prompt from file, or use default
if [ -f "${SVG_BASE}/examples/${prompt_id}/prompt.txt" ]; then
    prompt=$(cat "${SVG_BASE}/examples/${prompt_id}/prompt.txt")
else
    prompt=${PROMPT:-"A curious raccoon explores a sunlit forest clearing, its eyes bright with wonder."}
fi

# Input image path
image_path=${IMAGE_PATH:-"${SVG_BASE}/examples/${prompt_id}/image.jpg"}

# Check if image exists
if [ ! -f "$image_path" ]; then
    echo "Warning: Image not found at $image_path"
    echo "Please provide a valid image path via IMAGE_PATH environment variable"
    echo "Or set SVG_BASE to point to your Sparse-VideoGen directory"
    exit 1
fi

# ===== Output Settings =====

output_dir="outputs/svg2/wan_i2v"

# Build output path with configuration info
video_cfg="Step_${infer_step}-Res_${resolution}"
dense_attention_cfg="TFP_${first_times_fp}-LFP_${first_layers_fp}"
centroid_cfg="QC_${qc_kmeans}-KC_${kc_kmeans}-TopP_${top_p_k}"
kmeans_cfg="Init_${kmeans_iter_init}-Step_${kmeans_iter_step}-MinR_${min_kc_ratio}"
output_feature="${video_cfg}/${dense_attention_cfg}/${centroid_cfg}/${kmeans_cfg}"

# ===== Hardware Settings =====
num_gpus=${NUM_GPUS:-1}

# ===== Model Settings =====
model_id=${MODEL_ID:-"Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"}

# ===== Run Inference =====

echo "========================================"
echo " SVG2 WAN I2V 720p Inference"
echo "========================================"
echo "Model: $model_id"
echo "Resolution: $resolution"
echo "Steps: $infer_step"
echo "Prompt: $prompt"
echo "Image: $image_path"
echo ""
echo "SVG2 Configuration:"
echo "  Q Clusters: $qc_kmeans"
echo "  K Clusters: $kc_kmeans"
echo "  Top-p: $top_p_k"
echo "  First Times FP: $first_times_fp"
echo "  First Layers FP: $first_layers_fp"
echo "========================================"

# Create output directory
mkdir -p "${output_dir}/${output_feature}"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Run the Python script
python "${SCRIPT_DIR}/../svg2_wan_i2v_inference.py" \
    --model-path "$model_id" \
    --prompt "$prompt" \
    --image-path "$image_path" \
    --seed 0 \
    --num-inference-steps $infer_step \
    --resolution $resolution \
    --fps $fps \
    --attention-backend svg2_sparse_attn \
    --num-q-clusters $qc_kmeans \
    --num-k-clusters $kc_kmeans \
    --top-p-kmeans $top_p_k \
    --min-kc-ratio $min_kc_ratio \
    --kmeans-iter-init $kmeans_iter_init \
    --kmeans-iter-step $kmeans_iter_step \
    --first-times-fp $first_times_fp \
    --first-layers-fp $first_layers_fp \
    --num-gpus $num_gpus \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --output-path "${output_dir}/${output_feature}" \
    --output-filename "${prompt_id}-0.mp4"

echo ""
echo "Output saved to: ${output_dir}/${output_feature}/${prompt_id}-0.mp4"

