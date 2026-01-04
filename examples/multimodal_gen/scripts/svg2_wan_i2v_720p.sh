#!/bin/bash
# SVG2 WAN I2V 720p Inference Script - Multi-Config Top-p Testing
# 
# This script runs WAN I2V inference with SVG2 sparse attention,
# testing multiple top-p configurations by running separate processes
# to avoid OOM issues.

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
top_p_k=${TOP_P_VALUES:-"0.3,0.5,0.7,0.9"}  # Top-p values to test (comma-separated)
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
centroid_cfg="QC_${qc_kmeans}-KC_${kc_kmeans}-TopP_Multi" 
kmeans_cfg="Init_${kmeans_iter_init}-Step_${kmeans_iter_step}-MinR_${min_kc_ratio}"
output_feature="${video_cfg}/${dense_attention_cfg}/${centroid_cfg}/${kmeans_cfg}"
full_output_dir="${output_dir}/${output_feature}"

# ===== Hardware Settings =====
num_gpus=${NUM_GPUS:-1}

# ===== Comparison Settings =====
compare_with_dense=${COMPARE_WITH_DENSE:-false}

# ===== Model Settings =====
model_id=${MODEL_ID:-"/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_libra/user_spanaluo/opensource_model/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"}

# ===== Helper Function =====
run_inference() {
    local backend=$1
    local top_p=$2
    
    echo "----------------------------------------------------------------"
    echo "Running Inference with Backend: $backend"
    if [ -n "$top_p" ]; then
        echo "Top-p: $top_p"
    fi
    echo "----------------------------------------------------------------"

    # Build base arguments
    local args=(
        python "${SCRIPT_DIR}/../svg2_wan_i2v_inference.py"
        --model-path "$model_id"
        --prompt "$prompt"
        --image-path "$image_path"
        --seed 0
        --num-inference-steps $infer_step
        --resolution $resolution
        --fps $fps
        --attention-backend "$backend"
        --num-gpus $num_gpus
        --text-encoder-cpu-offload
        --pin-cpu-memory
        --output-path "$full_output_dir"
        --output-filename "${prompt_id}-0.mp4"
    )

    # Add SVG2 arguments if backend is svg2_sparse_attn
    if [ "$backend" == "svg2_sparse_attn" ]; then
        args+=(
            --num-q-clusters $qc_kmeans
            --num-k-clusters $kc_kmeans
            --top-p-kmeans "$top_p"
            --min-kc-ratio $min_kc_ratio
            --kmeans-iter-init $kmeans_iter_init
            --kmeans-iter-step $kmeans_iter_step
            --first-times-fp $first_times_fp
            --first-layers-fp $first_layers_fp
        )
    fi

    # Run command
    "${args[@]}"
    
    local status=$?
    if [ $status -eq 0 ]; then
        echo "Successfully finished: $backend ${top_p}"
    else
        echo "Failed: $backend ${top_p}"
        # We don't exit here to allow other configurations to try running
    fi
    
    # Sleep briefly to ensure OS reclaims all resources
    sleep 5
}

# ===== Main Execution Loop =====

echo "========================================"
echo " SVG2 WAN I2V 720p Inference (Safe Mode)"
echo " Output: $full_output_dir"
echo "========================================"

# Create output directory
mkdir -p "$full_output_dir"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Run SVG2 for each top_p value
IFS=',' read -ra TOP_P_ARRAY <<< "$top_p_k"
for p in "${TOP_P_ARRAY[@]}"; do
    # Remove whitespace
    p_clean=$(echo "$p" | tr -d ' ')
    if [ -n "$p_clean" ]; then
        run_inference "svg2_sparse_attn" "$p_clean"
    fi
done

# 2. Run Dense Baseline if requested
if [ "$compare_with_dense" = true ]; then
    run_inference "fa2" ""
fi

echo ""
echo "==========================================="
echo " All Tasks Complete!"
echo "==========================================="
