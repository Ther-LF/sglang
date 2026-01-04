#!/bin/bash
# SVG2 WAN I2V 720p Inference Script - Multi-Config Top-p Testing
# 
# This script runs WAN I2V inference with SVG2 sparse attention,
# testing multiple top-p configurations IN PARALLEL on different GPUs.
#
# Usage:
#   # Run with 4 top-p values on 4 GPUs in parallel
#   TOP_P_VALUES="0.3,0.5,0.7,0.9" ./svg2_wan_i2v_720p.sh
#
#   # Also run dense baseline (will use next available GPU)
#   COMPARE_WITH_DENSE=true TOP_P_VALUES="0.3,0.5,0.7" ./svg2_wan_i2v_720p.sh

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
# Total available GPUs (default 8)
TOTAL_GPUS=${TOTAL_GPUS:-8}

# ===== Comparison Settings =====
compare_with_dense=${COMPARE_WITH_DENSE:-false}

# ===== Model Settings =====
model_id=${MODEL_ID:-"/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_libra/user_spanaluo/opensource_model/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"}

# ===== Helper Function =====
run_inference_on_gpu() {
    local gpu_id=$1
    local backend=$2
    local top_p=$3
    local log_file=$4
    
    echo "[GPU $gpu_id] Starting: $backend ${top_p:+top_p=$top_p}"

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
        --num-gpus 1
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

    # Run command on specific GPU, redirect output to log file
    CUDA_VISIBLE_DEVICES=$gpu_id "${args[@]}" > "$log_file" 2>&1
    
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[GPU $gpu_id] ✓ Finished: $backend ${top_p:+top_p=$top_p}"
    else
        echo "[GPU $gpu_id] ✗ Failed: $backend ${top_p:+top_p=$top_p} (see $log_file)"
    fi
    
    return $status
}

# ===== Main Execution =====

echo "========================================"
echo " SVG2 WAN I2V 720p Inference"
echo " PARALLEL Mode (Multi-GPU)"
echo "========================================"
echo "Available GPUs: $TOTAL_GPUS"
echo "Output: $full_output_dir"
echo "========================================"

# Create output directory
mkdir -p "$full_output_dir"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse top-p values into array
IFS=',' read -ra TOP_P_ARRAY <<< "$top_p_k"

# Calculate total tasks
total_tasks=${#TOP_P_ARRAY[@]}
if [ "$compare_with_dense" = true ]; then
    ((total_tasks++))
fi

echo "Total tasks: $total_tasks"
echo "Top-p values: ${TOP_P_ARRAY[*]}"
if [ "$compare_with_dense" = true ]; then
    echo "Dense baseline: enabled"
fi
echo "========================================"

# Check if we have enough GPUs
if [ $total_tasks -gt $TOTAL_GPUS ]; then
    echo "Warning: More tasks ($total_tasks) than GPUs ($TOTAL_GPUS)."
    echo "         Some tasks will be queued."
fi

# Array to track background process PIDs
declare -a PIDS=()
declare -a TASK_NAMES=()

# Function to wait for a GPU slot
wait_for_gpu_slot() {
    while [ ${#PIDS[@]} -ge $TOTAL_GPUS ]; do
        # Wait for any background process to finish
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                # Process finished, remove from array
                unset 'PIDS[i]'
                unset 'TASK_NAMES[i]'
                # Re-index arrays
                PIDS=("${PIDS[@]}")
                TASK_NAMES=("${TASK_NAMES[@]}")
                return
            fi
        done
        sleep 2
    done
}

# Get next available GPU ID
get_next_gpu() {
    echo ${#PIDS[@]}
}

# Launch all SVG2 tasks
echo ""
echo "Launching SVG2 tasks..."
for p in "${TOP_P_ARRAY[@]}"; do
    p_clean=$(echo "$p" | tr -d ' ')
    if [ -n "$p_clean" ]; then
        wait_for_gpu_slot
        gpu_id=$(get_next_gpu)
        log_file="${full_output_dir}/log_svg2_p${p_clean}.txt"
        
        run_inference_on_gpu "$gpu_id" "svg2_sparse_attn" "$p_clean" "$log_file" &
        PIDS+=($!)
        TASK_NAMES+=("SVG2 p=$p_clean")
    fi
done

# Launch dense baseline if requested
if [ "$compare_with_dense" = true ]; then
    wait_for_gpu_slot
    gpu_id=$(get_next_gpu)
    log_file="${full_output_dir}/log_dense.txt"
    
    echo ""
    echo "Launching Dense baseline..."
    run_inference_on_gpu "$gpu_id" "fa2" "" "$log_file" &
    PIDS+=($!)
    TASK_NAMES+=("Dense (FA2)")
fi

# Wait for all background processes to complete
echo ""
echo "========================================"
echo " All tasks launched. Waiting for completion..."
echo " Running tasks: ${#PIDS[@]}"
echo "========================================"

# Wait for all and collect exit statuses
declare -a EXIT_STATUSES=()
for pid in "${PIDS[@]}"; do
    wait $pid
    EXIT_STATUSES+=($?)
done

# Print summary
echo ""
echo "==========================================="
echo " All Tasks Complete!"
echo "==========================================="
echo ""
echo "Results Summary:"
echo "----------------"

success_count=0
fail_count=0

for i in "${!TASK_NAMES[@]}"; do
    if [ "${EXIT_STATUSES[$i]}" -eq 0 ]; then
        echo "  ✓ ${TASK_NAMES[$i]}"
        ((success_count++))
    else
        echo "  ✗ ${TASK_NAMES[$i]} (FAILED)"
        ((fail_count++))
    fi
done

echo ""
echo "Total: $success_count succeeded, $fail_count failed"
echo ""
echo "Output directory: $full_output_dir"
echo "Log files: ${full_output_dir}/log_*.txt"
echo "==========================================="

# Exit with error if any task failed
if [ $fail_count -gt 0 ]; then
    exit 1
fi
