#!/bin/bash
# End-to-end KV cache quantization benchmark.
#
# This script:
# 1. Prepares data (if not already done)
# 2. Starts sglang server (baseline), runs benchmark
# 3. Applies patch, restarts server, runs benchmark
# 4. Analyzes results
#
# Usage: bash run_all.sh --model-path /path/to/Qwen3.5-4B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${1:-/vllm-workspace/models/Qwen3.5-4B}"
PORT=30000
DATA_FILE="$SCRIPT_DIR/data_2k.jsonl"
BASELINE_OUTPUT="$SCRIPT_DIR/results_baseline.jsonl"
QUANT_OUTPUT="$SCRIPT_DIR/results_quant_fp8.jsonl"

echo "=========================================="
echo "KV Cache Quantization Benchmark"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo ""

# Step 1: Prepare data
if [ ! -f "$DATA_FILE" ]; then
    echo "[Step 1] Preparing dataset..."
    python "$SCRIPT_DIR/prepare_data.py" \
        --model-path "$MODEL_PATH" \
        --output "$DATA_FILE"
else
    echo "[Step 1] Dataset already exists: $DATA_FILE"
fi

# Step 2: Run baseline (no quantization)
echo ""
echo "[Step 2] Starting baseline server..."
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port $PORT \
    --dtype bfloat16 \
    --disable-radix-cache &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 2
done

echo "Running baseline benchmark..."
python "$SCRIPT_DIR/run_benchmark.py" \
    --data "$DATA_FILE" \
    --port $PORT \
    --output "$BASELINE_OUTPUT"

# Kill baseline server
echo "Stopping baseline server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 5

# Step 3: Apply patch and run quantized version
echo ""
echo "[Step 3] Applying KV cache quantization patch..."
bash "$SCRIPT_DIR/apply_patch.sh"

echo "Starting quantized server..."
SGLANG_KV_QUANT_BENCHMARK=1 SGLANG_KV_QUANT_LAST_N=128 \
    python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port $PORT \
    --dtype bfloat16 \
    --disable-radix-cache &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 2
done

echo "Running quantized benchmark..."
python "$SCRIPT_DIR/run_benchmark.py" \
    --data "$DATA_FILE" \
    --port $PORT \
    --output "$QUANT_OUTPUT"

# Kill server
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Step 4: Revert patch
echo ""
echo "[Step 4] Reverting patch..."
bash "$SCRIPT_DIR/apply_patch.sh" --revert

# Step 5: Analyze
echo ""
echo "[Step 5] Analyzing results..."
python "$SCRIPT_DIR/analyze_results.py" \
    --baseline "$BASELINE_OUTPUT" \
    --experiment "$QUANT_OUTPUT"

echo ""
echo "Done! Results saved to:"
echo "  Baseline: $BASELINE_OUTPUT"
echo "  Quantized: $QUANT_OUTPUT"
