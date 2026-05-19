#!/bin/bash
# Apply KV cache quantization hook to the installed sglang package.
# This script patches model_runner.py to call our hook after forward_extend.
#
# Usage: bash apply_patch.sh
# To revert: bash apply_patch.sh --revert

SGLANG_DIR=$(python -c "import sglang; import os; print(os.path.dirname(sglang.__file__))")
MODEL_RUNNER="$SGLANG_DIR/srt/model_executor/model_runner.py"
HOOK_SRC="$(dirname $0)/kv_cache_quant_hook.py"
HOOK_DST="$SGLANG_DIR/srt/layers/attention/kv_cache_quant_hook.py"

if [ "$1" == "--revert" ]; then
    echo "Reverting patch..."
    if [ -f "${MODEL_RUNNER}.bak" ]; then
        cp "${MODEL_RUNNER}.bak" "$MODEL_RUNNER"
        rm -f "$HOOK_DST"
        echo "Reverted successfully."
    else
        echo "No backup found, nothing to revert."
    fi
    exit 0
fi

echo "Sglang directory: $SGLANG_DIR"
echo "Model runner: $MODEL_RUNNER"

# Backup original
cp "$MODEL_RUNNER" "${MODEL_RUNNER}.bak"

# Copy hook module
cp "$HOOK_SRC" "$HOOK_DST"
echo "Copied hook to: $HOOK_DST"

# Find the line where forward_extend result is assigned in _forward_raw
# Pattern: "ret, can_run_graph = self.forward_extend("
LINE_NUM=$(grep -n "ret, can_run_graph = self.forward_extend(" "$MODEL_RUNNER" | head -1 | cut -d: -f1)

if [ -z "$LINE_NUM" ]; then
    echo "ERROR: Could not find forward_extend call in _forward_raw"
    exit 1
fi

echo "Found forward_extend call at line $LINE_NUM"

# We need to find the closing parenthesis of this call (it may span multiple lines)
# Find the next line that starts with ")" or contains the full call
# Look for the elif/else after the forward_extend block
INJECT_LINE=$(awk -v start="$LINE_NUM" '
NR > start && /^[[:space:]]*(elif|else:)/ { print NR; exit }
' "$MODEL_RUNNER")

if [ -z "$INJECT_LINE" ]; then
    echo "ERROR: Could not find injection point after forward_extend"
    exit 1
fi

echo "Injecting hook before line $INJECT_LINE"

# Create the patch content
PATCH_CONTENT='        # === KV Cache Quantization Benchmark Hook ===
        if (
            os.environ.get("SGLANG_KV_QUANT_BENCHMARK") == "1"
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_idle()
        ):
            from sglang.srt.layers.attention.kv_cache_quant_hook import apply_kv_quant_hook
            apply_kv_quant_hook(forward_batch, self.model_config.num_hidden_layers)
        # === End Hook ==='

# Also need to add 'import os' at the top if not already there
if ! grep -q "^import os" "$MODEL_RUNNER"; then
    sed -i '1s/^/import os\n/' "$MODEL_RUNNER"
    INJECT_LINE=$((INJECT_LINE + 1))
fi

# Insert the hook code before the elif line
sed -i "${INJECT_LINE}i\\
        # === KV Cache Quantization Benchmark Hook ===\\
        if (\\
            os.environ.get(\"SGLANG_KV_QUANT_BENCHMARK\") == \"1\"\\
            and forward_batch.forward_mode.is_extend()\\
        ):\\
            from sglang.srt.layers.attention.kv_cache_quant_hook import apply_kv_quant_hook\\
            apply_kv_quant_hook(forward_batch, self.model_config.num_hidden_layers)\\
        # === End Hook ===" "$MODEL_RUNNER"

echo "Patch applied successfully!"
echo "To verify: grep -A5 'KV Cache Quantization' $MODEL_RUNNER"
