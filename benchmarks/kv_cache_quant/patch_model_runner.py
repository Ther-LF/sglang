"""Monkey-patch sglang's model runner to add KV cache quantization hook.

Run this before starting the sglang server when SGLANG_KV_QUANT_BENCHMARK=1.
This patches ModelRunner._forward_raw to apply FP8 quantize/dequantize
on the last N tokens' KV cache after each prefill (extend) operation.

Usage:
    SGLANG_KV_QUANT_BENCHMARK=1 python -c "import patch_model_runner" && python -m sglang.launch_server ...

Or more practically, this is imported at the top of run_benchmark.py when
launching the server subprocess with the hook enabled.
"""

import os
import sys

# Add the benchmark directory to path so kv_cache_quant_hook can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def patch():
    """Apply the KV cache quantization monkey-patch to ModelRunner._forward_raw."""
    from sglang.srt.model_executor.model_runner import ModelRunner
    from kv_cache_quant_hook import apply_kv_quant_hook

    original_forward_raw = ModelRunner._forward_raw

    def patched_forward_raw(self, forward_batch, *args, **kwargs):
        result = original_forward_raw(self, forward_batch, *args, **kwargs)

        # Apply quantization hook after extend (prefill) operations only
        if (
            os.environ.get("SGLANG_KV_QUANT_BENCHMARK") == "1"
            and forward_batch.forward_mode.is_extend()
        ):
            num_layers = self.model_config.num_hidden_layers
            apply_kv_quant_hook(forward_batch, num_layers)

        return result

    ModelRunner._forward_raw = patched_forward_raw
    print("[KV Quant Benchmark] Monkey-patch applied to ModelRunner._forward_raw")


if os.environ.get("SGLANG_KV_QUANT_BENCHMARK") == "1":
    patch()
