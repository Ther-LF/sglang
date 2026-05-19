"""KV Cache quantization/dequantization hook for benchmarking precision loss.

This module provides a hook that can be called after prefill to quantize
the last N tokens of a request's KV cache from BF16 to FP8 and back,
simulating the precision loss of stored gist-token KV caches.
"""

import os
import torch
from typing import Optional


def quantize_dequantize_fp8(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to FP8 E4M3 and immediately dequantize back to original dtype.

    Uses per-tensor absmax scaling to maximize dynamic range utilization.

    Args:
        tensor: Input tensor in BF16/FP16 format.

    Returns:
        Tensor after BF16 -> FP8_E4M3 -> BF16 round-trip.
    """
    original_dtype = tensor.dtype
    # FP8 E4M3 max representable value is 448.0
    FP8_MAX = 448.0
    amax = tensor.abs().amax()
    if amax == 0:
        return tensor
    scale = FP8_MAX / amax
    # Scale, cast to FP8, cast back, unscale
    scaled = tensor.float() * scale
    fp8 = scaled.to(torch.float8_e4m3fn)
    dequantized = fp8.to(original_dtype) / scale
    return dequantized


def apply_kv_quant_hook(forward_batch, num_layers: int):
    """Apply FP8 quantize/dequantize to the last N tokens of each request's KV cache.

    Should be called after forward_extend completes (KV cache is populated).
    Only operates on layers that have accessible KV buffers (full attention layers).

    Args:
        forward_batch: The ForwardBatch object containing KV pool references.
        num_layers: Number of transformer layers.
    """
    last_n_tokens = int(os.environ.get("SGLANG_KV_QUANT_LAST_N", "128"))

    token_to_kv_pool = forward_batch.token_to_kv_pool
    req_to_token_pool = forward_batch.req_to_token_pool

    batch_size = forward_batch.batch_size
    seq_lens = forward_batch.seq_lens
    req_pool_indices = forward_batch.req_pool_indices

    # Determine which layers have KV buffers available
    # Some models (e.g., Qwen3.5) have mixed attention types where only
    # certain layers (full_attention) store KV cache in the standard pool.
    # Check if the pool has a full_attention_layer_id_mapping (SWA pool)
    available_layers = []
    if hasattr(token_to_kv_pool, 'full_attention_layer_id_mapping'):
        available_layers = list(token_to_kv_pool.full_attention_layer_id_mapping.keys())
    else:
        # Standard pool: all layers are available
        for layer_id in range(num_layers):
            try:
                k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
                if k_buffer is not None:
                    available_layers.append(layer_id)
            except (ValueError, KeyError, IndexError):
                continue

    if not available_layers:
        return

    for i in range(batch_size):
        seq_len = seq_lens[i].item()
        req_pool_idx = req_pool_indices[i].item()

        # Get token indices for the last N positions
        start_pos = max(0, seq_len - last_n_tokens)
        token_indices = req_to_token_pool.req_to_token[req_pool_idx, start_pos:seq_len]

        for layer_id in available_layers:
            # Get K and V buffers for this layer
            k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
            v_buffer = token_to_kv_pool.get_value_buffer(layer_id)

            # Quantize/dequantize in-place for the selected token positions
            k_buffer[token_indices] = quantize_dequantize_fp8(k_buffer[token_indices])
            v_buffer[token_indices] = quantize_dequantize_fp8(v_buffer[token_indices])
