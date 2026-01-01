# SVG2 Sparse Attention Examples

This directory contains scripts to run WAN video generation with SVG2 (Semantic-Aware Sparse) Attention.

## Overview

SVG2 is a sparse attention mechanism that uses K-Means clustering and semantic-aware permutation to accelerate video generation. It was originally developed in the [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen) project.

## Quick Start

### Text-to-Video (T2V)

```bash
# Basic usage with sglang CLI
./svg2_wan_t2v_720p.sh

# Custom prompt
PROMPT="A majestic eagle soaring over mountains" ./svg2_wan_t2v_720p.sh

# Multiple GPUs
NUM_GPUS=4 ./svg2_wan_t2v_720p.sh
```

### Image-to-Video (I2V)

```bash
# Requires an input image
IMAGE_PATH="your_image.jpg" PROMPT="A cat playing with a ball" ./svg2_wan_i2v_720p.sh
```

## Configuration

### SVG2 Parameters (matching original SVG)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_q_clusters` | 300 | Number of query clusters for K-Means |
| `num_k_clusters` | 1000 | Number of key clusters for K-Means |
| `top_p_kmeans` | 0.9 | Top-p for block mask selection |
| `first_times_fp` | 0.35 | Fraction of early timesteps using full attention |
| `first_layers_fp` | 0.03 | Fraction of early layers using full attention |

### Warm-up Strategy

The `first_times_fp` and `first_layers_fp` parameters control when full attention is used:

- **first_layers_fp**: First N layers (e.g., 3% of layers) use full attention
- **first_times_fp**: Early timesteps (e.g., 35% of denoising steps) use full attention

This ensures quality during the critical early stages of generation while using sparse attention for the rest.

## Python API

```python
from sglang.multimodal_gen import DiffGenerator

# Create generator with SVG2 attention
generator = DiffGenerator.from_pretrained(
    model_path="Wan-AI/Wan2.1-T2V-14B-720P-Diffusers",
    attention_backend="svg2_sparse_attn",
    num_gpus=1,
)

# Generate video
result = generator.generate(
    sampling_params_kwargs=dict(
        prompt="A curious raccoon in a forest",
        num_inference_steps=40,
        save_output=True,
        output_path="outputs",
    )
)

generator.shutdown()
```

## Comparison with Original SVG

| Feature | Original SVG | sglang SVG2 |
|---------|-------------|-------------|
| K-Means | cuVS (RAPIDS) | Triton |
| Permutation | Triton | Triton |
| Block Sparse Attn | flashinfer (patched) | Masked Dense / SDPA |
| Full Attention Fallback | ✓ | ✓ |

## Notes

- The current implementation uses masked dense attention for the sparse blocks, which may not provide the same speedup as the original flashinfer-based implementation for very long sequences.
- For production use with long videos, consider integrating with flashinfer's `VariableBlockSparseAttentionWrapper`.

