"""
ResQ Mixed-Precision Quantization Config and Linear Method for SGLang.

Implements W4A4/W8A8 mixed-precision quantization following the ResQ method.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.resq.resq_utils import resq_linear_forward
from sglang.srt.utils import set_weight_attrs


class ResQConfig(QuantizationConfig):
    """Config for ResQ mixed-precision quantization.

    ResQ splits activations into high-variance (8-bit) and low-variance (4-bit)
    groups after PCA + Hadamard rotation.
    """

    def __init__(
        self,
        a_bits: int = 4,
        high_bits: int = 8,
        high_fraction: float = 0.125,
        a_sym: bool = False,
        w_bits: int = 4,
        w_sym: bool = True,
        clip_ratio: float = 1.0,
    ):
        super().__init__()
        self.a_bits = a_bits
        self.high_bits = high_bits
        self.high_fraction = high_fraction
        self.a_sym = a_sym
        self.w_bits = w_bits
        self.w_sym = w_sym
        self.clip_ratio = clip_ratio

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_name(cls) -> str:
        return "resq"

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ResQConfig":
        return cls(
            a_bits=config.get("a_bits", 4),
            high_bits=config.get("high_bits", 8),
            high_fraction=config.get("high_fraction", 0.125),
            a_sym=config.get("a_sym", False),
            w_bits=config.get("w_bits", 4),
            w_sym=config.get("w_sym", True),
            clip_ratio=config.get("clip_ratio", 1.0),
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            return ResQLinearMethod(self, prefix)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class ResQLinearMethod(LinearMethodBase):
    """Linear method for ResQ mixed-precision quantization.

    Stores two sets of quantized weights (main + high) and performs
    online activation quantization + shift-bias INT GEMM in the forward pass.
    """

    def __init__(self, quant_config: ResQConfig, prefix: str = ""):
        self.quant_config = quant_config
        self.prefix = prefix
        # Determine if this is a grouped layer (o_proj)
        self.is_grouped = "o_proj" in prefix

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create quantized weight parameters for ResQ."""
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        # Compute high/main dimensions
        high_bits_length = int(input_size_per_partition * self.quant_config.high_fraction)
        k_main = input_size_per_partition - high_bits_length

        # Main group weights: (N, K_main) int8 (centered integers)
        w_main = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, k_main, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w_main_int", w_main)

        # Main group scale: (N, 1) float16
        w_main_scale = ChannelQuantScaleParameter(
            data=torch.empty(output_size_per_partition, 1, dtype=torch.float16),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w_main_scale", w_main_scale)

        # Main group colsum: (N,) float32 - precomputed sum_k(w_int)
        w_main_colsum = ModelWeightParameter(
            data=torch.empty(output_size_per_partition, dtype=torch.float32),
            input_dim=None,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w_main_colsum", w_main_colsum)

        # High group weights: (N, K_high) int8
        if high_bits_length > 0:
            w_high = ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition, high_bits_length, dtype=torch.int8
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("w_high_int", w_high)

            # High group scale: (N, 1) float16
            w_high_scale = ChannelQuantScaleParameter(
                data=torch.empty(output_size_per_partition, 1, dtype=torch.float16),
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("w_high_scale", w_high_scale)

            # High group colsum: (N,) float32
            w_high_colsum = ModelWeightParameter(
                data=torch.empty(output_size_per_partition, dtype=torch.float32),
                input_dim=None,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("w_high_colsum", w_high_colsum)

        # Store metadata on the layer
        layer.high_bits_length = high_bits_length
        layer.k_main = k_main
        layer.is_grouped = self.is_grouped

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-process weights after loading: compute colsums if not provided."""
        # Ensure weights are on correct device and dtype
        layer.w_main_int = Parameter(layer.w_main_int.data, requires_grad=False)
        layer.w_main_scale = Parameter(layer.w_main_scale.data, requires_grad=False)

        # Compute colsum from weight integers if not loaded from checkpoint
        if hasattr(layer, "w_main_colsum"):
            # If colsum is all zeros (not loaded), compute it
            if layer.w_main_colsum.data.abs().sum() == 0:
                layer.w_main_colsum = Parameter(
                    layer.w_main_int.data.float().sum(dim=1),
                    requires_grad=False,
                )
            else:
                layer.w_main_colsum = Parameter(
                    layer.w_main_colsum.data, requires_grad=False
                )

        if hasattr(layer, "w_high_int"):
            layer.w_high_int = Parameter(layer.w_high_int.data, requires_grad=False)
            layer.w_high_scale = Parameter(layer.w_high_scale.data, requires_grad=False)
            if hasattr(layer, "w_high_colsum"):
                if layer.w_high_colsum.data.abs().sum() == 0:
                    layer.w_high_colsum = Parameter(
                        layer.w_high_int.data.float().sum(dim=1),
                        requires_grad=False,
                    )
                else:
                    layer.w_high_colsum = Parameter(
                        layer.w_high_colsum.data, requires_grad=False
                    )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply ResQ quantized linear: online activation quant + shift-bias GEMM."""
        # Get weight tensors
        w_main_int = layer.w_main_int
        w_main_scale = layer.w_main_scale
        w_high_int = getattr(layer, "w_high_int", None)
        w_high_scale = getattr(layer, "w_high_scale", None)
        high_bits_length = layer.high_bits_length

        # Determine groupsize for per-group layers
        grouped = layer.is_grouped
        groupsize = 0
        if grouped:
            # For o_proj: input is split into groups
            # groupsize is the full group size (main + high per group)
            K = x.shape[-1]
            ngroups = w_main_int.shape[1] // (layer.k_main // (K // layer.k_main))
            # Actually for grouped, we need the original groupsize
            # This will be set during weight loading
            groupsize = getattr(layer, "groupsize", 0)

        output = resq_linear_forward(
            x=x,
            w_main_int=w_main_int,
            w_main_scale=w_main_scale,
            w_high_int=w_high_int,
            w_high_scale=w_high_scale,
            a_bits=self.quant_config.a_bits,
            high_bits=self.quant_config.high_bits,
            high_bits_length=high_bits_length,
            a_sym=self.quant_config.a_sym,
            grouped=grouped,
            groupsize=groupsize,
            clip_ratio=self.quant_config.clip_ratio,
        )

        if bias is not None:
            output = output + bias

        return output
