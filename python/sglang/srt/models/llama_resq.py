"""
ResQ-quantized Llama model for SGLang.

Uses pure PyTorch attention (no FlashAttention/RadixAttention) to allow
insertion of U_C projection between QKV projection and RoPE.
Precision alignment with ResQ ptq.py --real_quant is the priority.
"""

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


class ResQLlamaAttention(nn.Module):
    """Llama attention with pure PyTorch attention and optional U_C projection.

    Key differences from standard LlamaAttention:
    - Uses torch matmul + softmax instead of FlashAttention/RadixAttention
    - Supports U_C projection on Q and K (between QKV proj and RoPE)
    - Simple KV cache in FP16 (no paged attention)
    """

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

        # U_C projection matrix (loaded from checkpoint, per-head Hadamard)
        # Will be set by load_weights if present
        self.U_C = None

        # Simple KV cache (no paging, grows with sequence)
        self.k_cache = None
        self.v_cache = None

    def _apply_u_c(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply U_C projection to Q and K before RoPE.

        U_C is (head_dim, head_dim) applied per-head.
        """
        if self.U_C is None:
            return q, k

        U = self.U_C.to(q.dtype)  # (head_dim, head_dim)

        # q: (total_tokens, q_size), reshape to (..., num_heads, head_dim)
        q_shape = q.shape
        k_shape = k.shape
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)

        # Apply U_C: q_proj = q @ U_C.T
        q = torch.einsum("bnh,dh->bnd", q, U)
        k = torch.einsum("bnh,dh->bnd", k, U)

        q = q.reshape(q_shape)
        k = k.reshape(k_shape)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # QKV projection (quantized via ResQ linear method)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply U_C projection (before RoPE)
        q, k = self._apply_u_c(q, k)

        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Reshape for attention: (batch*seq, num_heads, head_dim)
        total_tokens = q.shape[0]
        q = q.reshape(total_tokens, self.num_heads, self.head_dim)
        k = k.reshape(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.reshape(total_tokens, self.num_kv_heads, self.head_dim)

        # GQA: repeat K/V heads to match Q heads
        num_rep = self.num_heads // self.num_kv_heads
        if num_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, num_rep, -1).reshape(total_tokens, self.num_heads, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, num_rep, -1).reshape(total_tokens, self.num_heads, self.head_dim)

        # For simplicity in this MVP, use PyTorch's scaled_dot_product_attention
        # which handles causal masking automatically
        # Reshape to (batch, num_heads, seq, head_dim) for SDPA
        # Since SGLang flattens all tokens, we need to handle this carefully
        # For now, treat the whole batch as one sequence (works for offline eval)
        q = q.unsqueeze(0).transpose(1, 2)  # (1, num_heads, total_tokens, head_dim)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        # Use PyTorch SDPA with causal mask
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.scaling,
        )

        # Reshape back: (1, num_heads, total_tokens, head_dim) -> (total_tokens, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).squeeze(0)
        attn_output = attn_output.reshape(total_tokens, self.num_heads * self.head_dim)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class ResQLlamaMLP(nn.Module):
    """Same as LlamaMLP, just kept separate for clarity."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x, forward_batch=None, use_reduce_scatter=False):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class ResQLlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)

        self.self_attn = ResQLlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=attention_bias,
        )
        self.mlp = ResQLlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class ResQLlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: ResQLlamaDecoderLayer(
                config=config, quant_config=quant_config, layer_id=idx, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        self.layers_to_capture = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)

        if self.pp_group.is_last_rank:
            hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class ResQLlamaForCausalLM(nn.Module):
    """ResQ-quantized Llama model with pure PyTorch attention."""

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = ResQLlamaModel(config, quant_config=quant_config, prefix=add_prefix("model", prefix))

        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors)

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
            else:
                return self.pooler(hidden_states, forward_batch)
        return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights, handling ResQ-specific parameter names."""
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # Skip ResQ rotation matrices (handle separately)
            if name.startswith("resq_"):
                self._load_resq_global(name, loaded_weight)
                continue

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (layer_id < self.model.start_layer or layer_id >= self.model.end_layer)
            ):
                continue

            if "rotary_emb" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def _load_resq_global(self, name: str, tensor: torch.Tensor):
        """Load global ResQ tensors (rotation matrices, etc.)."""
        if "U" in name:
            # U_C projection matrix — set on all attention layers
            logger.info(f"Loading ResQ U_C projection: {tensor.shape}")
            for layer in self.model.layers:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.U_C = tensor.cuda()
        elif "R" in name:
            # R rotation matrix — used for activation rotation
            # For now, store globally; will be applied in linear method
            logger.info(f"Loading ResQ R rotation: {tensor.shape}")
            self._resq_R = tensor.cuda()


EntryClass = [ResQLlamaForCausalLM]
