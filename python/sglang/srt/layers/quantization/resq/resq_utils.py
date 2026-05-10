"""
ResQ quantization utilities.

Numerically identical to project-resq/fake_quant/utils/quant_utils.py
forward_real_quant() logic.
"""

import torch
from typing import Tuple, Optional


def find_params_asymmetric(
    x: torch.Tensor, maxq: int, clip_ratio: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token asymmetric quantization parameters.

    Args:
        x: (..., K) activation tensor. Scale/zero computed per-row (last dim).
        maxq: Maximum quantized value (e.g., 15 for 4-bit, 255 for 8-bit).
        clip_ratio: Clipping ratio for min/max (default 1.0 = no clipping).

    Returns:
        scale: (..., 1) per-token scale.
        zero: (..., 1) per-token zero point.
    """
    init_shape = x.shape
    reshaped_x = x.reshape(-1, x.shape[-1])

    tmp = torch.zeros(reshaped_x.shape[0], device=x.device)
    xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * clip_ratio
    xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * clip_ratio

    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)

    # Broadcast to match input shape: (..., 1)
    scale = scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
    zero = zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

    return scale, zero


def find_params_symmetric(
    x: torch.Tensor, maxq: int, clip_ratio: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token symmetric quantization parameters.

    Args:
        x: (..., K) activation tensor.
        maxq: Maximum quantized value (e.g., 7 for 4-bit, 127 for 8-bit).
        clip_ratio: Clipping ratio.

    Returns:
        scale: (..., 1) per-token scale.
        zero: (..., 1) zeros (always 0 for symmetric).
    """
    init_shape = x.shape
    reshaped_x = x.reshape(-1, x.shape[-1])

    tmp = torch.zeros(reshaped_x.shape[0], device=x.device)
    xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * clip_ratio
    xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * clip_ratio
    xmax = torch.maximum(torch.abs(xmin), xmax)

    tmp = xmax == 0
    scale = xmax / maxq
    scale[tmp] = 1

    scale = scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
    zero = torch.zeros_like(scale)

    return scale, zero


def find_params_per_token_groupwise(
    x: torch.Tensor, maxq: int, sym: bool, clip_ratio: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token-per-group quantization parameters (for grouped layers).

    Args:
        x: (batch, seq, ngroups, group_k) tensor.
        maxq: Maximum quantized value.
        sym: Whether symmetric quantization.
        clip_ratio: Clipping ratio.

    Returns:
        scale: (batch, seq, ngroups, 1) per-token-per-group scale.
        zero: (batch, seq, ngroups, 1) per-token-per-group zero point.
    """
    xmax = torch.amax(x, dim=3, keepdim=True) * clip_ratio
    xmin = torch.amin(x, dim=3, keepdim=True) * clip_ratio
    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmax == 0
        scale = xmax / maxq
        scale[tmp] = 1
        zero = torch.zeros_like(scale)
    else:
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / scale)

    return scale, zero


def quantize_activation(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: int,
    sym: bool,
) -> torch.Tensor:
    """Quantize activation to integer range.

    Args:
        x: Input activation tensor.
        scale: Quantization scale (same shape as x or broadcastable).
        zero: Zero point (same shape as x or broadcastable).
        maxq: Maximum quantized value.
        sym: Whether symmetric quantization.

    Returns:
        q_int: Quantized integer tensor (same shape as x).
            Asymmetric: unsigned [0, maxq].
            Symmetric: signed [-(maxq+1), maxq].
    """
    if sym:
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    else:
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q


def resq_gemm_per_token(
    q_act: torch.Tensor,
    s_x: torch.Tensor,
    z_x: torch.Tensor,
    w_int: torch.Tensor,
    s_w: torch.Tensor,
    shift: float,
    sym: bool,
) -> torch.Tensor:
    """Per-token shift+bias INT GEMM formula.

    Computes: output = s_x * s_w * (q_shifted @ w_int.T + bias)

    Args:
        q_act: (M, K) quantized activation (unsigned for asym, signed for sym).
        s_x: (M, 1) per-token activation scale.
        z_x: (M, 1) per-token zero point (unused for sym).
        w_int: (N, K) centered weight integers.
        s_w: (N,) or (N, 1) per-channel weight scale.
        shift: Shift value ((maxq+1)/2 for asym, 0 for sym).
        sym: Whether symmetric quantization.

    Returns:
        output: (M, N) float32 result.
    """
    q_act_f = q_act.float()
    s_x_f = s_x[..., :1].reshape(-1, 1).float()
    s_w_f = s_w.flatten().unsqueeze(0).float()  # (1, N)
    w_int_f = w_int.float()

    if sym:
        # Symmetric: q_act is already centered, no shift/bias needed
        output = s_x_f * s_w_f * (q_act_f @ w_int_f.T)
    else:
        z_x_f = z_x[..., :1].reshape(-1, 1).float()
        q_shifted = q_act_f - shift
        w_colsum = w_int_f.sum(dim=1, keepdim=True).T  # (1, N)
        bias = (shift - z_x_f) @ w_colsum  # (M, N)
        output = s_x_f * s_w_f * (q_shifted @ w_int_f.T + bias)

    return output


def resq_gemm_per_group(
    q_act: torch.Tensor,
    s_x: torch.Tensor,
    z_x: Optional[torch.Tensor],
    w_int: torch.Tensor,
    s_w: torch.Tensor,
    shift: float,
    sym: bool,
    ngroups: int,
) -> torch.Tensor:
    """Per-group shift+bias INT GEMM formula (for o_proj).

    Args:
        q_act: (M, ngroups, group_k) quantized activation.
        s_x: (M, ngroups, 1) per-token-per-group scale.
        z_x: (M, ngroups, 1) per-token-per-group zero (None for sym).
        w_int: (N, ngroups * group_k) centered weight integers.
        s_w: (N, 1) or (N,) per-channel weight scale.
        shift: Shift value.
        sym: Whether symmetric.
        ngroups: Number of groups.

    Returns:
        output: (M, N) float32 result.
    """
    M = q_act.shape[0]
    group_k = q_act.shape[-1]
    N = w_int.shape[0]
    s_w_f = s_w.flatten().unsqueeze(0).float()  # (1, N)
    w_int_f = w_int.float()

    output = torch.zeros(M, N, device=q_act.device, dtype=torch.float32)

    for g in range(ngroups):
        q_x_g = q_act[:, g, :].float()  # (M, group_k)
        s_x_g = s_x[:, g, :].float()  # (M, 1)
        q_w_g = w_int_f[:, g * group_k:(g + 1) * group_k]  # (N, group_k)

        if sym:
            output += s_x_g * s_w_f * (q_x_g @ q_w_g.T)
        else:
            z_x_g = z_x[:, g, :].float()  # (M, 1)
            q_shifted = q_x_g - shift
            w_colsum = q_w_g.sum(dim=1, keepdim=True).T  # (1, N)
            bias = (shift - z_x_g) @ w_colsum  # (M, N)
            output += s_x_g * s_w_f * (q_shifted @ q_w_g.T + bias)

    return output


def resq_linear_forward(
    x: torch.Tensor,
    w_main_int: torch.Tensor,
    w_main_scale: torch.Tensor,
    w_high_int: Optional[torch.Tensor],
    w_high_scale: Optional[torch.Tensor],
    a_bits: int,
    high_bits: int,
    high_bits_length: int,
    a_sym: bool,
    grouped: bool = False,
    groupsize: int = 0,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Complete ResQ quantized linear forward pass.

    Numerically identical to ActQuantWrapper.forward_real_quant() per-token path.

    Args:
        x: (batch, seq, K) input activation in FP16/BF16.
        w_main_int: (N, K_main) centered weight integers for main (4-bit) group.
        w_main_scale: (N, 1) per-channel weight scale for main group.
        w_high_int: (N, K_high) centered weight integers for high (8-bit) group.
        w_high_scale: (N, 1) per-channel weight scale for high group.
        a_bits: Activation bits for main group (e.g., 4).
        high_bits: Activation bits for high group (e.g., 8).
        high_bits_length: Number of channels in high group per token.
        a_sym: Whether activation quantization is symmetric.
        grouped: Whether to use per-group quantization (for o_proj).
        groupsize: Group size for per-group quantization.
        clip_ratio: Clipping ratio for quantization params.

    Returns:
        output: (batch, seq, N) FP16 result.
    """
    x_dtype = x.dtype
    batch_shape = x.shape[:-1]
    K = x.shape[-1]
    N = w_main_int.shape[0]

    # Quantization ranges
    if a_sym:
        maxq_main = 2 ** (a_bits - 1) - 1
        maxq_high = 2 ** (high_bits - 1) - 1
        shift_main = 0.0
        shift_high = 0.0
    else:
        maxq_main = 2 ** a_bits - 1
        maxq_high = 2 ** high_bits - 1
        shift_main = (maxq_main + 1) / 2.0
        shift_high = (maxq_high + 1) / 2.0

    if not grouped:
        # === Per-token path ===
        # Split activation: [..., :K_main] = main, [..., K_main:] = high
        K_main = K - high_bits_length
        x_main = x[..., :K_main]
        x_high = x[..., K_main:] if high_bits_length > 0 else None

        # Compute quantization params and quantize
        if a_sym:
            s_x_m, z_x_m = find_params_symmetric(x_main, maxq_main, clip_ratio)
        else:
            s_x_m, z_x_m = find_params_asymmetric(x_main, maxq_main, clip_ratio)
        q_m = quantize_activation(x_main, s_x_m, z_x_m, maxq_main, a_sym)

        # Flatten for matmul
        q_m_flat = q_m.reshape(-1, q_m.shape[-1])
        s_x_m_flat = s_x_m.reshape(-1, s_x_m.shape[-1])
        z_x_m_flat = z_x_m.reshape(-1, z_x_m.shape[-1])

        # Main GEMM
        y = resq_gemm_per_token(
            q_m_flat, s_x_m_flat, z_x_m_flat,
            w_main_int, w_main_scale, shift_main, a_sym,
        )

        # High GEMM
        if x_high is not None and w_high_int is not None:
            if a_sym:
                s_x_h, z_x_h = find_params_symmetric(x_high, maxq_high, clip_ratio)
            else:
                s_x_h, z_x_h = find_params_asymmetric(x_high, maxq_high, clip_ratio)
            q_h = quantize_activation(x_high, s_x_h, z_x_h, maxq_high, a_sym)

            q_h_flat = q_h.reshape(-1, q_h.shape[-1])
            s_x_h_flat = s_x_h.reshape(-1, s_x_h.shape[-1])
            z_x_h_flat = z_x_h.reshape(-1, z_x_h.shape[-1])

            y += resq_gemm_per_token(
                q_h_flat, s_x_h_flat, z_x_h_flat,
                w_high_int, w_high_scale, shift_high, a_sym,
            )

        return y.half().reshape(*batch_shape, N).to(x_dtype)

    else:
        # === Per-group path (o_proj) ===
        M = x.reshape(-1, K).shape[0]
        ngroups = K // groupsize

        # Reshape to (batch, seq, ngroups, groupsize)
        x_grouped = x.reshape(x.shape[0], x.shape[1], ngroups, groupsize)

        # Split within each group
        group_k_main = groupsize - high_bits_length
        x_main = x_grouped[..., :group_k_main]
        x_high = x_grouped[..., group_k_main:] if high_bits_length > 0 else None

        # Compute per-group quantization params
        if a_sym:
            s_x_m, z_x_m = find_params_per_token_groupwise(x_main, maxq_main, sym=True, clip_ratio=clip_ratio)
        else:
            s_x_m, z_x_m = find_params_per_token_groupwise(x_main, maxq_main, sym=False, clip_ratio=clip_ratio)
        q_m = quantize_activation(x_main, s_x_m, z_x_m, maxq_main, a_sym)

        # Flatten batch dims
        q_m_flat = q_m.reshape(M, ngroups, group_k_main)
        s_x_m_flat = s_x_m.reshape(M, ngroups, -1)
        z_x_m_flat = z_x_m.reshape(M, ngroups, -1) if not a_sym else None

        # Main GEMM per-group
        y = resq_gemm_per_group(
            q_m_flat, s_x_m_flat, z_x_m_flat,
            w_main_int, w_main_scale, shift_main, a_sym, ngroups,
        )

        # High GEMM per-group
        if x_high is not None and w_high_int is not None:
            group_k_high = high_bits_length
            if a_sym:
                s_x_h, z_x_h = find_params_per_token_groupwise(x_high, maxq_high, sym=True, clip_ratio=clip_ratio)
            else:
                s_x_h, z_x_h = find_params_per_token_groupwise(x_high, maxq_high, sym=False, clip_ratio=clip_ratio)
            q_h = quantize_activation(x_high, s_x_h, z_x_h, maxq_high, a_sym)

            q_h_flat = q_h.reshape(M, ngroups, group_k_high)
            s_x_h_flat = s_x_h.reshape(M, ngroups, -1)
            z_x_h_flat = z_x_h.reshape(M, ngroups, -1) if not a_sym else None

            y += resq_gemm_per_group(
                q_h_flat, s_x_h_flat, z_x_h_flat,
                w_high_int, w_high_scale, shift_high, a_sym, ngroups,
            )

        return y.half().reshape(*batch_shape, N).to(x_dtype)
