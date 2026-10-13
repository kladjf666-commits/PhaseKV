"""PhaseKV 量化算子: KIVI 式分组非对称伪量化 + 逻辑字节折算 (CH-2 架构)。

KIVI (Liu et al., ICML'24) 的关键观察: Key cache 沿 channel 维度存在系统性离群值,
宜按 *channel* 量化; Value cache 无显著 channel 离群, 宜按 *token* 量化。本模块据此:

  - K (per-channel): 沿 token(序列)轴分组, 每个 (head, channel) 在其 token 组内独立
    取 min/max。  => 同一 channel 的所有 token 共享一套量化参数, 抵消 channel 离群。
  - V (per-token):  沿 head_dim 轴分组, 每个 (head, token) 在其 head_dim 组内独立
    取 min/max。  => 每个 token 的 value 向量独立量化。

分组大小 ``group_size`` 控制粒度 (KIVI 默认 32 token / 32 channel); 每个量化组额外存
1 个 scale + 1 个 zero-point(此处用分组最小值 min 作为非对称零点), 默认各按 fp16=2B。

阶段 A 采用"量化-反量化"(fake quantization): 返回落在低比特栅格上的同 dtype 张量,
模型据此做真实前向, 从而得到低比特量化对精度的*真实*影响(非模拟)。逻辑显存占用按
目标位宽 + 分组开销解析折算 (物理张量仍是 fp16, 逻辑占用反映压缩后真实位宽)。

本模块不做任何精度回退 (no dense fallback): 量化即量化, 误差由相位预算分配控制。
"""
from __future__ import annotations

from typing import Tuple

import torch

#: 每个量化组的元数据开销 (scale + zero, 各按 fp16=2 字节)。
SCALE_ZERO_BYTES: int = 4


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _grouped_affine_quant_dequant(
    x: torch.Tensor, bits: int, axis: int, group_size: int
) -> torch.Tensor:
    """沿 ``axis`` 以 ``group_size`` 为粒度做非对称 min-max 伪量化并反量化。

    Args:
        x: 任意形状张量。
        bits: 目标位宽 (<16 时量化; >=16 原样返回)。
        axis: 量化分组所沿的维度 (该维度被切成若干 group)。
        group_size: 每组元素数; <=0 视为整个 axis 一组。

    Returns:
        与 ``x`` 同形状、同 dtype 的"量化-反量化"结果。
    """
    if bits >= 16:
        return x
    if bits < 1:
        raise ValueError(f"bits 必须 >=1, 实际 {bits}")
    qmax = (1 << bits) - 1

    orig_dtype = x.dtype
    axis = axis % x.dim()
    xf = x.float().movedim(axis, -1).contiguous()
    moved_shape = xf.shape
    length = moved_shape[-1]
    gs = length if group_size is None or group_size <= 0 else min(group_size, length)

    pad = (-length) % gs
    if pad:
        # 用该行最后一个真实值复制填充: 复制已有值不改变任何 group 的 min/max。
        tail = xf[..., -1:].expand(*xf.shape[:-1], pad)
        xf = torch.cat([xf, tail], dim=-1)
    padded_len = xf.shape[-1]
    n_groups = padded_len // gs

    xg = xf.reshape(*xf.shape[:-1], n_groups, gs)
    g_min = xg.amin(dim=-1, keepdim=True)
    g_max = xg.amax(dim=-1, keepdim=True)
    scale = (g_max - g_min).clamp_(min=1e-8) / qmax
    q = torch.round((xg - g_min) / scale).clamp_(0, qmax)
    deq = q * scale + g_min
    deq = deq.reshape(moved_shape[:-1] + (padded_len,))
    if pad:
        deq = deq[..., :length]
    deq = deq.movedim(-1, axis).contiguous()
    return deq.to(orig_dtype)


def quantize_key(key: torch.Tensor, bits: int, group_size: int = 32) -> torch.Tensor:
    """K 的 KIVI per-channel 伪量化。

    ``key`` 形状 ``[batch, num_kv_heads, seq, head_dim]``; 沿 token 轴 (-2) 分组,
    每个 channel(head_dim) 在其 token 组内独立量化。
    """
    if key.dim() < 2:
        raise ValueError("key 维度过低, 期望至少 [..., seq, head_dim]")
    return _grouped_affine_quant_dequant(key, bits, axis=-2, group_size=group_size)


def quantize_value(value: torch.Tensor, bits: int, group_size: int = 32) -> torch.Tensor:
    """V 的 KIVI per-token 伪量化。

    ``value`` 形状 ``[batch, num_kv_heads, seq, head_dim]``; 沿 head_dim 轴 (-1) 分组,
    每个 token 在其 head_dim 组内独立量化。
    """
    if value.dim() < 1:
        raise ValueError("value 维度过低")
    return _grouped_affine_quant_dequant(value, bits, axis=-1, group_size=group_size)


def quantize_kv(
    key: torch.Tensor, value: torch.Tensor, k_bits: int, v_bits: int,
    group_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对一段 K/V 同时做 KIVI 式伪量化, 返回反量化后的 (key, value)。"""
    return (quantize_key(key, k_bits, group_size),
            quantize_value(value, v_bits, group_size))


def kv_group_counts(num_tokens: int, head_dim: int, group_size: int) -> Tuple[int, int]:
    """单个 (layer, head) 下 K / V 的量化组数。

    K 沿 token 分组、每 channel 独立: head_dim * ceil(num_tokens / gs_token)。
    V 沿 head_dim 分组、每 token 独立: num_tokens * ceil(head_dim / gs_dim)。
    其中 gs_* 为 group_size 受对应轴长度的截断 (group_size<=0 表示整轴一组)。
    """
    if num_tokens <= 0:
        return 0, 0
    gs_token = num_tokens if group_size is None or group_size <= 0 else min(group_size, num_tokens)
    gs_dim = head_dim if group_size is None or group_size <= 0 else min(group_size, head_dim)
    k_groups = head_dim * _ceil_div(num_tokens, gs_token)
    v_groups = num_tokens * _ceil_div(head_dim, gs_dim)
    return k_groups, v_groups


def quantized_tier_bytes(
    num_tokens: int, num_layers: int, num_kv_heads: int, head_dim: int,
    k_bits: int, v_bits: int, group_size: int = 32, batch: int = 1,
) -> int:
    """折算一个量化层 (prompt 层或冷层) 的逻辑字节数。

    数据位 = batch * layers * heads * tokens * head_dim * bits (K、V 各算);
    元数据 = (K 组数 + V 组数) * SCALE_ZERO_BYTES。位宽 >=16 时无元数据开销。
    """
    if num_tokens <= 0:
        return 0
    elems_per_lh = num_tokens * head_dim
    fan = batch * num_layers * num_kv_heads
    k_data_bits = fan * elems_per_lh * k_bits
    v_data_bits = fan * elems_per_lh * v_bits
    overhead = 0
    k_groups, v_groups = kv_group_counts(num_tokens, head_dim, group_size)
    if k_bits < 16:
        overhead += fan * k_groups * SCALE_ZERO_BYTES
    if v_bits < 16:
        overhead += fan * v_groups * SCALE_ZERO_BYTES
    return (k_data_bits + v_data_bits) // 8 + overhead


def fp16_tier_bytes(
    num_tokens: int, num_layers: int, num_kv_heads: int, head_dim: int, batch: int = 1,
) -> int:
    """折算一个 fp16 层 (decode 热环) 的逻辑字节数 (K、V 各 2 字节, 无量化开销)。"""
    if num_tokens <= 0:
        return 0
    return 2 * batch * num_layers * num_kv_heads * num_tokens * head_dim * 2
