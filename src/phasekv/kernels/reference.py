"""PhaseKV 阶段 B: 三层混精度融合 decode 注意力的 golden reference (CH-2 架构)。

本模块用纯 PyTorch 定义单遍混精融合注意力的*正确语义*, 作为 ``attention_triton`` kernel
的数值基准 (kernel 必须逐值匹配本参考, 误差仅来自浮点次序)。它一次消费三层:

    [ packed 低比特 prompt 层 ] + [ packed 低比特冷层 ] + [ FP16 热层 ]

反量化后按时间升序拼成完整 K/V, 施加 GQA 展开, 做标准缩放点积注意力 (decode 单步 query
attends 全部已缓存过去 KV, 无因果掩码)。

注意: 本参考为求正确, 把三层*全部反量化回 fp16 再拼接*, 因此不省 HBM、不产出加速——加速
只来自 Triton kernel 的片上反量化 (compressed-domain execution)。本参考的价值是: (1) 锁定
阶段 B 的正确语义与数值目标; (2) 证明"packed 三层融合"与阶段 A fake-quant 逐值一致 (精度
不因真打包而变)。不做 dense 回退。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from .packing import PackedTensor, unpack_dequantize


@dataclass
class ThreeTierKV:
    """一层 transformer 的三层异构 KV (阶段 B 的融合 kernel 输入)。

    prompt/cold 以 ``PackedTensor`` 低比特驻留; hot 为 fp16。任一层可为 None (该层为空,
    如 decode 尚无冷层、或纯 prompt)。所有非空层形状须为 ``[b, Hkv, len, hd]``。
    """

    prompt_k: Optional[PackedTensor] = None
    prompt_v: Optional[PackedTensor] = None
    cold_k: Optional[PackedTensor] = None
    cold_v: Optional[PackedTensor] = None
    hot_k: Optional[torch.Tensor] = None
    hot_v: Optional[torch.Tensor] = None


def _dequant_concat(kv: ThreeTierKV) -> tuple:
    """把三层反量化并按 prompt->cold->hot 时间升序拼成 (K, V), 均 [b, Hkv, S, hd] fp16。"""
    ks: List[torch.Tensor] = []
    vs: List[torch.Tensor] = []
    if kv.prompt_k is not None:
        ks.append(unpack_dequantize(kv.prompt_k))
        vs.append(unpack_dequantize(kv.prompt_v))
    if kv.cold_k is not None:
        ks.append(unpack_dequantize(kv.cold_k))
        vs.append(unpack_dequantize(kv.cold_v))
    if kv.hot_k is not None:
        ks.append(kv.hot_k.to(torch.float16))
        vs.append(kv.hot_v.to(torch.float16))
    if not ks:
        raise ValueError("ThreeTierKV 三层皆空, 无可注意的 KV")
    return torch.cat(ks, dim=-2), torch.cat(vs, dim=-2)


def _expand_gqa(x: torch.Tensor, num_attention_heads: int) -> torch.Tensor:
    """把 [b, Hkv, S, hd] 沿 head 维展开成 [b, Hq, S, hd] (GQA 组内广播)。"""
    b, hkv, s, hd = x.shape
    if hkv == num_attention_heads:
        return x
    if num_attention_heads % hkv != 0:
        raise ValueError(f"注意力头数 {num_attention_heads} 不是 KV 头数 {hkv} 的整数倍")
    group = num_attention_heads // hkv
    return x.repeat_interleave(group, dim=1)


def fused_decode_attention_reference(
    q: torch.Tensor, kv: ThreeTierKV, num_attention_heads: int,
) -> torch.Tensor:
    """三层混精融合注意力的 golden reference。

    Args:
        q: 查询 ``[b, Hq, q_len, hd]`` (decode 单步 q_len=1); fp16。
        kv: 三层异构 KV (prompt/cold packed 低比特 + hot fp16)。
        num_attention_heads: 查询头数 Hq (用于 GQA 展开)。

    Returns:
        注意力输出 ``[b, Hq, q_len, hd]`` fp16。
    """
    if q.dim() != 4:
        raise ValueError(f"q 须为 [b, Hq, q_len, hd], 实际 {tuple(q.shape)}")
    head_dim = q.shape[-1]
    k_all, v_all = _dequant_concat(kv)          # [b, Hkv, S, hd]
    k_all = _expand_gqa(k_all, num_attention_heads)
    v_all = _expand_gqa(v_all, num_attention_heads)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q.float(), k_all.float().transpose(-1, -2)) * scale  # [b,Hq,q_len,S]
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_all.float())    # [b, Hq, q_len, hd]
    return out.to(torch.float16)


def dense_fp16_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_attention_heads: int,
) -> torch.Tensor:
    """未压缩 fp16 基准注意力 (供量化误差评估与 kernel 对拍); 语义同上, 无因果掩码。"""
    head_dim = q.shape[-1]
    k = _expand_gqa(k, num_attention_heads)
    v = _expand_gqa(v, num_attention_heads)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(torch.float16)
