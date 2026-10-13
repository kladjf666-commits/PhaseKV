"""压缩器公共算子: 缓存切片 (eviction) 与伪量化 (QA / CH-3 维护)。

这些算子直接作用于 transformers ``DynamicCache`` 的 ``key_cache`` / ``value_cache``,
保证与运行时 (runtime.py) 的真实绝对位置 (position_ids) 处理一致:
  - eviction 仅按时间顺序保留 token 子集 (升序), 保留的 K 已 baked-in 原始 RoPE,
    运行时对新 query 传入真实位置, 故相对距离正确。
  - 伪量化不改变张量形状与位置, 只降低数值精度并按位宽折算逻辑显存。
"""
from __future__ import annotations

from typing import List, Sequence, Union

import torch


def cache_num_layers(cache) -> int:
    return len(cache.key_cache)


def cache_seq_len(cache, layer_idx: int = 0) -> int:
    if len(cache.key_cache) <= layer_idx or cache.key_cache[layer_idx] is None:
        return 0
    return cache.key_cache[layer_idx].shape[-2]


def slice_cache(cache, keep_indices: Union[torch.Tensor, Sequence[torch.Tensor]]):
    """按 token 下标保留 KV (eviction)。

    Args:
        cache: transformers Cache。
        keep_indices: 统一的 1D LongTensor (所有层相同), 或长度为层数的列表 (逐层不同)。
            下标会被升序排序以保持时间顺序 (positional 一致性)。

    Returns:
        原 cache (就地修改 key_cache/value_cache)。
    """
    num_layers = cache_num_layers(cache)
    per_layer = isinstance(keep_indices, (list, tuple))
    for layer in range(num_layers):
        idx = keep_indices[layer] if per_layer else keep_indices
        if idx is None:
            continue
        device = cache.key_cache[layer].device
        idx = torch.as_tensor(idx, dtype=torch.long, device=device)
        idx, _ = torch.sort(idx)
        cache.key_cache[layer] = cache.key_cache[layer].index_select(-2, idx).contiguous()
        cache.value_cache[layer] = cache.value_cache[layer].index_select(-2, idx).contiguous()
    return cache


def keep_sink_recent_indices(seq_len: int, sink: int, recent: int,
                             device=None) -> torch.Tensor:
    """构造 [0..sink) ∪ [seq_len-recent..seq_len) 的保留下标 (去重升序)。"""
    sink = max(0, min(sink, seq_len))
    recent = max(0, min(recent, seq_len))
    head = torch.arange(0, sink, dtype=torch.long, device=device)
    tail_start = max(sink, seq_len - recent)
    tail = torch.arange(tail_start, seq_len, dtype=torch.long, device=device)
    return torch.unique(torch.cat([head, tail]))


def fake_quantize(tensor: torch.Tensor, bits: int, axis: int = -1) -> torch.Tensor:
    """非对称 min-max 伪量化: 沿 ``axis`` 分组量化到 ``bits`` 位再反量化。

    返回与输入同 dtype 的"量化-反量化"后张量, 用于评估量化对精度的影响。
    """
    if bits >= 16:
        return tensor
    qmax = (1 << bits) - 1
    t = tensor.float()
    t_min = t.amin(dim=axis, keepdim=True)
    t_max = t.amax(dim=axis, keepdim=True)
    scale = (t_max - t_min).clamp(min=1e-8) / qmax
    q = torch.round((t - t_min) / scale).clamp_(0, qmax)
    deq = q * scale + t_min
    return deq.to(tensor.dtype)


def quantized_cache_bytes(cache, bits: int, group_axis_size: int) -> int:
    """折算量化后的 KV Cache 逻辑字节数。

    每个元素占 ``bits`` 位; 每个量化组额外存 1 个 scale + 1 个 zero/min (各按 fp16=2B)。
    group_axis_size 为分组维度长度 (axis=-1 时即 head_dim)。
    """
    total_bits = 0
    overhead_bytes = 0
    for tensor in list(cache.key_cache) + list(cache.value_cache):
        if tensor is None:
            continue
        numel = tensor.numel()
        total_bits += numel * bits
        num_groups = numel // max(1, group_axis_size)
        overhead_bytes += num_groups * 2 * 2  # scale + min, fp16 each
    return total_bits // 8 + overhead_bytes
