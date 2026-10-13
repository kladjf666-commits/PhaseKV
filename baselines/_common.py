"""后端竞品基线的公共算子 (CH-5)。

服务于 baselines/ 下忠实复刻的竞品方法 (SnapKV / KIVI / SCOPE / PyramidKV), 与
eval/ 的 ``KVCompressor`` 契约、runtime 的"真实绝对位置 (true position)"约定保持一致。

为什么不复用 eval/compressors/common.slice_cache:
    该函数对同一层的所有 head 施加同一组保留下标 (index_select 作用于序列轴, 跨 head
    广播)。但 SnapKV / SCOPE / PyramidKV 的忠实实现是 *逐 head* 各自挑选要保留的 token,
    不同 head 保留的 token 下标不同。故本模块提供按 (batch, kv_head) 逐头 gather 的
    eviction: 每个 head 保留相同的 token *数量* (物理张量保持矩形), 但可保留不同的
    token *下标*; 保留下标按时间升序排列, 使被保留 K 的 baked-in RoPE 对随后新 query 的
    相对距离仍然精确 (新 query 由 runtime 以真实绝对位置施加 RoPE)。

GQA: 注意力按 query head 给出 [b, Hq, q, kv], KV cache 按 kv head 存储 [b, Hkv, seq, hd];
本模块把同一 kv group 内的 query head 重要度求和聚合到 kv head。

KIVI 分组量化: 沿指定轴以 group_size 为粒度做非对称 min-max 伪量化-反量化, 并提供与之
一致的逻辑字节折算 (K 沿 token 轴 per-channel; V 沿 head_dim 轴 per-token)。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


def cache_num_layers(cache) -> int:
    return len(cache.key_cache)


def cache_seq_len(cache, layer_idx: int = 0) -> int:
    if len(cache.key_cache) <= layer_idx or cache.key_cache[layer_idx] is None:
        return 0
    return cache.key_cache[layer_idx].shape[-2]


def num_kv_heads_of(model_config) -> int:
    return getattr(model_config, "num_key_value_heads",
                   getattr(model_config, "num_attention_heads"))


def aggregate_to_kv_heads(scores: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """把按 query head 的分数 [b, Hq, ...] 聚合到 kv head [b, Hkv, ...] (GQA 组内求和)。"""
    b, hq = scores.shape[0], scores.shape[1]
    if hq == num_kv_heads:
        return scores
    if hq % num_kv_heads != 0:
        raise ValueError(f"query head 数 {hq} 不是 kv head 数 {num_kv_heads} 的整数倍")
    group = hq // num_kv_heads
    tail = scores.shape[2:]
    return scores.view(b, num_kv_heads, group, *tail).sum(dim=2)


def observation_votes(attn_layer: torch.Tensor, num_kv_heads: int,
                      obs_window: int) -> torch.Tensor:
    """SnapKV 观察窗口投票: 用最近 ``obs_window`` 个 query 行对各 key 位置的注意力求和。

    Args:
        attn_layer: 单层注意力 [b, Hq, q_len, kv_len]。
        num_kv_heads: KV head 数 (GQA 聚合目标)。
        obs_window: 观察窗口长度 (取 query 序列末尾 W 行)。

    Returns:
        [b, num_kv_heads, kv_len] 的每 key 位置投票分数。
    """
    q_len = attn_layer.shape[-2]
    w = max(1, min(obs_window, q_len))
    votes = attn_layer[:, :, -w:, :].float().sum(dim=2)  # [b, Hq, kv_len]
    return aggregate_to_kv_heads(votes, num_kv_heads)


def pool_scores(scores: torch.Tensor, kernel: Optional[int],
                mode: str = "maxpool") -> torch.Tensor:
    """对重要度分数沿序列轴做 1D 池化 (SnapKV 的聚类: 让被选 token 形成连续簇)。

    Args:
        scores: [b, heads, length]。
        kernel: 池化核大小; None 或 <=1 表示不池化。
        mode: ``maxpool`` 或 ``avgpool``。
    """
    if kernel is None or kernel <= 1:
        return scores
    b, h, length = scores.shape
    x = scores.reshape(b * h, 1, length)
    pad = kernel // 2
    if mode == "maxpool":
        pooled = F.max_pool1d(x, kernel_size=kernel, stride=1, padding=pad)
    elif mode == "avgpool":
        pooled = F.avg_pool1d(x, kernel_size=kernel, stride=1, padding=pad)
    else:
        raise ValueError(f"未知 pooling: {mode}, 仅支持 maxpool/avgpool")
    pooled = pooled[..., :length]
    return pooled.reshape(b, h, length)


def evict_per_head(cache, keep_indices: Sequence[Optional[torch.Tensor]]):
    """按逐头下标保留 KV (eviction)。

    Args:
        cache: transformers Cache。
        keep_indices: 长度为层数的列表; 第 l 项为 ``[b, num_kv_heads, budget_l]`` 的
            LongTensor (同层内各 head 的 budget 必须一致以保持矩形; 层间可不同)。
            为 None 的层不改动。每个 head 的下标会被升序排序以保持时间顺序。

    Returns:
        原 cache (就地修改)。
    """
    num_layers = cache_num_layers(cache)
    for layer in range(num_layers):
        idx = keep_indices[layer]
        if idx is None:
            continue
        k = cache.key_cache[layer]
        v = cache.value_cache[layer]
        idx = idx.to(device=k.device, dtype=torch.long)
        idx, _ = torch.sort(idx, dim=-1)
        head_dim = k.shape[-1]
        gather_idx = idx.unsqueeze(-1).expand(idx.shape[0], idx.shape[1], idx.shape[2], head_dim)
        cache.key_cache[layer] = k.gather(-2, gather_idx).contiguous()
        cache.value_cache[layer] = v.gather(-2, gather_idx).contiguous()
    return cache


def topk_keep_per_head(pooled_prefix: torch.Tensor, select_k: int,
                       prefix_len: int, recent_lo: int, recent_hi: int) -> torch.Tensor:
    """构造"候选区 top-k ∪ 最近窗口"的逐头保留下标。

    Args:
        pooled_prefix: [b, Hkv, prefix_len] 候选区 (观察窗口之外) 的池化重要度。
        select_k: 从候选区每头选取的 token 数。
        prefix_len: 候选区长度。
        recent_lo, recent_hi: 始终保留的最近窗口 [recent_lo, recent_hi) (所有 head 相同)。

    Returns:
        [b, Hkv, select_k + (recent_hi - recent_lo)] 的保留下标 (未排序, evict 时再排)。
    """
    b, hkv, _ = pooled_prefix.shape
    device = pooled_prefix.device
    if select_k > 0:
        k = min(select_k, prefix_len)
        top = torch.topk(pooled_prefix, k=k, dim=-1).indices.long()  # [b, Hkv, k]
    else:
        top = torch.zeros((b, hkv, 0), dtype=torch.long, device=device)
    recent = torch.arange(recent_lo, recent_hi, device=device, dtype=torch.long)
    recent = recent.view(1, 1, -1).expand(b, hkv, recent.numel())
    return torch.cat([top, recent], dim=-1)


# --------------------------------------------------------------------------- #
# KIVI 分组伪量化 + 逻辑字节折算
# --------------------------------------------------------------------------- #

#: 每个量化组的元数据开销 (scale + zero, 各按 fp16 = 2 字节)。与 src/phasekv/quant.py 一致。
SCALE_ZERO_BYTES: int = 4


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def grouped_fake_quant(x: torch.Tensor, bits: int, axis: int,
                       group_size: int) -> torch.Tensor:
    """沿 ``axis`` 以 ``group_size`` 为粒度做非对称 min-max 伪量化-反量化。

    返回与 ``x`` 同形状、同 dtype 的低比特栅格张量, 供模型据此做真实前向以测量量化对
    精度的真实影响 (非模拟)。bits>=16 时原样返回。
    """
    if bits is None or bits >= 16:
        return x
    if bits < 1:
        raise ValueError(f"bits 必须 >=1, 实际 {bits}")
    qmax = (1 << bits) - 1
    orig_dtype = x.dtype
    axis = axis % x.dim()
    xf = x.float().movedim(axis, -1).contiguous()
    moved_shape = xf.shape
    length = moved_shape[-1]
    gs = length if (group_size is None or group_size <= 0) else min(group_size, length)
    pad = (-length) % gs
    if pad:
        tail = xf[..., -1:].expand(*xf.shape[:-1], pad)
        xf = torch.cat([xf, tail], dim=-1)
    padded_len = xf.shape[-1]
    n_groups = padded_len // gs
    xg = xf.reshape(*xf.shape[:-1], n_groups, gs)
    g_min = xg.amin(dim=-1, keepdim=True)
    g_max = xg.amax(dim=-1, keepdim=True)
    scale = (g_max - g_min).clamp_(min=1e-8) / qmax
    q = torch.round((xg - g_min) / scale).clamp_(0, qmax)
    deq = (q * scale + g_min).reshape(moved_shape[:-1] + (padded_len,))
    if pad:
        deq = deq[..., :length]
    return deq.movedim(-1, axis).contiguous().to(orig_dtype)


def quantize_token_slice(cache, token_slice: slice, k_bits: int, v_bits: int,
                         group_size: int):
    """对 cache 所有层在 ``token_slice`` 区间做 KIVI 式量化 (就地)。

    K: 沿 token 轴 (-2) per-channel 分组; V: 沿 head_dim 轴 (-1) per-token 分组。
    """
    for layer in range(cache_num_layers(cache)):
        k = cache.key_cache[layer]
        v = cache.value_cache[layer]
        k[:, :, token_slice, :] = grouped_fake_quant(
            k[:, :, token_slice, :], k_bits, axis=-2, group_size=group_size)
        v[:, :, token_slice, :] = grouped_fake_quant(
            v[:, :, token_slice, :], v_bits, axis=-1, group_size=group_size)
    return cache


def kivi_logical_bytes(cache, quantized_len: int, k_bits: int, v_bits: int,
                       group_size: int) -> int:
    """折算 KIVI 混精 KV Cache 的逻辑字节数。

    前 ``quantized_len`` 个 token 已量化 (K per-channel / V per-token, 含分组元数据开销);
    其余为最近 FP16 残差 (每元素 2 字节, 无元数据)。
    """
    total = 0
    for layer in range(cache_num_layers(cache)):
        k = cache.key_cache[layer]
        if k is None:
            continue
        b, h, seq, head_dim = k.shape
        ql = max(0, min(quantized_len, seq))
        res = seq - ql
        fan = b * h
        if ql > 0:
            k_data_bits = fan * ql * head_dim * k_bits
            v_data_bits = fan * ql * head_dim * v_bits
            gs_tok = ql if group_size <= 0 else min(group_size, ql)
            gs_dim = head_dim if group_size <= 0 else min(group_size, head_dim)
            k_groups = fan * head_dim * _ceil_div(ql, gs_tok)
            v_groups = fan * ql * _ceil_div(head_dim, gs_dim)
            overhead = 0
            if k_bits < 16:
                overhead += k_groups * SCALE_ZERO_BYTES
            if v_bits < 16:
                overhead += v_groups * SCALE_ZERO_BYTES
            total += (k_data_bits + v_data_bits) // 8 + overhead
        if res > 0:
            total += 2 * fan * res * head_dim * 2  # K、V 各 fp16=2B
    return total
