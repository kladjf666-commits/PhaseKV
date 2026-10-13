"""PhaseKV Decode 层: FP16 热环 + 懒批冷压 (CH-2 架构)。

对应 idea 文档第 3.1-2/3 节。decode 阶段带宽受限 (洞见 1), 任何 per-step 压缩开销都在
和注意力读取抢带宽, 故:

  - 热层 (hot): 最近 ``recent_window`` 个生成 token 保持 FP16, 追加 O(1) (新 token
    重要度未知 → 不压)。由模型前向自然追加, 本模块不动它。
  - 冷层 (cooled): 滑出热窗口的较老生成 token, *每 G 步* 懒批一次性处理 (而非 per-step),
    用*廉价信号* (value-norm, 洞见 3: decode 取注意力分数昂贵) 逐 head 选择保留
    ``decode_keep_ratio`` 比例, 驱逐其余, 并对保留者做 ``cold_bits`` 位 KIVI 量化。
  - 永不触碰 Prompt 层 [0, K0): 冷压只在 decode 区间内进行。

懒批: 让滑出热窗口的 token 累积到一个批 (>= ``lazy_g``) 再统一冷却, 把冷压成本摊薄到
每 G 步一次, 避免每步都付选择/量化开销。冷层 token 一旦成型即量化冻结, 后续不再重量化
(避免反复 fake-quant 误差累积)。
"""
from __future__ import annotations

import torch

from . import hetero_cache as hc
from . import quant


def maintain(
    cache,
    *,
    step: int,
    model_config,
    state: hc.HeteroTierState,
    decode_keep_ratio: float,
    lazy_g: int,
) -> None:
    """每个 decode step 后维护三层布局; 满足懒批条件时冷却一批老生成 token (就地)。

    新 token 已由模型前向追加到物理末尾 (FP16 热层)。本函数只在"滑出热窗口的 token
    累积到 >= lazy_g"时触发一次批量冷却。
    """
    k0 = state.prompt_len_phys
    recent = state.recent_window
    seq = hc.cache_seq_len(cache)
    hot_len = seq - k0 - state.cold_len
    aged = hot_len - recent                      # 已滑出热窗口、待冷却的 token 数
    if aged < max(1, lazy_g):
        return                                   # 未攒够一个懒批, 暂不冷却

    base = k0 + state.cold_len                    # 冷候选批的物理起点
    batch = aged                                  # 本次冷却的批大小 (>= lazy_g)
    keep_b = batch if decode_keep_ratio >= 1.0 else max(1, int(round(decode_keep_ratio * batch)))

    if keep_b >= batch:
        # 无驱逐 (decode_keep_ratio>=1): 直接量化整批为冷层。
        _quantize_cold(cache, slice(base, base + batch), state)
        state.cold_len += batch
        return

    num_layers = hc.cache_num_layers(cache)
    keep_per_layer = []
    for layer in range(num_layers):
        v = cache.value_cache[layer]
        b, hkv = v.shape[0], v.shape[1]
        device = v.device
        signal = hc.value_norm_signal(v, slice(base, base + batch))   # [b, Hkv, batch]
        sel_local = hc.topk_per_head(signal, keep_b)                  # 批内相对下标
        sel = sel_local + base                                         # 还原到物理下标
        prompt_idx = _uniform_range(0, k0, b, hkv, device)
        cold_idx = _uniform_range(k0, base, b, hkv, device)            # 已成型冷层 [K0, base)
        hot_idx = _uniform_range(base + batch, seq, b, hkv, device)    # 热层尾 [base+batch, seq)
        keep = torch.cat([prompt_idx, cold_idx, sel, hot_idx], dim=-1)
        keep_per_layer.append(keep)

    hc.evict_per_head(cache, keep_per_layer)
    # 驱逐后保留批紧接已成型冷层之后: [base, base+keep_b) (base 不变, 因前缀未动)。
    _quantize_cold(cache, slice(base, base + keep_b), state)
    state.cold_len += keep_b


def _quantize_cold(cache, token_slice: slice, state: hc.HeteroTierState) -> None:
    """对冷层区间做 KIVI 式 ``cold_bits`` 位伪量化 (K per-channel / V per-token, 就地)。"""
    if state.cold_bits >= 16 or token_slice.stop <= token_slice.start:
        return
    for layer in range(hc.cache_num_layers(cache)):
        k = cache.key_cache[layer]
        v = cache.value_cache[layer]
        k[:, :, token_slice, :] = quant.quantize_key(
            k[:, :, token_slice, :], state.cold_bits, state.group_size)
        v[:, :, token_slice, :] = quant.quantize_value(
            v[:, :, token_slice, :], state.cold_bits, state.group_size)


def _uniform_range(lo: int, hi: int, b: int, hkv: int, device) -> torch.Tensor:
    """构造所有 head 相同的 [lo, hi) 保留下标, 形状 [b, Hkv, hi-lo]。"""
    idx = torch.arange(lo, hi, device=device, dtype=torch.long)
    return idx.view(1, 1, -1).expand(b, hkv, idx.numel())
