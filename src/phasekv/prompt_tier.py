"""PhaseKV Prompt 层 (prefill 一次成型, 只读) (CH-2 架构)。

对应 idea 文档第 3.1-1 节。prefill 结束时对 prompt 的 KV 做一次性结构化压缩:

  1. 选择 (洞见 3): prefill 本就在算全量注意力, 拿注意力分数做选择*零额外成本*。
     用 SnapKV 式观察窗口投票 + per-head 聚类 pooling, 每个 kv head 各自保留
     ``sink + 选中 prompt token`` 共 K0 个 (各 head 下标不同, 数量相同, 张量保持矩形)。
  2. 打包 (洞见 2): prompt-KV 读放大 = 解码长度 T、只写一次且写在带宽富余的 prefill,
     故给它*最狠*的低比特预算。对保留的 K/V 做 KIVI 式 per-channel(K)/per-token(V)
     ``prompt_bits`` 位伪量化。
  3. 冻结: 成型后不再改写 (无 per-step 重量化开销), 是读放大 = T 的对象, 值得最狠压。

选择需要注意力分数 (requires_attention=True, runtime 以 eager 提供)。保留下标按时间
升序, baked-in RoPE 对随后新 query 相对距离精确。
"""
from __future__ import annotations

import torch

from . import hetero_cache as hc
from . import quant


def select_prompt_indices(
    attentions,
    *,
    prompt_len: int,
    model_config,
    prefill_keep_ratio: float,
    sink_tokens: int,
    obs_window: int,
    pool_kernel: int,
    pooling: str,
) -> list[torch.Tensor]:
    """仅返回逐层、逐 KV head 的原始位置，不改写或量化 prefill 缓存。

    每层下标形状为 [B,Hkv,K0]，严格升序且没有重复。短 Prompt 时在
    K0 预算内优先保留 sink，再保留尽可能多的最近 token。
    """
    if attentions is None or not attentions:
        raise ValueError("PhaseKV prompt 选择需要每层 prefill attentions")
    if prompt_len <= 0 or not 0.0 < prefill_keep_ratio <= 1.0:
        raise ValueError("prompt_len 和 prefill_keep_ratio 无效")
    if sink_tokens < 0 or obs_window <= 0:
        raise ValueError("sink_tokens 不能为负，obs_window 必须为正")

    hkv = hc.num_kv_heads_of(model_config)
    sink = min(sink_tokens, prompt_len)
    budget = min(prompt_len, max(1, sink, round(prompt_len * prefill_keep_ratio)))
    sink_keep = min(sink, budget)
    recent_keep = min(obs_window, budget - sink_keep, prompt_len - sink_keep)
    recent_start = prompt_len - recent_keep
    select_k = budget - sink_keep - recent_keep
    result: list[torch.Tensor] = []

    for attention in attentions:
        if attention is None or attention.shape[-1] != prompt_len:
            raise ValueError("prefill attention 缺失或 key 长度与 prompt_len 不一致")
        b = attention.shape[0]
        device = attention.device
        sink_idx = torch.arange(sink_keep, device=device).view(1, 1, -1).expand(b, hkv, -1)
        recent_idx = torch.arange(recent_start, prompt_len, device=device)
        recent_idx = recent_idx.view(1, 1, -1).expand(b, hkv, -1)
        if select_k:
            votes = hc.observation_votes(attention, hkv, min(obs_window, prompt_len))
            candidates = hc.pool_scores(
                votes[..., sink_keep:recent_start], pool_kernel, pooling)
            top = hc.topk_per_head(candidates, select_k) + sink_keep
        else:
            top = sink_idx[..., :0]
        keep = torch.cat((sink_idx, top, recent_idx), dim=-1)
        if keep.shape[-1] != budget:
            raise RuntimeError("Prompt 保留数量与预算不一致")
        result.append(keep.sort(dim=-1).values)
    return result


def select_and_pack(
    cache,
    *,
    attentions,
    prompt_len: int,
    model_config,
    state: hc.HeteroTierState,
    prefill_keep_ratio: float,
    sink_tokens: int,
    obs_window: int,
    pool_kernel: int,
    pooling: str,
) -> None:
    """对 prefill 后的 prompt KV 做 per-head 注意力选择 + KIVI 低比特打包 (就地)。

    更新 ``state.prompt_len_phys`` 为成型后的 prompt 层物理长度 K0。
    """
    keep_per_layer = select_prompt_indices(
        attentions, prompt_len=prompt_len, model_config=model_config,
        prefill_keep_ratio=prefill_keep_ratio, sink_tokens=sink_tokens,
        obs_window=obs_window, pool_kernel=pool_kernel, pooling=pooling)
    if len(keep_per_layer) != hc.cache_num_layers(cache):
        raise ValueError("attention 层数与缓存层数不一致")
    if keep_per_layer[0].shape[-1] < prompt_len:
        hc.evict_per_head(cache, keep_per_layer)
    state.prompt_len_phys = hc.cache_seq_len(cache)
    _pack_prompt(cache, state.prompt_len_phys, state)


def _pack_prompt(cache, prompt_phys_len: int, state: hc.HeteroTierState) -> None:
    """对 prompt 层 [0, prompt_phys_len) 做 KIVI 式 ``prompt_bits`` 位伪量化打包 (就地)。"""
    if prompt_phys_len <= 0 or state.prompt_bits >= 16:
        return
    token_slice = slice(0, prompt_phys_len)
    for layer in range(hc.cache_num_layers(cache)):
        k = cache.key_cache[layer]
        v = cache.value_cache[layer]
        k[:, :, token_slice, :] = quant.quantize_key(
            k[:, :, token_slice, :], state.prompt_bits, state.group_size)
        v[:, :, token_slice, :] = quant.quantize_value(
            v[:, :, token_slice, :], state.prompt_bits, state.group_size)
