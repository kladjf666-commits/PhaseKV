"""基于注意力分数的 KV 压缩器 (QA / CH-3 维护)。

包含:
  - ObservedAttentionEviction: H2O 式 heavy-hitter 驱逐 (按累计注意力保留重要 token)。
  - HeterogeneousReference: 演示"prefill 与 decode 异构策略"(用户核心 idea) 的参考实现
      * prefill: 按注意力 heavy-hitter 对静态 prompt 做结构化压缩;
      * decode:  对增长的生成段仅保留最近窗口 (廉价流式策略)。

两者均需 ``requires_attention=True``, 运行时会以 eager attention 提供分数。
"""
from __future__ import annotations

from typing import List, Optional

import torch

from ..interface import CompressionConfig, KVCompressor
from . import common


def _layer_importance(attn_layer: torch.Tensor) -> torch.Tensor:
    """由注意力张量 [batch, heads, q_len, kv_len] 计算每个 key 位置的重要度 [kv_len]。"""
    return attn_layer.float().sum(dim=(0, 1, 2))


class ObservedAttentionEviction(KVCompressor):
    """H2O 式: 保留 sink + 最近窗口 + 历史累计注意力最高的 heavy-hitter。

    参考: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference"。
    """

    requires_attention = True
    physically_evicts = True

    def __init__(self, config: CompressionConfig):
        if config.recent_window is None:
            raise ValueError("ObservedAttentionEviction 需要 config.recent_window")
        if config.prefill_keep_ratio is None and config.max_cache_tokens is None:
            raise ValueError("需要 prefill_keep_ratio 或 max_cache_tokens 指定预算")
        super().__init__(config)
        self._importance: Optional[List[torch.Tensor]] = None

    def reset(self, model_config):
        self._importance = None

    def _budget(self, seq_len: int) -> int:
        if self.config.max_cache_tokens is not None:
            return min(seq_len, self.config.max_cache_tokens)
        return max(self.config.sink_tokens + self.config.recent_window,
                   int(round(seq_len * self.config.prefill_keep_ratio)))

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        if attentions is None:
            raise ValueError("ObservedAttentionEviction 需要 attentions")
        num_layers = common.cache_num_layers(cache)
        budget = self._budget(prompt_len)
        sink, recent = self.config.sink_tokens, self.config.recent_window
        self._importance = []
        keep_per_layer = []
        for layer in range(num_layers):
            imp = _layer_importance(attentions[layer])  # [prompt_len]
            self._importance.append(imp.clone())
            if prompt_len <= budget:
                keep = torch.arange(prompt_len, dtype=torch.long, device=imp.device)
            else:
                forced = common.keep_sink_recent_indices(prompt_len, sink, recent, imp.device)
                masked = imp.clone()
                masked[forced] = float("-inf")  # 已强制保留, 不参与 top-k
                extra_k = max(0, budget - forced.numel())
                if extra_k > 0:
                    top = torch.topk(masked, k=min(extra_k, prompt_len)).indices
                    keep = torch.unique(torch.cat([forced, top]))
                else:
                    keep = forced
            keep_per_layer.append(keep)
        cache = common.slice_cache(cache, keep_per_layer)
        # 同步收缩 importance 以与物理缓存对齐
        self._importance = [self._importance[l].index_select(0, torch.sort(keep_per_layer[l])[0].to(self._importance[l].device))
                            for l in range(num_layers)]
        return cache

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        if attentions is None:
            raise ValueError("ObservedAttentionEviction 需要 attentions")
        num_layers = common.cache_num_layers(cache)
        seq_len = common.cache_seq_len(cache)
        sink, recent = self.config.sink_tokens, self.config.recent_window
        budget = max(sink + recent, self._budget(seq_len))
        keep_per_layer = []
        new_importance = []
        for layer in range(num_layers):
            step_imp = _layer_importance(attentions[layer])  # [kv_len = seq_len]
            imp = self._importance[layer]
            # 新增 token 的累计重要度从本步注意力起算
            if imp.numel() < seq_len:
                pad = torch.zeros(seq_len - imp.numel(), device=imp.device, dtype=imp.dtype)
                imp = torch.cat([imp, pad])
            imp = imp + step_imp
            if seq_len <= budget:
                keep = torch.arange(seq_len, dtype=torch.long, device=imp.device)
            else:
                forced = common.keep_sink_recent_indices(seq_len, sink, recent, imp.device)
                masked = imp.clone()
                masked[forced] = float("-inf")
                extra_k = max(0, budget - forced.numel())
                top = torch.topk(masked, k=min(extra_k, seq_len)).indices if extra_k > 0 else forced[:0]
                keep = torch.unique(torch.cat([forced, top]))
            keep_per_layer.append(keep)
            new_importance.append(imp.index_select(0, torch.sort(keep)[0]))
        cache = common.slice_cache(cache, keep_per_layer)
        self._importance = new_importance
        return cache


class HeterogeneousReference(KVCompressor):
    """异构 prefill/decode 压缩参考 (演示用户 idea)。

    prefill: 注意力 heavy-hitter, 每层保留固定预算 K0 = round(prompt_len * prefill_keep_ratio)
             (含 sink), 对静态 prompt 做结构化稀疏。
    decode:  对生成段仅保留最近 ``recent_window``; 永不触碰已选中的 prefill 块。
    """

    requires_attention = True
    physically_evicts = True

    def __init__(self, config: CompressionConfig):
        if config.prefill_keep_ratio is None:
            raise ValueError("HeterogeneousReference 需要 config.prefill_keep_ratio")
        if config.recent_window is None:
            raise ValueError("HeterogeneousReference 需要 config.recent_window (decode 窗口)")
        super().__init__(config)
        self._k0: int = 0  # prefill 阶段每层保留的 token 数 (统一)

    def reset(self, model_config):
        self._k0 = 0

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        if attentions is None:
            raise ValueError("HeterogeneousReference 需要 attentions")
        num_layers = common.cache_num_layers(cache)
        sink = self.config.sink_tokens
        k0 = max(sink, int(round(prompt_len * self.config.prefill_keep_ratio)))
        k0 = min(k0, prompt_len)
        self._k0 = k0
        keep_per_layer = []
        for layer in range(num_layers):
            if prompt_len <= k0:
                keep = torch.arange(prompt_len, dtype=torch.long, device=cache.key_cache[layer].device)
            else:
                imp = _layer_importance(attentions[layer])
                forced = torch.arange(0, sink, dtype=torch.long, device=imp.device)
                masked = imp.clone()
                if sink > 0:
                    masked[:sink] = float("-inf")
                extra_k = max(0, k0 - sink)
                top = torch.topk(masked, k=min(extra_k, prompt_len)).indices
                keep = torch.unique(torch.cat([forced, top]))
            keep_per_layer.append(keep)
        return common.slice_cache(cache, keep_per_layer)

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        seq_len = common.cache_seq_len(cache)
        num_decode = seq_len - self._k0
        if num_decode <= self.config.recent_window:
            return cache
        head = torch.arange(0, self._k0, dtype=torch.long)
        tail = torch.arange(seq_len - self.config.recent_window, seq_len, dtype=torch.long)
        keep = torch.unique(torch.cat([head, tail]))
        return common.slice_cache(cache, keep)
