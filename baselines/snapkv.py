"""SnapKV 忠实基线 (CH-5 后端)。

参考: Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation"
(NeurIPS'24)。

核心机制 (忠实复刻):
  1. 观察窗口 (observation window): 取 prompt 末尾 ``obs_window`` 个 token 作为"查询代理",
     用它们对 prompt 各位置的注意力分数投票, 估计哪些 prompt token 在生成时最被关注。
  2. per-head 聚类选择: 投票分数先沿序列做 1D 池化 (默认 maxpool, kernel=7), 使被选中的
     token 形成连续簇 (保留局部上下文); 再 *逐 head* 取 top-k。
  3. 始终保留观察窗口本身 (最近 token), 与选中的历史 token 合并为该 head 的保留集合。
  4. SnapKV 只压缩 prefill; decode 阶段不再驱逐 (新 token 全部以 FP16 追加)。

与 eval 框架的衔接:
  - ``requires_attention=True``: prefill 需要注意力分数做观察窗口投票 (runtime 以 eager
    注意力提供)。
  - per-head 不同下标但同一 budget, 物理张量保持矩形, 各层长度一致 (与 runtime 的统一
    cache_position / 因果掩码兼容)。
  - 保留下标按时间升序, baked-in RoPE 对新 query 相对距离精确。
"""
from __future__ import annotations

import torch

from eval.interface import CompressionConfig, KVCompressor
from . import _common


class SnapKVCompressor(KVCompressor):
    """SnapKV: prefill 观察窗口投票 + per-head 聚类 pooling 选择 (decode 不压)。"""

    requires_attention = True
    physically_evicts = True

    def __init__(self, config: CompressionConfig):
        if config.max_cache_tokens is None and config.prefill_keep_ratio is None:
            raise ValueError("SnapKV 需要 max_cache_tokens 或 prefill_keep_ratio 指定 prompt 预算")
        super().__init__(config)
        self.obs_window = int(config.extra.get("obs_window", 32))
        self.pool_kernel = int(config.extra.get("pool_kernel", 7))
        self.pooling = str(config.extra.get("pooling", "maxpool"))
        if self.obs_window <= 0:
            raise ValueError("obs_window 必须为正整数")

    def _budget(self, prompt_len: int) -> int:
        if self.config.max_cache_tokens is not None:
            budget = self.config.max_cache_tokens
        else:
            budget = int(round(prompt_len * self.config.prefill_keep_ratio))
        return min(prompt_len, max(self.obs_window, budget))

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        if attentions is None:
            raise ValueError("SnapKV 需要 attentions (prefill 观察窗口投票)")
        budget = self._budget(prompt_len)
        if prompt_len <= budget:
            return cache
        num_kv_heads = _common.num_kv_heads_of(model_config)
        w = min(self.obs_window, prompt_len)
        prefix_len = prompt_len - w           # 候选区: 观察窗口之外的历史 token
        select_k = max(0, budget - w)         # 从候选区每头选取的数量
        keep_per_layer = []
        for layer in range(_common.cache_num_layers(cache)):
            votes = _common.observation_votes(attentions[layer], num_kv_heads, w)
            pooled = _common.pool_scores(votes[..., :prefix_len], self.pool_kernel, self.pooling)
            keep = _common.topk_keep_per_head(
                pooled, select_k, prefix_len, recent_lo=prefix_len, recent_hi=prompt_len)
            keep_per_layer.append(keep)
        return _common.evict_per_head(cache, keep_per_layer)

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        return cache
