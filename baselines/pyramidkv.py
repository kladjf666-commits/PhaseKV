"""PyramidKV 忠实基线 (CH-5 后端)。

参考: Cai et al., "PyramidKV: Dynamic KV Cache Compression based on Pyramidal
Information Funneling" (2024)。

核心机制 (忠实复刻, 不简化):
  1. 金字塔式层间预算: 注意力信息在层间呈"漏斗"汇聚——低层注意力分散 (需更多 KV),
     高层高度集中于少数 token (需更少 KV)。故按层线性递减分配 *选择预算*: 第 0 层最大
     (max_num), 末层最小 (min_num), 二者均值等于目标平均选择预算, 从而全网平均预算守恒。
       avg_select = avg_budget - window         # 平均"非窗口"选择预算
       min_num    = avg_select // beta          # 末层 (最瘦)
       max_num    = 2 * avg_select - min_num     # 首层 (最宽)
       select_k(l)= round(max_num - l * (max_num - min_num) / (L - 1))
     每层再叠加始终保留的末尾观察窗口 window, 故第 l 层总保留 = select_k(l) + window。
  2. 层内选择沿用 SnapKV: 末尾 obs_window 个 query 对各 key 位置投票 → 1D 池化聚类
     → per-head top-k; 观察窗口 (最近 token) 始终保留。
  3. 只压 prefill; decode 不驱逐 (与 SnapKV 一致, 新 token 全 FP16 追加)。

与 runtime 的兼容性 (关键, 本轮已验证):
  - 金字塔令各层物理长度不同。本框架 runtime 的 decode ``cache_position`` 取自第 0 层
    物理长度, 而线性递减预算 (clamp 后仍单调非增) 令 *第 0 层恒为全网最长* → 该
    cache_position 覆盖所有更短的层; transformers eager 注意力按各层真实
    ``key_states.shape[-2]`` 切片因果掩码, 故逐层异长在 decode 下正确 (新 query 对每层
    全部保留 key 可见, 物理因果性成立)。
  - per-head 不同下标、同层同 budget 保持矩形; 保留下标升序, baked-in RoPE 对随后新
    query 的相对距离精确 (新 query 由 runtime 以真实绝对位置施加 RoPE)。
  - 逐层异长的逻辑显存由基类 ``cache_memory_bytes`` 按物理张量逐层求和如实反映 (eviction
    类, fp16 物理), 无需覆写。
"""
from __future__ import annotations

from typing import List, Optional

from eval.interface import CompressionConfig, KVCompressor
from . import _common


class PyramidKVCompressor(KVCompressor):
    """PyramidKV: 层间金字塔选择预算 + 层内 SnapKV 选择 (decode 不压)。"""

    requires_attention = True
    physically_evicts = True

    def __init__(self, config: CompressionConfig):
        if config.max_cache_tokens is None and config.prefill_keep_ratio is None:
            raise ValueError(
                "PyramidKV 需要 max_cache_tokens 或 prefill_keep_ratio 指定平均预算")
        super().__init__(config)
        self.obs_window = int(config.extra.get("obs_window", 32))
        self.pool_kernel = int(config.extra.get("pool_kernel", 7))
        self.pooling = str(config.extra.get("pooling", "maxpool"))
        self.beta = float(config.extra.get("beta", 20.0))
        if self.obs_window <= 0:
            raise ValueError("obs_window 必须为正整数")
        if self.beta <= 1.0:
            raise ValueError("beta 必须 > 1 (控制金字塔陡峭度; 越大首尾层差异越大)")

    def _avg_budget(self, prompt_len: int) -> int:
        """目标平均每层保留 token 数 (含观察窗口)。"""
        if self.config.max_cache_tokens is not None:
            budget = self.config.max_cache_tokens
        else:
            budget = int(round(prompt_len * self.config.prefill_keep_ratio))
        return min(prompt_len, max(self.obs_window, budget))

    def _layer_select_k(self, avg_budget: int, num_layers: int,
                        prefix_len: int, window: int) -> List[int]:
        """各层 *非窗口* 选择数 (金字塔: 首层 max_num 线性递减到末层 min_num)。

        prefix_len 为候选区长度 (= prompt_len - window); select_k 落在 [0, prefix_len]。
        线性插值保证首层取 max_num、末层取 min_num, 均值 = avg_select, 平均预算守恒。
        """
        avg_select = max(0, avg_budget - window)
        if num_layers <= 1:
            return [min(prefix_len, avg_select)]
        min_num = int(avg_select // self.beta)
        max_num = 2 * avg_select - min_num
        # 候选区不足以容纳 max_num 时按官方做法夹紧, 并把差额补回 min_num 端。
        if max_num > prefix_len:
            max_num = prefix_len
            min_num = max(0, 2 * avg_select - max_num)
        min_num = max(0, min(min_num, max_num))
        step = (max_num - min_num) / (num_layers - 1)
        out: List[int] = []
        for layer in range(num_layers):
            k = int(round(max_num - layer * step))
            out.append(max(0, min(prefix_len, k)))
        return out

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        if attentions is None:
            raise ValueError("PyramidKV 需要 attentions (prefill 观察窗口投票)")
        num_layers = _common.cache_num_layers(cache)
        num_kv_heads = _common.num_kv_heads_of(model_config)
        avg_budget = self._avg_budget(prompt_len)
        window = min(self.obs_window, prompt_len)
        prefix_len = prompt_len - window
        select_per_layer = self._layer_select_k(avg_budget, num_layers, prefix_len, window)

        keep_per_layer: List[Optional[object]] = []
        for layer in range(num_layers):
            select_k = select_per_layer[layer]
            if select_k >= prefix_len:
                keep_per_layer.append(None)  # 该层保留全部 prompt, 不压
                continue
            votes = _common.observation_votes(attentions[layer], num_kv_heads, window)
            pooled = _common.pool_scores(
                votes[..., :prefix_len], self.pool_kernel, self.pooling)
            keep = _common.topk_keep_per_head(
                pooled, select_k, prefix_len, recent_lo=prefix_len, recent_hi=prompt_len)
            keep_per_layer.append(keep)
        return _common.evict_per_head(cache, keep_per_layer)

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        return cache

    def describe(self) -> dict:
        base = super().describe()
        base["pyramidkv"] = {
            "obs_window": self.obs_window,
            "pool_kernel": self.pool_kernel,
            "pooling": self.pooling,
            "beta": self.beta,
        }
        return base
