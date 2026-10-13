"""受控分段消融压缩器 (QA / CH-3 维护)。

用途: 为 accuracy-per-byte 拆分图 (idea 文档第 2 节图2) 提供一个把"段"作为唯一自变量
的受控压缩器。SnapKV 只压 prompt、SCOPE 只压 generated, 但两者用的是不同的选择机制
(池化投票 vs heavy-hitter+recent), 直接对比会把"段"与"机制"混淆。本压缩器在 *同一*
注意力 heavy-hitter 选择机制下, 只压缩其中一段、另一段完全保留, 从而干净地度量"每字节
精度"随段的差异:

- segment='prompt'   : prefill 后按 prompt 注意力累计重要度做 heavy-hitter top-k 压到预算;
                       decode 段全保留 (on_decode_step 为 no-op)。
- segment='generated': prefill 全保留 prompt; decode 过程仅对生成段按累计注意力压到预算
                       (始终强制保留最近窗口), 永不触碰 prompt。

预算 (二选一):
- max_cache_tokens: 目标段保留 token 数的硬上限。
- prefill_keep_ratio (prompt 段) / decode_keep_ratio (generated 段): 目标段保留比例 (0,1]。
recent_window: 仅 generated 段使用, 强制保留的最近窗口 (默认 8), 防止驱逐刚生成的 token
              导致解码退化。

位置正确性: 仅按时间升序保留 token 子集, 保留的 K 已 baked-in 原始 RoPE; runtime 对新
query 传入真实绝对位置, 故相对距离精确 (与 common.slice_cache / runtime.py 一致)。
"""
from __future__ import annotations

from typing import List, Optional

import torch

from ..interface import CompressionConfig, KVCompressor
from . import common


def _layer_importance(attn_layer: torch.Tensor) -> torch.Tensor:
    """由注意力张量 [batch, heads, q_len, kv_len] 计算每个 key 位置的重要度 [kv_len]。"""
    return attn_layer.float().sum(dim=(0, 1, 2))


class SegmentAblationEvict(KVCompressor):
    """受控分段消融: 在同一 heavy-hitter 机制下只压 prompt 段或只压 generated 段。"""

    requires_attention = True
    physically_evicts = True

    _DEFAULT_RECENT = 8

    def __init__(self, config: CompressionConfig):
        segment = config.extra.get("segment")
        if segment not in ("prompt", "generated"):
            raise ValueError("segment_ablation 需要 extra.segment ∈ {'prompt','generated'}")
        if (config.max_cache_tokens is None
                and config.prefill_keep_ratio is None
                and config.decode_keep_ratio is None):
            raise ValueError(
                "segment_ablation 需要 max_cache_tokens, 或目标段对应的 "
                "prefill_keep_ratio(prompt)/decode_keep_ratio(generated) 指定预算")
        super().__init__(config)
        self._segment = segment
        self._prompt_len = 0
        self._gen_imp: Optional[List[torch.Tensor]] = None  # 逐层生成段累计重要度

    def reset(self, model_config):
        self._prompt_len = 0
        self._gen_imp = None

    def _prompt_budget(self, prompt_len: int) -> int:
        if self.config.max_cache_tokens is not None:
            return min(prompt_len, self.config.max_cache_tokens)
        ratio = self.config.prefill_keep_ratio
        return max(1, min(prompt_len, int(round(prompt_len * ratio))))

    def _gen_budget(self, logical_total: int, recent: int) -> int:
        # 预算按"逻辑累计生成数"计算, 而非物理(驱逐后)长度: 否则一旦驱逐到 recent
        # floor, 物理长度被钉死在 floor, 比例预算会自锁死在最小值。
        if self.config.max_cache_tokens is not None:
            budget = min(logical_total, self.config.max_cache_tokens)
        else:
            budget = int(round(logical_total * self.config.decode_keep_ratio))
        return max(recent, min(logical_total, budget))

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        self._prompt_len = common.cache_seq_len(cache)
        if self._segment != "prompt":
            return cache
        if attentions is None:
            raise ValueError("segment_ablation(prompt) 需要 attentions")
        budget = self._prompt_budget(self._prompt_len)
        if self._prompt_len <= budget:
            return cache
        num_layers = common.cache_num_layers(cache)
        keep_per_layer = []
        for layer in range(num_layers):
            imp = _layer_importance(attentions[layer])  # [prompt_len]
            top = torch.topk(imp, k=min(budget, imp.numel())).indices
            keep_per_layer.append(top)
        cache = common.slice_cache(cache, keep_per_layer)
        self._prompt_len = common.cache_seq_len(cache)
        return cache

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        if self._segment != "generated":
            return cache
        if attentions is None:
            raise ValueError("segment_ablation(generated) 需要 attentions")
        seq_len = common.cache_seq_len(cache)
        gen_count = seq_len - self._prompt_len   # 物理生成段长度(可能已被驱逐)
        if gen_count <= 0:
            return cache
        recent = self.config.recent_window or self._DEFAULT_RECENT
        logical_total = step + 1                  # 至今逻辑生成 token 数(不受驱逐影响)
        budget = self._gen_budget(logical_total, recent)
        num_layers = common.cache_num_layers(cache)
        if self._gen_imp is None:
            self._gen_imp = [None] * num_layers
        do_evict = gen_count > budget
        keep_per_layer = [] if do_evict else None
        new_imp = []
        for layer in range(num_layers):
            step_imp = attentions[layer][:, :, :, self._prompt_len:].float().sum(dim=(0, 1, 2))  # [gen_count]
            imp = self._gen_imp[layer]
            if imp is None:
                imp = step_imp.new_zeros(0)
            if imp.numel() < gen_count:
                imp = torch.cat([imp, step_imp.new_zeros(gen_count - imp.numel())])
            imp = imp + step_imp
            if do_evict:
                recent_idx = torch.arange(gen_count - recent, gen_count,
                                          device=imp.device, dtype=torch.long)
                masked = imp.clone()
                masked[recent_idx] = float("-inf")
                extra_k = max(0, budget - recent)
                if extra_k > 0:
                    top = torch.topk(masked, k=min(extra_k, gen_count)).indices
                    gen_keep = torch.unique(torch.cat([top, recent_idx]))
                else:
                    gen_keep = torch.unique(recent_idx)
                gen_keep, _ = torch.sort(gen_keep)
                prompt_idx = torch.arange(self._prompt_len, device=imp.device, dtype=torch.long)
                keep = torch.cat([prompt_idx, self._prompt_len + gen_keep])
                keep_per_layer.append(keep)
                new_imp.append(imp.index_select(0, gen_keep))
            else:
                new_imp.append(imp)
        self._gen_imp = new_imp
        if do_evict:
            cache = common.slice_cache(cache, keep_per_layer)
        return cache
