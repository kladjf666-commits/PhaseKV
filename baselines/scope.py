"""SCOPE 忠实基线 (CH-5 后端)。

参考: Wu et al., "SCOPE: Optimizing Key-Value Cache Compression in Long-context
Generation" (ACL'25 Findings)。SCOPE 自称首个从 phase 层面压缩 KV 的工作, 是与本项目
PhaseKV 最近、最直接对打的相位基线 (必比)。

核心观察与机制 (忠实复刻):
  - 长输出生成中 heavy-hitter 会随解码漂移: prefill 时选定的重要 token 未必覆盖 decode
    阶段真正需要的 token。故 SCOPE 把 prefill 与 decode 的 KV 压缩解耦:
  - prefill 阶段: *保留* prompt 的 KV (把压缩预算让给 decode 段), 不破坏长上下文信息。
  - decode 阶段: 仅对 *生成段* 施加固定预算 = 滑动窗口 (最近 ``recent_window``, 始终保留)
    + heavy-hitter (按生成段累计注意力逐 head 选取), 永不触碰 prompt。

与 eval 框架的衔接:
  - ``requires_attention=True``: decode heavy-hitter 需要注意力分数 (runtime 以 eager 提供)。
  - 生成段逐 head 选择, prompt 段所有 head 共享全保留; 各层总长一致 (prompt_len +
    decode 预算), 与 runtime 的统一 cache_position / 因果掩码兼容。
  - 保留下标按时间升序, baked-in RoPE 对新 query 相对距离精确。
"""
from __future__ import annotations

import torch

from eval.interface import CompressionConfig, KVCompressor
from . import _common


class SCOPECompressor(KVCompressor):
    """SCOPE: prefill 保留 prompt + decode 段滑动窗口 + heavy-hitter 固定预算 (逐 head)。"""

    requires_attention = True
    physically_evicts = True

    def __init__(self, config: CompressionConfig):
        if config.recent_window is None:
            raise ValueError("SCOPE 需要 recent_window (decode 滑动窗口)")
        if config.max_cache_tokens is None and config.decode_keep_ratio is None:
            raise ValueError("SCOPE 需要 max_cache_tokens 或 decode_keep_ratio 指定 decode 段预算")
        super().__init__(config)
        self._prompt_len = 0
        self._gen_imp = None  # 每层生成段累计注意力重要度 [b, Hkv, gen_phys]

    def reset(self, model_config):
        self._prompt_len = 0
        self._gen_imp = None

    def _gen_budget(self, gen_count: int) -> int:
        if self.config.max_cache_tokens is not None:
            budget = self.config.max_cache_tokens
        else:
            budget = int(round(gen_count * self.config.decode_keep_ratio))
        return max(self.config.recent_window, budget)

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        self._prompt_len = _common.cache_seq_len(cache)
        self._gen_imp = None
        return cache

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        if attentions is None:
            raise ValueError("SCOPE 需要 attentions (decode heavy-hitter 投票)")
        num_kv_heads = _common.num_kv_heads_of(model_config)
        num_layers = _common.cache_num_layers(cache)
        seq_len = _common.cache_seq_len(cache)
        gen_count = seq_len - self._prompt_len
        if gen_count <= 0:
            return cache
        recent = self.config.recent_window
        budget = self._gen_budget(gen_count)
        do_evict = gen_count > budget
        if self._gen_imp is None:
            self._gen_imp = [None] * num_layers

        new_imp = []
        keep_per_layer = [] if do_evict else None
        for layer in range(num_layers):
            step_imp = attentions[layer][:, :, :, self._prompt_len:].float().sum(dim=2)  # [b, Hq, gen_count]
            step_imp = _common.aggregate_to_kv_heads(step_imp, num_kv_heads)             # [b, Hkv, gen_count]
            imp = self._gen_imp[layer]
            if imp is None:
                imp = step_imp.new_zeros(step_imp.shape[0], step_imp.shape[1], 0)
            if imp.shape[-1] < gen_count:
                pad = step_imp.new_zeros(imp.shape[0], imp.shape[1], gen_count - imp.shape[-1])
                imp = torch.cat([imp, pad], dim=-1)
            imp = imp + step_imp

            if do_evict:
                cand_len = gen_count - recent            # 候选区: 生成段中较老者
                select_k = max(0, budget - recent)
                cand = imp[..., :cand_len]
                if select_k > 0:
                    top = torch.topk(cand, k=min(select_k, cand_len), dim=-1).indices.long()
                else:
                    top = imp.new_zeros((imp.shape[0], imp.shape[1], 0), dtype=torch.long)
                recent_idx = torch.arange(cand_len, gen_count, device=imp.device, dtype=torch.long)
                recent_idx = recent_idx.view(1, 1, -1).expand(imp.shape[0], imp.shape[1], recent)
                gen_keep, _ = torch.sort(torch.cat([top, recent_idx], dim=-1), dim=-1)
                prompt_idx = torch.arange(0, self._prompt_len, device=imp.device, dtype=torch.long)
                prompt_idx = prompt_idx.view(1, 1, -1).expand(imp.shape[0], imp.shape[1], self._prompt_len)
                keep = torch.cat([prompt_idx, self._prompt_len + gen_keep], dim=-1)
                keep_per_layer.append(keep)
                new_imp.append(imp.gather(-1, gen_keep))
            else:
                new_imp.append(imp)

        self._gen_imp = new_imp
        if do_evict:
            cache = _common.evict_per_head(cache, keep_per_layer)
        return cache
