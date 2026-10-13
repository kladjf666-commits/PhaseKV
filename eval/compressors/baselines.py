"""参考 KV 压缩器 (QA / CH-3 维护)。

这些是评估框架自带的"参考实现", 用途有二:
  1. 验证 eval 框架本身正确 (harness self-test fixtures);
  2. 作为团队方法接入前的占位基线, 供对比。

竞品级基线 (H2O / SnapKV / KIVI 原版等) 由后端 CH-5 在 baselines/ 实现并接入,
同样继承 ``KVCompressor`` 契约。

位置正确性说明:
  - FullCache / KVQuant: 不驱逐, 位置完全精确。
  - RecentWindow (无 sink): 保留连续最近窗口, RoPE 相对距离精确。
  - StreamingLLM (sink+recent): 存在被驱逐区间, 本参考实现采用真实绝对位置 (不重编码),
    与原论文"位置重编码"略有差异; 详见 docs/qa/RISK_LIST.md (R-04)。
"""
from __future__ import annotations

import torch

from ..interface import CompressionConfig, KVCompressor
from . import common


class FullCache(KVCompressor):
    """不做任何压缩的参考基线 (上界精度 / 满显存)。"""

    requires_attention = False
    physically_evicts = False

    def __init__(self, config: CompressionConfig | None = None):
        super().__init__(config or CompressionConfig(name="full"))

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        return cache

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        return cache


class RecentWindow(KVCompressor):
    """仅保留最近 ``recent_window`` 个 token (滑动窗口)。位置精确。"""

    requires_attention = False
    physically_evicts = True

    def __init__(self, config: CompressionConfig):
        if config.recent_window is None:
            raise ValueError("RecentWindow 需要 config.recent_window")
        super().__init__(config)

    def _trim(self, cache):
        seq_len = common.cache_seq_len(cache)
        budget = self.config.recent_window
        if seq_len <= budget:
            return cache
        keep = torch.arange(seq_len - budget, seq_len, dtype=torch.long)
        return common.slice_cache(cache, keep)

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        return self._trim(cache)

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        return self._trim(cache)


class StreamingLLM(KVCompressor):
    """Attention-sink + 最近窗口: 保留前 ``sink_tokens`` + 最近 ``recent_window``。

    参考: Xiao et al., "Efficient Streaming Language Models with Attention Sinks"。
    本参考实现使用真实绝对位置 (不做位置重编码), 见 RISK_LIST R-04。
    """

    requires_attention = False
    physically_evicts = True

    def __init__(self, config: CompressionConfig):
        if config.recent_window is None:
            raise ValueError("StreamingLLM 需要 config.recent_window")
        super().__init__(config)

    def _trim(self, cache):
        seq_len = common.cache_seq_len(cache)
        budget = self.config.sink_tokens + self.config.recent_window
        if seq_len <= budget:
            return cache
        keep = common.keep_sink_recent_indices(
            seq_len, self.config.sink_tokens, self.config.recent_window)
        return common.slice_cache(cache, keep)

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        return self._trim(cache)

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        return self._trim(cache)


class KVQuant(KVCompressor):
    """KV Cache 伪量化 (非对称 min-max, 沿 head_dim 分组)。位置精确, 不驱逐。

    用于度量"量化位宽 -> 精度/显存"的权衡。逻辑显存按 ``config.bits`` 折算。
    """

    requires_attention = False
    physically_evicts = False

    def __init__(self, config: CompressionConfig):
        if config.bits is None:
            raise ValueError("KVQuant 需要 config.bits")
        super().__init__(config)
        self._quant_axis = config.extra.get("quant_axis", -1)

    def _quant_cache(self, cache):
        bits = self.config.bits
        if bits >= 16:
            return cache
        for layer in range(common.cache_num_layers(cache)):
            cache.key_cache[layer] = common.fake_quantize(
                cache.key_cache[layer], bits, axis=self._quant_axis)
            cache.value_cache[layer] = common.fake_quantize(
                cache.value_cache[layer], bits, axis=self._quant_axis)
        return cache

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        return self._quant_cache(cache)

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        return self._quant_cache(cache)

    def cache_memory_bytes(self, cache) -> int:
        head_dim = cache.key_cache[0].shape[-1] if cache.key_cache else 1
        group_size = head_dim if self._quant_axis in (-1, cache.key_cache[0].dim() - 1) else 1
        return common.quantized_cache_bytes(cache, self.config.bits, group_size)
