"""KIVI 忠实基线 (CH-5 后端)。

参考: Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
(ICML'24)。

核心机制 (忠实复刻):
  1. K cache 沿 channel 维度存在系统性离群值 => 按 *per-channel* 量化 (沿 token 轴分组,
     每个 channel 在其 token 组内独立取 min/max), 抵消 channel 离群。
  2. V cache 无显著 channel 离群 => 按 *per-token* 量化 (沿 head_dim 轴分组, 每个 token
     在其 head_dim 组内独立量化)。
  3. 最近 ``residual_length`` 个 token 保留 FP16 残差 (streaming 新近 token 重要度未知,
     不立即量化); 当其滑出残差窗口并凑满一个 group_size 组时, 才批量量化为低比特。
  4. 分组粒度 ``group_size`` (KIVI 默认 32); 量化-反量化 (fake quant) 让模型据低比特栅格
     做真实前向, 测得量化对精度的真实影响。逻辑显存按目标位宽 + 分组元数据折算。

无任何 dense / 高精度回退: 量化即量化, 误差由位宽与残差长度控制 (遵守团队约定)。

与 eval 框架的衔接:
  - ``requires_attention=False`` / ``physically_evicts=False``: 纯量化, 不驱逐, 位置精确,
    可用快速 sdpa 注意力。
  - 物理张量仍为 fp16 (fake quant), 逻辑显存由 ``cache_memory_bytes`` 按混精折算。
"""
from __future__ import annotations

from eval.interface import CompressionConfig, KVCompressor
from . import _common


class KIVICompressor(KVCompressor):
    """KIVI: K per-channel / V per-token 分组量化 + 最近 FP16 残差。"""

    requires_attention = False
    physically_evicts = False

    def __init__(self, config: CompressionConfig):
        if config.bits is None:
            raise ValueError("KIVI 需要 bits (量化位宽, 原文 2-bit)")
        super().__init__(config)
        self.k_bits = int(config.extra.get("k_bits", config.bits))
        self.v_bits = int(config.extra.get("v_bits", config.bits))
        self.group_size = int(config.extra.get("group_size", 32))
        if config.recent_window is not None:
            self.residual_length = int(config.recent_window)
        else:
            self.residual_length = int(config.extra.get("residual_length", 128))
        if self.group_size <= 0:
            raise ValueError("KIVI group_size 必须为正整数")
        if self.residual_length < self.group_size:
            raise ValueError("KIVI residual_length 应 >= group_size")
        self._quantized_len = 0

    def reset(self, model_config):
        self._quantized_len = 0

    def _quantize_settled(self, cache):
        """量化已滑出残差窗口且凑满整组的 token (增量, 量化网格对齐到位置 0)。"""
        seq = _common.cache_seq_len(cache)
        gs = self.group_size
        target = (max(0, seq - self.residual_length) // gs) * gs
        if target <= self._quantized_len:
            return cache
        token_slice = slice(self._quantized_len, target)
        _common.quantize_token_slice(cache, token_slice, self.k_bits, self.v_bits, gs)
        self._quantized_len = target
        return cache

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        return self._quantize_settled(cache)

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        return self._quantize_settled(cache)

    def cache_memory_bytes(self, cache) -> int:
        return _common.kivi_logical_bytes(
            cache, self._quantized_len, self.k_bits, self.v_bits, self.group_size)
