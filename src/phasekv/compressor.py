"""PhaseKV 压缩器: 接 CH-3 的 KVCompressor 契约 (CH-2 架构, 阶段 A)。

把三层异构缓存 (hetero_cache) + prompt 层选择打包 (prompt_tier) + decode 热环/懒批冷压
(decode_tier) + roofline 预算分配器 (planner) 串成一个可被 eval 装载的压缩器。

阶段 A 定位 (idea 3.4): 操作标准 transformers Cache, 跑真实模型, 报告*真实精度* +
*真实逻辑显存* (按三层位宽解析折算) + *真实峰值显存*; **不**报告 decode 延迟加速 (那需
阶段 B 的融合 kernel)。严禁任何模拟数据; 无 dense 回退。

与基线的差异 (忠实实现, 非简化):
  - vs SnapKV (只压 prefill, decode 全 FP16, 无量化): PhaseKV 给 prompt 层叠加最狠低比特
    量化, 且额外管理 decode 冷层。
  - vs KIVI (统一量化, 不分 prompt/generated, 无选择): PhaseKV 把 prompt 单列为只读、最狠
    压的选择+量化层, decode 内部再分热/冷。
  - vs SCOPE (只驱逐不量化, 只压生成段、prompt 全保留): PhaseKV 反向——依"读放大"洞见对
    prompt 最激进压缩, 且叠加量化与阶段感知的压缩*时机* (prefill 一次 / decode 懒批)。
"""


#_init__()：读取保留比例、量化位宽、热窗口等参数。
#validate()：检查位宽只能为 2/3/4/8/16，比例必须在 (0, 1]。
#run_planner()：根据模型尺寸、Prompt 长度和目标显存比例自动规划参数。
#on_prefill()：调用 prompt_tier.select_and_pack() 压缩 Prompt KV。
#on_decode_step()：调用 decode_tier.maintain() 管理生成阶段的热/冷缓存。
#cache_memory_bytes()：返回三层缓存的逻辑显存占用。
#escribe()：导出当前配置和 planner 的规划结果。

from __future__ import annotations

from typing import Optional

from eval.interface import CompressionConfig, KVCompressor

from . import decode_tier
from . import hetero_cache as hc
from . import planner as planner_mod
from . import prompt_tier

_VALID_BITS = (2, 3, 4, 8, 16)


class PhaseKVCompressor(KVCompressor):
    """PhaseKV: 相位异构三层缓存 (prompt 只读最狠压 + decode 热环 FP16 + 冷层懒批量化)。"""

    requires_attention = True   # prompt 层选择需 prefill 注意力分数 (洞见 3)
    physically_evicts = True    # prompt 选择 + decode 冷层驱逐均物理改变张量

    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        ex = config.extra

        self.prefill_keep_ratio = (config.prefill_keep_ratio
                                   if config.prefill_keep_ratio is not None else 0.5)
        self.decode_keep_ratio = (config.decode_keep_ratio
                                  if config.decode_keep_ratio is not None else 1.0)
        self.recent_window = config.recent_window if config.recent_window is not None else 32
        self.sink_tokens = config.sink_tokens

        # 相位异构位宽 (洞见 2: prompt 读放大大 → 最狠; cold 读放大约 T/2 → 温和)。
        self.prompt_bits = int(ex.get("prompt_bits", config.bits if config.bits is not None else 2))
        self.cold_bits = int(ex.get("cold_bits", 4))#
        self.group_size = int(ex.get("group_size", 32))
        self.lazy_g = int(ex.get("lazy_g", 16))
        self.obs_window = int(ex.get("obs_window", 32))
        self.pool_kernel = int(ex.get("pool_kernel", 7))
        self.pooling = str(ex.get("pooling", "maxpool"))

        # planner 模式: 由 roofline 成本模型在目标显存下推导上述超参 (idea 3.2)。
        self.use_planner = bool(ex.get("use_planner", False))
        self.target_memory_ratio = ex.get("target_memory_ratio", None)
        if self.target_memory_ratio is not None:
            self.use_planner = True
        self.decode_len_hint = int(ex.get("decode_len_hint", 256))

        self._validate()
        self.planned: Optional[dict] = None
        self.state = self._fresh_state()

    def _validate(self) -> None:
        for nm, b in (("prompt_bits", self.prompt_bits), ("cold_bits", self.cold_bits)):
            if b not in _VALID_BITS:
                raise ValueError(f"{nm} 仅支持 {_VALID_BITS}, 实际 {b}")
        if not (0.0 < self.prefill_keep_ratio <= 1.0):
            raise ValueError(f"prefill_keep_ratio 必须落在 (0,1], 实际 {self.prefill_keep_ratio}")
        if not (0.0 < self.decode_keep_ratio <= 1.0):
            raise ValueError(f"decode_keep_ratio 必须落在 (0,1], 实际 {self.decode_keep_ratio}")
        if self.recent_window <= 0:
            raise ValueError("recent_window 必须为正整数")
        if self.group_size <= 0:
            raise ValueError("group_size 必须为正整数")
        if self.lazy_g <= 0:
            raise ValueError("lazy_g 必须为正整数")
        if self.obs_window <= 0:
            raise ValueError("obs_window 必须为正整数")

    def _fresh_state(self) -> hc.HeteroTierState:
        return hc.HeteroTierState(
            prompt_bits=self.prompt_bits, cold_bits=self.cold_bits,
            group_size=self.group_size, recent_window=self.recent_window)

    def reset(self, model_config) -> None:
        self.state = self._fresh_state()

    def _run_planner(self, prompt_len: int, model_config) -> None:
        """用 roofline 成本模型在目标显存约束下推导相位异构超参 (idea 3.2)。"""
        dims = planner_mod.ModelDims.from_config(model_config)
        inp = planner_mod.PlanInput(
            dims=dims, prompt_len=prompt_len, decode_len=self.decode_len_hint,
            target_memory_ratio=(self.target_memory_ratio
                                 if self.target_memory_ratio is not None else 0.3),
            group_size=self.group_size, sink_tokens=self.sink_tokens)
        out = planner_mod.plan(inp)
        self.prefill_keep_ratio = out.prefill_keep_ratio
        self.prompt_bits = out.prompt_bits
        self.recent_window = out.recent_window
        self.decode_keep_ratio = out.decode_keep_ratio
        self.cold_bits = out.cold_bits
        self.lazy_g = out.lazy_g
        self.sink_tokens = out.sink_tokens
        self.group_size = out.group_size
        # 同步到本条样本的三层状态。
        self.state.prompt_bits = self.prompt_bits
        self.state.cold_bits = self.cold_bits
        self.state.group_size = self.group_size
        self.state.recent_window = self.recent_window
        self.planned = {
            "prefill_keep_ratio": out.prefill_keep_ratio, "prompt_bits": out.prompt_bits,
            "recent_window": out.recent_window, "decode_keep_ratio": out.decode_keep_ratio,
            "cold_bits": out.cold_bits, "lazy_g": out.lazy_g,
            "est_memory_ratio": out.est_memory_ratio, "feasible": out.feasible,
            "roofline": out.roofline,
        }

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        if self.use_planner:
            self._run_planner(prompt_len, model_config)
        prompt_tier.select_and_pack(
            cache, attentions=attentions, prompt_len=prompt_len, model_config=model_config,
            state=self.state, prefill_keep_ratio=self.prefill_keep_ratio,
            sink_tokens=self.sink_tokens, obs_window=self.obs_window,
            pool_kernel=self.pool_kernel, pooling=self.pooling)
        return cache

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        decode_tier.maintain(
            cache, step=step, model_config=model_config, state=self.state,
            decode_keep_ratio=self.decode_keep_ratio, lazy_g=self.lazy_g)
        return cache

    def cache_memory_bytes(self, cache) -> int:
        return self.state.logical_bytes(cache)

    def describe(self) -> dict:
        base = super().describe()
        base["phasekv"] = {
            "prefill_keep_ratio": self.prefill_keep_ratio,
            "decode_keep_ratio": self.decode_keep_ratio,
            "recent_window": self.recent_window,
            "sink_tokens": self.sink_tokens,
            "prompt_bits": self.prompt_bits,
            "cold_bits": self.cold_bits,
            "group_size": self.group_size,
            "lazy_g": self.lazy_g,
            "obs_window": self.obs_window,
            "use_planner": self.use_planner,
        }
        if self.planned is not None:
            base["phasekv"]["planned"] = self.planned
        return base
