"""PhaseKV Roofline 预算分配器 (CH-2 架构)。

把 idea 文档第 2/3.2 节的三条硬件洞见落成*可计算*的成本模型, 并据此在给定目标显存下
为相位异构缓存分配超参 (prompt 选择比/位宽、decode 热窗口、冷层位宽/选择比、懒批 G)。

核心量化关系 (全部为解析推导, 非模拟实验数据):

  洞见 1 (相位瓶颈不同):
    - prefill 注意力 FLOPs ~ O(P^2), 字节 ~ O(P); 算术强度高 => 计算受限。
    - decode 每步注意力 FLOPs ~ O(S), 读字节 ~ O(S); 算术强度 ~ 注意力头数/KV头数,
      远低于 ridge point => 带宽受限。
  洞见 2 (读放大):
    - 一个 prompt token 的 KV 在随后每个 decode step 都被读取一次 => 读放大 = T。
    - 第 i 个 generated token 的 KV 仅被其后的 (T-i) 步读取 => 平均读放大 ~ T/2。
    - 因此把 1 字节从 prompt 层压掉, 在整段解码中省下的 HBM 流量是把 1 字节从 decode
      层压掉的约 2 倍 => 应给 prompt 层更激进的预算。

分配器目标 (idea 3.2 的两层字典序目标, 受显存约束):
  1. 主目标 "目标显存下精度最高": 在 "稳态逻辑显存 <= 目标" 的可行域内, *用满预算* ——
     选逻辑字节 (= 保留 token 数 x 位宽) 最大的配置。逻辑字节是精度的单调代理: 保留越多
     token、位宽越高, 伪量化/驱逐引入的误差越小 => 精度越高。这修正了"只压到最激进而浪费
     宽松预算"的退化 (那会让 target 放宽也拿不到更高精度)。
  2. 次目标 "decode 每步读字节最少": 同等逻辑字节预算下, 最小化整段解码 KV 读流量
     (= sum_token 逻辑字节 * 该 token 的读放大)。因 prompt 层读放大 = T、generated 层 ~ T/2,
     同样多的字节放到 generated (冷/热) 层比放到 prompt 层读流量更省 => 该次目标等价于
     "把预算优先分给读放大低的 decode 层" (洞见 2 推论)。
不可行域 (无配置满足约束) 退化为最激进 (逻辑字节最小) 以尽量逼近约束。

这是真实的受约束离散优化 (网格搜索 + 成本模型), 输出是*超参配置*; 真实精度/加速比仍由
eval 实测, 分配器不产出任何被当作实测结果上报的数值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import quant


# --------------------------------------------------------------------------- #
# 硬件与模型规格
# --------------------------------------------------------------------------- #
@dataclass
class HardwareSpec:
    """目标 GPU 的 roofline 规格。

    bandwidth 为 HBM 实测/标称带宽 (字节/秒); peak_flops 为 fp16 矩阵乘有效吞吐
    (FLOP/秒)。peak_flops 仅影响 ridge point 的精确位置, 不影响 decode 带宽受限这一
    定性结论; 终版图 1 的精确 FLOPs 应由 CH-3 微基准实测标定后覆盖此默认值。
    """

    name: str = "RTX 4080 SUPER"
    peak_flops: float = 50.0e12          # ~fp16 张量核有效吞吐 (近似, 可被实测覆盖)
    hbm_bandwidth: float = 736.0e9       # 256-bit GDDR6X @ 23Gbps = 736 GB/s (标称)

    @property
    def ridge_intensity(self) -> float:
        """ridge point 算术强度 (FLOP/Byte): 高于此为计算受限, 低于此为带宽受限。"""
        return self.peak_flops / self.hbm_bandwidth


@dataclass
class ModelDims:
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, model_config) -> "ModelDims":
        num_heads = getattr(model_config, "num_attention_heads")
        num_kv = getattr(model_config, "num_key_value_heads", num_heads)
        head_dim = getattr(model_config, "head_dim", None)
        if head_dim is None:
            head_dim = model_config.hidden_size // num_heads
        return cls(model_config.num_hidden_layers, num_heads, num_kv, head_dim)


# --------------------------------------------------------------------------- #
# Roofline 诊断 (供洞见 1 的图 1 与策略论证)
# --------------------------------------------------------------------------- #
def prefill_attention_intensity(prompt_len: int, dims: ModelDims, batch: int = 1) -> float:
    """prefill 自注意力的算术强度 (FLOP/Byte)。

    QK^T 与 AV 各 ~ 2 * B * H * P^2 * d FLOPs; KV 写入 HBM ~ 2 * B * H_kv * P * d * 2B。
    """
    p = float(prompt_len)
    flops = 2.0 * 2.0 * batch * dims.num_attention_heads * p * p * dims.head_dim
    kv_bytes = 2.0 * batch * dims.num_kv_heads * p * dims.head_dim * 2.0
    return flops / max(1.0, kv_bytes)


def decode_attention_intensity(cache_len: int, dims: ModelDims, batch: int = 1) -> float:
    """decode 单步自注意力的算术强度 (FLOP/Byte)。

    每步 1 个 query 对 S 个 key: FLOPs ~ 2 * 2 * B * H * S * d; 读 KV ~ 2 * B * H_kv * S * d * 2B。
    强度 ~ H / H_kv 量级, 与 S 无关, 通常远低于 ridge => 带宽受限。
    """
    s = float(max(1, cache_len))
    flops = 2.0 * 2.0 * batch * dims.num_attention_heads * s * dims.head_dim
    kv_bytes = 2.0 * batch * dims.num_kv_heads * s * dims.head_dim * 2.0
    return flops / max(1.0, kv_bytes)


def roofline_report(prompt_len: int, decode_len: int, dims: ModelDims,
                    hw: HardwareSpec, batch: int = 1) -> dict:
    """产出 roofline 诊断 (用于图 1 与策略说明), 全部为解析量。"""
    pf = prefill_attention_intensity(prompt_len, dims, batch)
    final_len = prompt_len + decode_len
    dc = decode_attention_intensity(final_len, dims, batch)
    ridge = hw.ridge_intensity
    return {
        "hardware": hw.name,
        "ridge_intensity_flop_per_byte": ridge,
        "prefill_intensity": pf,
        "prefill_bound": "compute" if pf >= ridge else "bandwidth",
        "decode_intensity": dc,
        "decode_bound": "compute" if dc >= ridge else "bandwidth",
        "kv_read_amplification_prompt": decode_len,
        "kv_read_amplification_generated_avg": decode_len / 2.0,
    }


# --------------------------------------------------------------------------- #
# 预算分配
# --------------------------------------------------------------------------- #
@dataclass
class PlanInput:
    dims: ModelDims
    prompt_len: int
    decode_len: int
    target_memory_ratio: float = 0.3       # 目标稳态逻辑显存 / 全量 fp16
    batch: int = 1
    group_size: int = 32
    hardware: HardwareSpec = field(default_factory=HardwareSpec)
    # 搜索网格 (可覆盖)
    prompt_bits_grid: Tuple[int, ...] = (2, 3, 4)
    cold_bits_grid: Tuple[int, ...] = (2, 4, 8)
    prompt_keep_grid: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    cold_keep_grid: Tuple[float, ...] = (0.3, 0.5, 0.7, 1.0)
    recent_window_grid: Tuple[int, ...] = (64, 128, 256)
    lazy_g_grid: Tuple[int, ...] = (8, 16, 32)
    sink_tokens: int = 4


@dataclass
class PlanOutput:
    prefill_keep_ratio: float
    prompt_bits: int
    recent_window: int
    decode_keep_ratio: float
    cold_bits: int
    lazy_g: int
    sink_tokens: int
    group_size: int
    # 诊断
    est_steady_bytes: int
    full_fp16_bytes: int
    est_memory_ratio: float
    est_decode_read_bytes: float
    roofline: dict
    feasible: bool

    def to_compressor_kwargs(self) -> dict:
        """转成 build_compressor 可直接消费的参数。"""
        return {
            "prefill_keep_ratio": self.prefill_keep_ratio,
            "decode_keep_ratio": self.decode_keep_ratio,
            "sink_tokens": self.sink_tokens,
            "recent_window": self.recent_window,
            "prompt_bits": self.prompt_bits,
            "cold_bits": self.cold_bits,
            "lazy_g": self.lazy_g,
            "group_size": self.group_size,
        }


def full_fp16_kv_bytes(dims: ModelDims, total_tokens: int, batch: int = 1) -> int:
    """全量未压缩 fp16 KV 的字节数 (K、V 各一份)。"""
    return 2 * batch * dims.num_layers * dims.num_kv_heads * total_tokens * dims.head_dim * 2


def _steady_tokens(prompt_len: int, decode_len: int, prompt_keep: float,
                   recent_window: int, cold_keep: float, sink: int) -> Tuple[int, int, int]:
    """估算解码结束时三层的稳态 token 数 (prompt 保留 / 冷层 / 热环)。"""
    k0 = max(sink, int(round(prompt_len * prompt_keep)))
    k0 = min(k0, prompt_len)
    hot = min(recent_window, decode_len)
    cooled = max(0, decode_len - hot)
    cold = int(round(cooled * cold_keep))
    return k0, cold, hot


def estimate_layout_bytes(inp: PlanInput, prompt_keep: float, prompt_bits: int,
                          recent_window: int, cold_keep: float, cold_bits: int) -> Tuple[int, int]:
    """估算某配置下的稳态逻辑字节, 返回 (总逻辑字节, 整段解码读流量代理)。

    读流量代理 = prompt 字节 * 读放大(T) + 冷/热字节 * 平均读放大(T/2)。
    """
    d = inp.dims
    k0, cold, hot = _steady_tokens(inp.prompt_len, inp.decode_len, prompt_keep,
                                   recent_window, cold_keep, inp.sink_tokens)
    prompt_bytes = quant.quantized_tier_bytes(
        k0, d.num_layers, d.num_kv_heads, d.head_dim, prompt_bits, prompt_bits,
        inp.group_size, inp.batch)
    cold_bytes = quant.quantized_tier_bytes(
        cold, d.num_layers, d.num_kv_heads, d.head_dim, cold_bits, cold_bits,
        inp.group_size, inp.batch)
    hot_bytes = quant.fp16_tier_bytes(hot, d.num_layers, d.num_kv_heads, d.head_dim, inp.batch)
    total = prompt_bytes + cold_bytes + hot_bytes

    t = float(max(1, inp.decode_len))
    read_proxy = prompt_bytes * t + (cold_bytes + hot_bytes) * (t / 2.0)
    return total, read_proxy


def plan(inp: PlanInput) -> PlanOutput:
    """在目标显存约束下搜索读放大最优的相位异构配置。"""
    total_tokens = inp.prompt_len + inp.decode_len
    full_bytes = full_fp16_kv_bytes(inp.dims, total_tokens, inp.batch)
    target_bytes = inp.target_memory_ratio * full_bytes

    candidates: List[Tuple[tuple, dict]] = []
    for pk in inp.prompt_keep_grid:
        for pb in inp.prompt_bits_grid:
            for rw in inp.recent_window_grid:
                for ck in inp.cold_keep_grid:
                    for cb in inp.cold_bits_grid:
                        total, read_proxy = estimate_layout_bytes(inp, pk, pb, rw, ck, cb)
                        feasible = total <= target_bytes
                        # 字典序评分 (idea 3.2, 见模块 docstring):
                        #   feasible 内: 主目标用满预算 => -total 最小 (total 最大); 次目标同
                        #   预算读流量最小 => read_proxy; 末位偏好读放大低的 cold 层高 keep/bit。
                        #   infeasible: 退化为最激进 => total 最小, 尽量逼近约束。
                        # feasible 组 (标志 0) 恒排在 infeasible 组 (标志 1) 之前, 故两组第二
                        # 键 (-total vs total) 语义不同亦不会跨组比较。
                        if feasible:
                            score = (0, -total, read_proxy, -ck, -cb)
                        else:
                            score = (1, total, read_proxy, -ck, -cb)
                        candidates.append((score, {
                            "prefill_keep_ratio": pk, "prompt_bits": pb,
                            "recent_window": rw, "decode_keep_ratio": ck,
                            "cold_bits": cb, "total": total, "read_proxy": read_proxy,
                            "feasible": feasible,
                        }))
    candidates.sort(key=lambda x: x[0])
    best = candidates[0][1]

    # 懒批 G: 在网格内取使 "冷压批量摊薄" 与 "热环不过度膨胀" 折中的中位值; 取与
    # recent_window 同量级但不超过它的最大候选, 保证每次冷批至少冷却一个完整批。
    g_choices = [g for g in inp.lazy_g_grid if g <= best["recent_window"]]
    lazy_g = max(g_choices) if g_choices else min(inp.lazy_g_grid)

    return PlanOutput(
        prefill_keep_ratio=best["prefill_keep_ratio"],
        prompt_bits=best["prompt_bits"],
        recent_window=best["recent_window"],
        decode_keep_ratio=best["decode_keep_ratio"],
        cold_bits=best["cold_bits"],
        lazy_g=lazy_g,
        sink_tokens=inp.sink_tokens,
        group_size=inp.group_size,
        est_steady_bytes=best["total"],
        full_fp16_bytes=full_bytes,
        est_memory_ratio=best["total"] / max(1, full_bytes),
        est_decode_read_bytes=best["read_proxy"],
        roofline=roofline_report(inp.prompt_len, inp.decode_len, inp.dims,
                                 inp.hardware, inp.batch),
        feasible=best["feasible"],
    )
