"""评估指标计算 (QA / CH-3 维护)。

涵盖三大类指标:
  - 内存:   KV Cache 逻辑字节、相对 full-cache 的显存节省比例、峰值显存。
  - 速度:   TTFT (首 token 延迟)、TPOT (每输出 token 延迟)、decode 吞吐 (tok/s)。
  - 精度:   困惑度 (perplexity) 聚合、token 级准确率、与基线输出的一致率。

所有函数均为纯计算, 不依赖具体模型, 可独立单测。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch


# --------------------------------------------------------------------------- #
# 内存指标
# --------------------------------------------------------------------------- #
def kv_cache_dtype_bytes(num_layers: int, num_kv_heads: int, head_dim: int,
                         seq_len: int, dtype_bytes: int = 2, batch: int = 1) -> int:
    """按张量维度解析计算 KV Cache 理论字节数 (K 和 V 各一份)。

    bytes = 2(K,V) * batch * num_layers * num_kv_heads * seq_len * head_dim * dtype_bytes
    """
    if min(num_layers, num_kv_heads, head_dim, seq_len, batch) < 0:
        raise ValueError("维度不能为负")
    return 2 * batch * num_layers * num_kv_heads * seq_len * head_dim * dtype_bytes


def memory_saving_ratio(compressed_bytes: int, full_bytes: int) -> float:
    """显存节省比例 = 1 - 压缩后 / full-cache; 取值 [0,1] (越大越省)。"""
    if full_bytes <= 0:
        raise ValueError("full_bytes 必须为正")
    return 1.0 - (compressed_bytes / full_bytes)


def compression_ratio(compressed_bytes: int, full_bytes: int) -> float:
    """压缩比 = full-cache / 压缩后 (>=1, 越大压得越狠)。"""
    if compressed_bytes <= 0:
        raise ValueError("compressed_bytes 必须为正")
    return full_bytes / compressed_bytes


# --------------------------------------------------------------------------- #
# 速度指标
# --------------------------------------------------------------------------- #
@dataclass
class LatencyStats:
    """一次生成过程的时延统计。"""

    ttft_s: float                      # 首 token 延迟 (= prefill 时间)
    decode_step_times_s: list = field(default_factory=list)  # 每个 decode step 耗时
    num_generated: int = 0             # 实际生成 token 数 (含首 token)

    @property
    def tpot_s(self) -> float:
        """每输出 token 平均延迟 (不含首 token, 即纯 decode 平均)。"""
        if not self.decode_step_times_s:
            return 0.0
        return sum(self.decode_step_times_s) / len(self.decode_step_times_s)

    @property
    def decode_throughput(self) -> float:
        """decode 阶段吞吐 (tok/s), 基于 decode steps 总耗时。"""
        total = sum(self.decode_step_times_s)
        if total <= 0:
            return 0.0
        return len(self.decode_step_times_s) / total

    @property
    def total_time_s(self) -> float:
        return self.ttft_s + sum(self.decode_step_times_s)

    @property
    def end_to_end_throughput(self) -> float:
        """端到端吞吐 (tok/s) = 生成 token 数 / 总时间 (含 prefill)。"""
        if self.total_time_s <= 0:
            return 0.0
        return self.num_generated / self.total_time_s

    def percentile_decode_ms(self, pct: float) -> float:
        """decode step 时延的分位数 (毫秒)。"""
        if not self.decode_step_times_s:
            return 0.0
        if not (0 <= pct <= 100):
            raise ValueError("pct 必须在 [0,100]")
        ordered = sorted(self.decode_step_times_s)
        k = (len(ordered) - 1) * (pct / 100.0)
        low = math.floor(k)
        high = math.ceil(k)
        if low == high:
            return ordered[int(k)] * 1000.0
        frac = k - low
        return (ordered[low] * (1 - frac) + ordered[high] * frac) * 1000.0


# --------------------------------------------------------------------------- #
# 精度指标
# --------------------------------------------------------------------------- #
@dataclass
class PerplexityAccumulator:
    """流式困惑度累加器: 跨 chunk 累计负对数似然 (NLL) 与 token 数。"""

    total_nll: float = 0.0
    total_tokens: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor,
               ignore_index: int = -100) -> None:
        """累加一段的 NLL。

        Args:
            logits: 形状 ``[..., seq, vocab]`` 的预测分布 (已对齐到对应 labels)。
            labels: 形状 ``[..., seq]`` 的目标 token id。
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(f"logits/labels 形状不匹配: {logits.shape} vs {labels.shape}")
        logits = logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        flat_logp = log_probs.reshape(-1, log_probs.shape[-1])
        flat_labels = labels.reshape(-1)
        mask = flat_labels != ignore_index
        if mask.sum() == 0:
            return
        valid_labels = flat_labels[mask]
        token_logp = flat_logp[mask].gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)
        self.total_nll += float((-token_logp).sum().item())
        self.total_tokens += int(mask.sum().item())

    @property
    def perplexity(self) -> float:
        if self.total_tokens == 0:
            return float("nan")
        return math.exp(self.total_nll / self.total_tokens)

    @property
    def mean_nll(self) -> float:
        if self.total_tokens == 0:
            return float("nan")
        return self.total_nll / self.total_tokens


def token_match_rate(pred_ids: Sequence[int], ref_ids: Sequence[int]) -> float:
    """逐位置 token 一致率 (用于"压缩 vs full-cache 同 prompt 输出"的偏移度量)。"""
    n = min(len(pred_ids), len(ref_ids))
    if n == 0:
        return float("nan")
    matches = sum(1 for i in range(n) if pred_ids[i] == ref_ids[i])
    return matches / n


def first_divergence_index(pred_ids: Sequence[int], ref_ids: Sequence[int]) -> Optional[int]:
    """返回与参考输出首次不一致的位置下标; 完全一致返回 None。"""
    n = min(len(pred_ids), len(ref_ids))
    for i in range(n):
        if pred_ids[i] != ref_ids[i]:
            return i
    if len(pred_ids) != len(ref_ids):
        return n
    return None


def exact_substring_score(prediction: str, answers: Sequence[str]) -> float:
    """needle/QA 类的子串命中: 任一标准答案是预测的子串则记 1, 否则 0。"""
    pred = prediction.strip().lower()
    for ans in answers:
        if ans.strip().lower() in pred:
            return 1.0
    return 0.0


# --------------------------------------------------------------------------- #
# 显存测量 (依赖 CUDA, 但对 CPU 退化为 0 以便测试)
# --------------------------------------------------------------------------- #
def peak_gpu_memory_bytes(device: Optional[torch.device] = None) -> int:
    """返回自上次 reset 以来的峰值已分配显存 (字节)。"""
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def reset_peak_gpu_memory(device: Optional[torch.device] = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
