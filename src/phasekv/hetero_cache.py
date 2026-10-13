"""PhaseKV 三层异构缓存：状态管理 + 底层算子 (CH-2 架构)。

idea 文档第 3.1 节把 KV Cache 按"生命周期/访问模式"分三层, 每层不同表示与布局:

  1. Prompt 层 (frozen, 只读): prefill 一次成型——per-head 注意力 heavy-hitter 选择 +
     KIVI 式低比特打包 (prompt_bits, 最狠), 成型后不再改写 (读放大 = T 的对象)。
  2. Decode 热层 (hot, FP16 环形窗口): 最近 recent_window 个生成 token, 全精度追加。
  3. Decode 冷层 (cooled, 量化+驱逐): 滑出热窗口的较老生成 token, 每 G 步懒批一次
     "廉价信号 (value-norm) 选择 + 低比特量化 (cold_bits)", 永不触碰 Prompt 层。

阶段 A (本模块) 操作标准 transformers ``DynamicCache``: 用 fake-quant (量化-反量化)
让模型据低比特栅格做*真实前向*得到*真实精度*, 物理张量仍是 fp16, 逻辑显存由三层位宽
解析折算 (反映压缩后真实占用)。真正的低比特打包存储 + 单遍混精融合 kernel 是阶段 B。

物理布局 (单层 [b, Hkv, seq, hd], 时间升序):
    [ 0 .. K0 )                prompt 层 (per-head 各自选中的 prompt token, 已量化, 冻结)
    [ K0 .. K0+cold_len )       decode 冷层 (per-head 各自保留的老生成 token, 已量化)
    [ K0+cold_len .. seq )      decode 热层 (最近生成 token, FP16)

不同 head 在 prompt/冷层可保留不同 token *下标*, 但同层各 head 保留相同 *数量*
(物理张量保持矩形)。保留下标按时间升序, baked-in RoPE 对随后新 query 的相对距离精确
(新 query 由 runtime 以真实绝对位置施加 RoPE)。本模块不做任何 dense 回退。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from . import quant


# --------------------------------------------------------------------------- #
# Cache 形状探针
# --------------------------------------------------------------------------- #
def cache_num_layers(cache) -> int:
    return len(cache.key_cache)


def cache_seq_len(cache, layer_idx: int = 0) -> int:
    if len(cache.key_cache) <= layer_idx or cache.key_cache[layer_idx] is None:
        return 0
    return cache.key_cache[layer_idx].shape[-2]


def num_kv_heads_of(model_config) -> int:
    return getattr(model_config, "num_key_value_heads",getattr(model_config, "num_attention_heads"))


# --------------------------------------------------------------------------- #
# GQA 聚合 / 重要度信号
# --------------------------------------------------------------------------- #
def aggregate_to_kv_heads(scores: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """把按 query head 的分数 [b, Hq, ...] 聚合到 kv head [b, Hkv, ...] (GQA 组内求和)。"""
    b, hq = scores.shape[0], scores.shape[1]
    if hq == num_kv_heads:
        return scores
    if hq % num_kv_heads != 0:
        raise ValueError(f"query head 数 {hq} 不是 kv head 数 {num_kv_heads} 的整数倍")
    group = hq // num_kv_heads
    tail = scores.shape[2:]
    return scores.view(b, num_kv_heads, group, *tail).sum(dim=2)


def observation_votes(attn_layer: torch.Tensor, num_kv_heads: int,
                      obs_window: int) -> torch.Tensor:
    """prefill 观察窗口投票 (洞见 3: prefill 算注意力本就免费, 用它做选择)。

    用 prompt 末尾 ``obs_window`` 个 query 行对各 key 位置的注意力求和, 估计哪些 prompt
    token 在生成时最受关注; 再 GQA 聚合到 kv head。返回 [b, Hkv, kv_len]。
    """
    q_len = attn_layer.shape[-2]
    w = max(1, min(obs_window, q_len))
    votes = attn_layer[:, :, -w:, :].float().sum(dim=2)  # [b, Hq, kv_len]
    return aggregate_to_kv_heads(votes, num_kv_heads)


def pool_scores(scores: torch.Tensor, kernel: Optional[int],
                mode: str = "maxpool") -> torch.Tensor:
    """对重要度分数沿序列轴做 1D 池化, 使被选 token 形成连续簇 (保留局部上下文)。"""
    if kernel is None or kernel <= 1:
        return scores
    b, h, length = scores.shape
    x = scores.reshape(b * h, 1, length)
    pad = kernel // 2  #求出0填充数
    if mode == "maxpool":
        pooled = F.max_pool1d(x, kernel_size=kernel, stride=1, padding=pad)#kernel即为滑动窗口大小，每个元素值都选窗口内的最大值，这样注意力最大的token附近的token分数也会变大，保留局部上下文
    elif mode == "avgpool":
        pooled = F.avg_pool1d(x, kernel_size=kernel, stride=1, padding=pad)#只选取连续一段都重要的片段
    else:
        raise ValueError(f"未知 pooling: {mode}, 仅支持 maxpool/avgpool")
    return pooled[..., :length].reshape(b, h, length)


#只选取token_slice片段做L2范数计算，如token_slice = slice(20, 40)
def value_norm_signal(value_layer: torch.Tensor, token_slice: slice) -> torch.Tensor:
    """decode 冷层的*廉价*重要度信号 (洞见 3: decode 取注意力分数昂贵, 改用廉价信号)。

    用 Value 向量的 L2 范数作为 token 重要度的廉价代理: 直接对 ``value_cache`` 的
    head_dim 轴取范数, 无需 materialize 注意力分数、不破坏 Flash/SDPA。

    Args:
        value_layer: 单层 V cache [b, Hkv, seq, hd]。
        token_slice: 取重要度的 token 区间 (冷候选区)。

    Returns:
        [b, Hkv, len(token_slice)] 的 per-(head, token) 重要度。
    """
    seg = value_layer[:, :, token_slice, :].float()
    return torch.linalg.vector_norm(seg, ord=2, dim=-1)


# --------------------------------------------------------------------------- #
# 逐 head eviction
# --------------------------------------------------------------------------- #
#本身不判断哪些 token 重要，只负责执行已经生成好的 keep_indices
#budget 是每个 head 最终保留的 token 数
def evict_per_head(cache, keep_indices: Sequence[Optional[torch.Tensor]]):
    """按逐头下标保留 KV (eviction)。

    Args:
        cache: transformers Cache。
        keep_indices: 长度为层数的列表; 第 l 项为 [b, num_kv_heads, budget_l] 的
            LongTensor (同层内各 head 的 budget 必须一致以保持矩形; 层间可不同);
            为 None 的层不改动。每个 head 的下标会被升序排序以保持时间顺序。

    Returns:
        原 cache (就地修改)。
    """
    for layer in range(cache_num_layers(cache)):
        idx = keep_indices[layer]#idx: [b, Hkv, budget]，一层内的所有 head 必须保留相同数量
        if idx is None:
            continue
        k = cache.key_cache[layer]#将该层的cache提取出来
        v = cache.value_cache[layer]
        idx = idx.to(device=k.device, dtype=torch.long)
        idx, _ = torch.sort(idx, dim=-1)#原本的下标按TOP-K分数排列，现在改成按下标索引排列
        head_dim = k.shape[-1]
        gather_idx = idx.unsqueeze(-1).expand(idx.shape[0], idx.shape[1], idx.shape[2], head_dim)#增加一个维度，并将最后一个维度扩展成head_dim
        cache.key_cache[layer] = k.gather(-2, gather_idx).contiguous()
        cache.value_cache[layer] = v.gather(-2, gather_idx).contiguous()
    return cache


def topk_per_head(signal: torch.Tensor, select_k: int) -> torch.Tensor:
    """按重要度 [b, Hkv, n] 逐 head 取 top-``select_k`` 的局部下标 (相对区间起点)。

    Returns:
        [b, Hkv, min(select_k, n)] 的 LongTensor (未排序; evict 时再排)。
    """
    n = signal.shape[-1]
    k = max(0, min(select_k, n))
    if k == 0:
        return signal.new_zeros((signal.shape[0], signal.shape[1], 0), dtype=torch.long)
    return torch.topk(signal, k=k, dim=-1).indices.long()


# --------------------------------------------------------------------------- #
# 三层状态 + 逻辑字节折算
# --------------------------------------------------------------------------- #
#不直接存储 K/V 张量，而是记录如何解释 cache 中的 K/V
@dataclass
class HeteroTierState:
    """跟踪三层异构缓存在物理 cache 上的边界与量化位宽 (单条样本生命周期内)。

    prompt_len_phys : prompt 层的物理 token 数 K0 (每 head 相同数量, 已量化, 冻结)。
    cold_len        : decode 冷层的物理 token 数 (每 head 相同数量, 已量化)。
    prompt_bits     : prompt 层量化位宽 (洞见 2: prompt 读放大大, 最狠压)。
    cold_bits       : decode 冷层量化位宽 (读放大约 T/2, 压得比 prompt 温和)。
    group_size      : KIVI 分组粒度。
    recent_window   : decode 热层 FP16 窗口长度。
    """

    prompt_len_phys: int = 0#Prompt 层经过筛选后，实际保留的 token 数-
    cold_len: int = 0#Decode 冷层当前保存的 token 数
    prompt_bits: int = 2#Prompt 层的逻辑量化位宽
    cold_bits: int = 4#Decode 冷层的逻辑量化位宽
    group_size: int = 32#量化时每组包含多少个元素
    recent_window: int = 32#Decode 热层计划保存的最近 token 数

    def reset(self) -> None:
        self.prompt_len_phys = 0
        self.cold_len = 0

    def hot_len(self, cache) -> int:
        return max(0, cache_seq_len(cache) - self.prompt_len_phys - self.cold_len)

    def logical_bytes(self, cache) -> int:
        """三层逻辑显存字节: prompt(prompt_bits) + cold(cold_bits) + hot(fp16)。"""
        if cache_num_layers(cache) == 0 or cache.key_cache[0] is None:
            return 0
        b, hkv, seq, head_dim = cache.key_cache[0].shape
        layers = cache_num_layers(cache)
        k0 = max(0, min(self.prompt_len_phys, seq))
        cold = max(0, min(self.cold_len, seq - k0))
        hot = max(0, seq - k0 - cold)
        total = 0
        total += quant.quantized_tier_bytes(
            k0, layers, hkv, head_dim, self.prompt_bits, self.prompt_bits,
            self.group_size, b)
        total += quant.quantized_tier_bytes(
            cold, layers, hkv, head_dim, self.cold_bits, self.cold_bits,
            self.group_size, b)
        total += quant.fp16_tier_bytes(hot, layers, hkv, head_dim, b)
        return total
