"""真实打包的 PhaseKV 缓存，供 Qwen2 的自定义 decode attention 使用。

布局按层保存为 [只读 prompt] + [冻结的 cold 批次 ...] + [FP16 hot]。
K 沿 token 轴做 per-channel 分组量化；V 沿 head_dim 轴做 per-token 分组量化。
标准 Hugging Face attention 的 ``Cache.update`` 要求返回完整的稠密 K/V，
这与压缩域执行冲突。因此 Qwen2 attention 适配器必须调用 ``append_hot``，
再直接消费 ``layer_tiers``；普通 ``update`` 会明确报错。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import torch
from transformers.cache_utils import Cache

from .kernels.packing import PackedTensor, quantize_pack


_PACKED_BITS = (2, 4, 8)


@dataclass
class PackedKVSegment:#表示一段已经压缩打包的 KV
    key: PackedTensor
    value: PackedTensor

    @property
    def length(self) -> int:
        return self.key.orig_shape[2]#返回原始长度，即打包前的 token 数量，seq_len

    @property
    def nbytes(self) -> int:#返回该段 KV 占用的物理字节数，包括打包数据和 scale/min
        return self.key.nbytes + self.value.nbytes


@dataclass
#hot_k 和 hot_v 是提前分配好的四维张量，形状是 [Batch, Head, capacity, dim]。其中 capacity 是总容量，但只有前 hot_len 个 token 位置已经写入有效数据。
class LayerTiers:#表示模型某一层的完整缓存。一个 LayerTiers 对象只对应模型的一层。它里面的 cold 列表装的是这一层中先后冻结的多个 KV 段，不是多个模型层
    prompt: PackedKVSegment
    cold: list[PackedKVSegment] = field(default_factory=list)#保存旧 token 的压缩段列表，初始为空。default_factory=list 保证每个 LayerTiers 对象都有自己的列表。
    hot_k: Optional[torch.Tensor] = None  # 预分配 [B, Hkv, capacity, hd]
    hot_v: Optional[torch.Tensor] = None
    hot_len: int = 0

    @property
    def cold_len(self) -> int:
        return sum(segment.length for segment in self.cold)#取出cold段列表中每个 PackedKVSegment 的长度，暂时命名为segment，并将它们相加，得到该层冷段的总长度。

    @property
    def physical_len(self) -> int:
        return self.prompt.length + self.cold_len + self.hot_len

    def hot_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        """仅返回已写入部分；未初始化的预留空间绝不可参与注意力。"""
        return self.hot_k[:, :, : self.hot_len, :], self.hot_v[:, :, : self.hot_len, :]


class PhaseKVCache(Cache):
    """保留真实低比特字节的 Cache；首版支持单 token decode。

    ``get_seq_length`` 返回物理保留长度，``seen_tokens`` 返回真实绝对长度。
    Qwen2 前向应显式传入真实 ``position_ids`` 和物理 ``cache_position``。
    本类不供未改造的 Qwen2Attention 或 ``cache_implementation='static'`` 使用。
    """

    def __init__(
        self,
        layers: list[LayerTiers],
        *,
        seen_tokens: int,
        prompt_bits: int,
        cold_bits: int,
        group_size: int,
        recent_window: int,
        lazy_g: int,
        decode_keep_ratio: float,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("PhaseKVCache 至少需要一层 KV")
        if prompt_bits not in _PACKED_BITS or cold_bits not in _PACKED_BITS:
            raise ValueError(f"真实打包目前只支持 {_PACKED_BITS} bit")
        if group_size <= 0 or recent_window <= 0 or lazy_g <= 0:
            raise ValueError("group_size、recent_window、lazy_g 必须为正整数")
        if not 0.0 < decode_keep_ratio <= 1.0:
            raise ValueError("decode_keep_ratio 必须落在 (0, 1]")
        self.layers = layers
        self._seen_tokens = int(seen_tokens)
        self.prompt_bits = prompt_bits
        self.cold_bits = cold_bits
        self.group_size = group_size
        self.recent_window = recent_window
        self.lazy_g = lazy_g
        self.decode_keep_ratio = decode_keep_ratio
        self._pending_layers: set[int] = set()

    @classmethod
    @torch.no_grad()
    def from_dense_prefill(
        cls,
        dense_cache,
        *,
        prompt_len: int,
        keep_indices: Optional[Sequence[torch.Tensor]] = None,#本身是一个列表，里面存着要保留的 prompt token 的索引列表，按层提供。每个索引张量的形状为 [B, Hkv, K0]，其中 K0 是要保留的 token 数量，表示最后一维有 k0个格子。格子里存的值，例如 0、2、5、19，才是原 prompt 的索引
        prompt_bits: int = 2,
        cold_bits: int = 4,
        group_size: int = 32,
        recent_window: int = 32,
        lazy_g: int = 16,
        decode_keep_ratio: float = 1.0,
    ) -> "PhaseKVCache":
        """从原始 prefill K/V 直接选择并打包，不经过 fake-quant。

        ``keep_indices[layer]`` 为 [B,Hkv,K0] 的原始 prompt 位置，可逐 head 不同。
        所有层须有相同 K0；调用方随后应释放 dense_cache 与 prefill 输出引用。
        """
        if prompt_len <= 0:
            raise ValueError("prompt_len 必须为正")
        if prompt_bits not in _PACKED_BITS or cold_bits not in _PACKED_BITS:
            raise ValueError(f"真实打包目前只支持 {_PACKED_BITS} bit")
        if group_size <= 0 or recent_window <= 0 or lazy_g <= 0:
            raise ValueError("group_size、recent_window、lazy_g 必须为正整数")
        if not 0.0 < decode_keep_ratio <= 1.0:
            raise ValueError("decode_keep_ratio 必须落在 (0, 1]")
        keys, values = dense_cache.key_cache, dense_cache.value_cache
        if len(keys) == 0 or len(keys) != len(values):
            raise ValueError("prefill 缓存缺少完整 K/V 层")
        if keep_indices is not None and len(keep_indices) != len(keys):
            raise ValueError("keep_indices 的层数与缓存不一致")

        capacity = recent_window + lazy_g#稳定区。当热区超过这个长度时，最老的 token 会被冷却到 cold 层
        layers: list[LayerTiers] = []
        retained_length: Optional[int] = None
        for layer_idx, (key, value) in enumerate(zip(keys, values)):#zip(keys, values) 把同一模型层的 K、V 配成一对；enumerate 再给这对数据一个层编号 layer_idx
            if key is None or value is None or key.shape != value.shape or key.ndim != 4:
                raise ValueError(f"第 {layer_idx} 层 K/V 必须同形状 [B,Hkv,S,hd]")
            if key.dtype not in (torch.float16, torch.bfloat16) or value.dtype != key.dtype:
                raise ValueError("prefill K/V 必须同为 float16 或 bfloat16")
            if key.shape[2] != prompt_len:
                raise ValueError(f"第 {layer_idx} 层长度与 prompt_len 不一致")
            if keep_indices is not None:
                idx = keep_indices[layer_idx].to(device=key.device, dtype=torch.long)
                if idx.ndim != 3 or idx.shape[:2] != key.shape[:2] or idx.shape[2] == 0:
                    raise ValueError(f"第 {layer_idx} 层 keep_indices 形状无效")
                idx = idx.sort(dim=-1).values
                if bool((idx < 0).any()) or bool((idx >= prompt_len).any()):
                    raise ValueError("keep_indices 超出 prompt 范围")
                if bool((idx[..., 1:] == idx[..., :-1]).any()):
                    raise ValueError("keep_indices 含重复位置")
                gather_idx = idx.unsqueeze(-1).expand(*idx.shape, key.shape[-1])
                key = key.gather(2, gather_idx)
                value = value.gather(2, gather_idx)
            # 模型可以用 BF16 保持 prefill 数值稳定；压缩域 decode 当前使用 FP16。
            key = key.to(torch.float16)
            value = value.to(torch.float16)
            if retained_length is None:
                retained_length = key.shape[2]
            elif key.shape[2] != retained_length:
                raise ValueError("各层必须保留相同数量的 prompt token")

            segment = PackedKVSegment(
                quantize_pack(key, prompt_bits, axis=-2, group_size=group_size),
                quantize_pack(value, prompt_bits, axis=-1, group_size=group_size),
            )
            shape = (key.shape[0], key.shape[1], capacity, key.shape[3])
            #分配k、v的热层空间，等待写入。形状是 [B, Hkv, capacity, hd]，其中 capacity 是总容量，但只有前 hot_len 个 token 位置已经写入有效数据
            layers.append(LayerTiers(
                prompt=segment,
                hot_k=torch.empty(shape, dtype=torch.float16, device=key.device),#预分配
                hot_v=torch.empty(shape, dtype=torch.float16, device=key.device),
            ))

        return cls(
            layers, seen_tokens=prompt_len, prompt_bits=prompt_bits,
            cold_bits=cold_bits, group_size=group_size,
            recent_window=recent_window, lazy_g=lazy_g,
            decode_keep_ratio=decode_keep_ratio,
        )

    def __len__(self) -> int:
        return len(self.layers)

    def layer_tiers(self, layer_idx: int) -> LayerTiers:
        return self.layers[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.layers[0 if layer_idx is None else layer_idx].physical_len

    def get_max_cache_shape(self) -> None:
        return None

    @property
    def true_seq_length(self) -> int:
        """下一 token 应使用的绝对 RoPE 位置。"""
        return self._seen_tokens

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        raise RuntimeError(
            "PhaseKVCache 不返回稠密 K/V；请在 Qwen2 自定义 attention 中使用 "
            "append_hot() + layer_tiers()"
        )

    @torch.no_grad()
    def append_hot(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """在该层追加一个已施加 RoPE 的 K 和对应 V，供当前 query 立即读取。"""
        if layer_idx in self._pending_layers:
            raise RuntimeError(f"第 {layer_idx} 层在当前 decode 步已追加")
        layer = self.layers[layer_idx]
        expected = (*layer.hot_k.shape[:2], 1, layer.hot_k.shape[-1])
        if key.shape != expected or value.shape != expected:
            raise ValueError(f"decode K/V 期望形状 {expected}")
        if key.dtype != torch.float16 or value.dtype != torch.float16:
            raise ValueError("热层目前只支持 float16")
        if key.device != layer.hot_k.device or value.device != layer.hot_v.device:
            raise ValueError("decode K/V 与该层缓存必须在同一设备")
        if layer.hot_len >= layer.hot_k.shape[2]:
            raise RuntimeError("热层空间已满；上一 decode 步未正确结束")
        layer.hot_k[:, :, layer.hot_len : layer.hot_len + 1, :].copy_(key)#输出是逐token生成，所以每次只追加一个 token 的 K/V
        layer.hot_v[:, :, layer.hot_len : layer.hot_len + 1, :].copy_(value)
        layer.hot_len += 1
        self._pending_layers.add(layer_idx)#记录

#举个例子
#hot 区（处理前共 48 个）
#[ 最旧的 16 个 | 最近的 32 个 ]
 #  aged          remaining
    @torch.no_grad()
    def finish_decode_step(self) -> None:
        """所有层完成注意力后，按懒批规则把过期热 token 冻结为冷段。"""
        if self._pending_layers != set(range(len(self.layers))):
            raise RuntimeError("必须先让每一层各追加一个 decode token")
        for layer in self.layers:
            aged = layer.hot_len - self.recent_window
            if aged < self.lazy_g:
                continue
            key = layer.hot_k[:, :, :aged, :]
            value = layer.hot_v[:, :, :aged, :]
            keep_count = max(1, round(self.decode_keep_ratio * aged))#只保留一部分token
            if keep_count < aged:
                importance = torch.linalg.vector_norm(value.float(), dim=-1)#计算L2范数
                indices = importance.topk(keep_count, dim=-1).indices.sort(dim=-1).values
                gather_idx = indices.unsqueeze(-1).expand(*indices.shape, key.shape[-1])#gather(dim, index) 的意思是：按照 index 中存放的位置，在指定维度取值
                key = key.gather(2, gather_idx)
                value = value.gather(2, gather_idx)
            layer.cold.append(PackedKVSegment(
                quantize_pack(key, self.cold_bits, axis=-2, group_size=self.group_size),
                quantize_pack(value, self.cold_bits, axis=-1, group_size=self.group_size),
            ))
            remaining = layer.hot_len - aged #保留的token
            if remaining:
                layer.hot_k[:, :, :remaining, :].copy_(layer.hot_k[:, :, aged : layer.hot_len, :].clone())
                layer.hot_v[:, :, :remaining, :].copy_(layer.hot_v[:, :, aged : layer.hot_len, :].clone())
            layer.hot_len = remaining
        lengths = {layer.physical_len for layer in self.layers}
        if len(lengths) != 1:
            raise RuntimeError("各层冷却后物理长度不一致")
        self._seen_tokens += 1
        self._pending_layers.clear()

    def physical_nbytes(self) -> int:
        """常驻 KV 张量实际分配的字节数，包含热层预留空间。"""
        total = 0
        for layer in self.layers:
            total += layer.prompt.nbytes
            total += sum(segment.nbytes for segment in layer.cold)
            total += layer.hot_k.numel() * layer.hot_k.element_size()
            total += layer.hot_v.numel() * layer.hot_v.element_size()
        return total

    @torch.no_grad()
    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """供后续 beam search 集成；仅允许在两个 decode 步之间调用。"""
        if self._pending_layers:
            raise RuntimeError("decode 步尚未结束，不能重排缓存")

        def reorder_packed(packed: PackedTensor) -> PackedTensor:
            index = beam_idx.to(packed.packed.device)
            return replace(
                packed,
                packed=packed.packed.index_select(0, index),
                scale=packed.scale.index_select(0, index),
                mins=packed.mins.index_select(0, index),
                orig_shape=(index.numel(), *packed.orig_shape[1:]),
            )

        for layer in self.layers:
            layer.prompt = PackedKVSegment(
                reorder_packed(layer.prompt.key), reorder_packed(layer.prompt.value))
            layer.cold = [PackedKVSegment(
                reorder_packed(segment.key), reorder_packed(segment.value))
                for segment in layer.cold]
            layer.hot_k = layer.hot_k.index_select(0, beam_idx.to(layer.hot_k.device))
            layer.hot_v = layer.hot_v.index_select(0, beam_idx.to(layer.hot_v.device))
