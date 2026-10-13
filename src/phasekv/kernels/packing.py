"""PhaseKV 阶段 B 数据地基: 真实低比特打包 (CH-2 架构)。

阶段 A (``phasekv.quant``) 用 fake-quant——量化后立即反量化回 fp16, 物理张量仍是 fp16,
逻辑显存靠位宽解析折算。阶段 B 要兑现"带宽受限 decode 的真加速", prompt/冷层必须以*真实
低比特打包*存于 HBM, 单遍融合 kernel 在片上反量化, 从而真正减少每步 decode 的 HBM 读字节。

本模块是该路线的地基, 与 ``quant`` 的 KIVI 式分组非对称量化*数值等价*(由自检保证):

  - ``pack_low_bits`` / ``unpack_low_bits``: 无损 sub-byte 打包原语, 沿最后一维把 2/4/8 bit
    的整数码紧密塞入 uint8 (2bit=>4 值/字节, 4bit=>2, 8bit=>1)。
  - ``quantize_pack`` / ``unpack_dequantize``: 沿指定轴分组做 min-max 非对称量化, 产出
    ``PackedTensor``(打包码 + 紧凑 per-group scale/min); 反量化结果与
    ``quant._grouped_affine_quant_dequant`` 逐值一致 (公式同源)。
  - ``PackedTensor.nbytes``: 真实物理字节 (打包数据位 + fp16 scale/min), 与
    ``quant.quantized_tier_bytes`` 的逻辑折算口径一致, 使"阶段 A 逻辑显存 == 阶段 B 物理显存"。

不做任何 dense 回退; 打包只改变存储密度, 不改变数值 (无损于量化栅格)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

_BITS_PER_BYTE = 8
_PACKABLE_BITS = (2, 4, 8)
#: 每个量化组的紧凑元数据字节 (scale + min, 各按 fp16=2 字节), 与 quant.SCALE_ZERO_BYTES 对齐。
SCALE_MIN_BYTES = 4


def _vals_per_byte(bits: int) -> int:#返回一个字节能放几个码
    if bits not in _PACKABLE_BITS:
        raise ValueError(f"打包位宽仅支持 {_PACKABLE_BITS}, 实际 {bits}")
    return _BITS_PER_BYTE // bits


# --------------------------------------------------------------------------- #
# 无损 sub-byte 打包原语 (沿最后一维)
# --------------------------------------------------------------------------- #
def pack_low_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:#把的一个码装进字节
    """沿最后一维把 ``bits`` 位整数码紧密打包进 uint8。

    Args:
        codes: uint8 张量, 值域须落在 ``[0, 2**bits)``。
        bits: 2/4/8。8 时无需打包, 原样返回。

    Returns:
        uint8 张量, 最后一维长度 = ceil(L / (8//bits)); 尾部不足一字节处补 0 占位。
    """
    if codes.dtype != torch.uint8:
        raise ValueError(f"codes 必须为 uint8, 实际 {codes.dtype}")
    vpb = _vals_per_byte(bits)
    if bits == _BITS_PER_BYTE:
        return codes.contiguous()
    length = codes.shape[-1]
    pad = (-length) % vpb
    if pad:
        codes = torch.cat([codes, codes.new_zeros(*codes.shape[:-1], pad)], dim=-1)
    n_bytes = codes.shape[-1] // vpb#计算打包后的字节数
    grouped = codes.reshape(*codes.shape[:-1], n_bytes, vpb).to(torch.int32)#将codes重新调整形状，使其最后两维分别表示字节数和每个字节中的值数
    shifts = torch.arange(vpb, device=codes.device, dtype=torch.int32) * bits#生成从 0 到 vpb-1 的整数，不包含 vpb
    packed = (grouped << shifts).sum(dim=-1)
    return packed.to(torch.uint8).contiguous()


def unpack_low_bits(packed: torch.Tensor, bits: int, length: int) -> torch.Tensor:
    """``pack_low_bits`` 的逆: 从 uint8 还原前 ``length`` 个 ``bits`` 位整数码。"""
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed 必须为 uint8, 实际 {packed.dtype}")
    vpb = _vals_per_byte(bits)
    if bits == _BITS_PER_BYTE:
        return packed[..., :length].contiguous()
    mask = (1 << bits) - 1
    p = packed.to(torch.int32).unsqueeze(-1)
    shifts = torch.arange(vpb, device=packed.device, dtype=torch.int32) * bits
    vals = (p >> shifts) & mask
    vals = vals.reshape(*packed.shape[:-1], packed.shape[-1] * vpb)
    return vals[..., :length].to(torch.uint8).contiguous()


# --------------------------------------------------------------------------- #
# 分组量化 <-> 打包码 (与 quant 数值等价)
# --------------------------------------------------------------------------- #
@dataclass
class PackedTensor:
    """一段 K/V 的真实低比特打包表示 (阶段 B 的 HBM 驻留形态)。

    packed     : uint8, 打包码 (沿量化轴移到末维后打包)。
    scale, mins: fp16, per-group 量化参数 (形状 ``[..., n_groups, 1]``, 量化轴在末维)。
    bits       : 量化/打包位宽。
    axis       : 原张量上的量化轴 (已规约为非负)。
    group_size : 分组粒度 (受量化轴长度截断)。
    orig_shape : 原张量形状 (量化轴未移动前)。
    axis_len   : 量化轴的真实长度 (打包含尾部补齐, 反量化按此截断)。
    """

    packed: torch.Tensor
    scale: torch.Tensor
    mins: torch.Tensor
    bits: int
    axis: int
    group_size: int
    orig_shape: Tuple[int, ...]
    axis_len: int

    @property
    def nbytes(self) -> int:
        """真实物理字节: 打包数据 + fp16 scale/min。"""
        data = self.packed.numel()  # uint8, 已是紧密打包字节数，numel返回张量里面的元素数
        meta = self.scale.numel() + self.mins.numel()  # 每个 fp16 = 2 字节
        return int(data + meta * 2)


def quantize_pack(x: torch.Tensor, bits: int, axis: int, group_size: int) -> PackedTensor:
    """沿 ``axis`` 分组做 min-max 非对称量化, 并把整数码真实打包 (无 dequant 回 fp16)。

    量化公式与 ``quant._grouped_affine_quant_dequant`` 同源 (scale=(max-min)/qmax,
    q=round((x-min)/scale)), 因此 ``unpack_dequantize`` 的结果与阶段 A fake-quant 逐值一致。
    """
    if bits not in _PACKABLE_BITS:
        raise ValueError(f"bits 仅支持 {_PACKABLE_BITS}, 实际 {bits}")
    qmax = (1 << bits) - 1
    orig_shape = tuple(x.shape)#把张量的形状转为元组，并保存起来。这样可以在后续的操作中使用这个元组来获取张量的维度信息，而不需要每次都调用 x.shape。
    axis = axis % x.dim()#axis对x.  dim()取模，确保axis在合法范围内。x.dim()返回张量的维度数，axis % x.dim()可以将负数的轴索引转换为正数索引。
    #例如，如果x是一个三维张量，x.dim()返回3，那么axis可以是0、1或2。如果axis是-1，它将被转换为2（最后一个轴）。如果axis是-2，它将被转换为1（倒数第二个轴），以此类推。

    xf = x.float().movedim(axis, -1).contiguous()#k一般沿着seq方向量化，v沿dim方向量化【batch，head，seq_len，dim】
    moved_shape = xf.shape
    length = moved_shape[-1]
    gs = length if group_size is None or group_size <= 0 else min(group_size, length)

    pad = (-length) % gs#pad的值需要为正数
    if pad:
        tail = xf[..., -1:].expand(*xf.shape[:-1], pad)#将xf的最后一个元素沿着最后一维扩展pad个元素，形成一个新的张量tail。这个操作确保了在量化过程中，数据的长度是gs的整数倍，从而避免了分组时出现不完整的组。
        xf = torch.cat([xf, tail], dim=-1)#拼接
    padded = xf.shape[-1]
    n_groups = padded // gs#取得分组数

    xg = xf.reshape(*xf.shape[:-1], n_groups, gs)#将张量xf重新调整形状，使其最后两维分别表示分组数和每组的大小。具体来说，xg的形状为(*xf.shape[:-1], n_groups, gs)，其中*xf.shape[:-1]表示除了最后一维之外的所有维度，n_groups表示分组数，gs表示每组的大小。这样做的目的是为了方便后续对每个分组进行量化操作。
    g_min = xg.amin(dim=-1, keepdim=True)#计算每个分组的最小值
    g_max = xg.amax(dim=-1, keepdim=True)#计算每个分组的最大值
    scale = (g_max - g_min).clamp_(min=1e-8) / qmax#计算每个分组的量化比例因子（scale），用于将浮点数映射到整数码。scale的计算公式为：(g_max - g_min) / qmax，其中g_max和g_min分别表示每个分组的最大值和最小值，qmax表示量化码的最大值（即2^bits - 1）。为了避免除以零的情况，使用clamp_函数将scale的最小值限制为1e-8。
    codes_g = torch.round((xg - g_min) / scale).clamp_(0, qmax).to(torch.uint8)#将每个分组的浮点数映射到整数码，并将结果限制在[0, qmax]范围内。
    # 截去分组补齐位: 只打包真实 token 的码, 使物理字节 == quant 的逻辑折算 (不含 padding)。
    codes = codes_g.reshape(*moved_shape[:-1], padded)[..., :length].contiguous()#将分组后的整数码重新调整形状，使其与原始张量的形状一致，并截去分组补齐位，只保留真实 token 的码。

    packed = pack_low_bits(codes, bits)
    return PackedTensor(
        packed=packed,
        scale=scale.squeeze(-1).to(torch.float16).contiguous(),
        mins=g_min.squeeze(-1).to(torch.float16).contiguous(),
        bits=bits, axis=axis, group_size=gs,
        orig_shape=orig_shape, axis_len=length,
    )


def unpack_dequantize(pt: PackedTensor) -> torch.Tensor:
    """``quantize_pack`` 的逆: 反量化回 fp16, 形状与量化轴还原为原张量。"""
    axis = pt.axis
    moved_prefix = tuple(pt.orig_shape[i] for i in range(len(pt.orig_shape)) if i != axis)
    gs = pt.group_size
    n_groups = pt.scale.shape[-1]        # 真实组数 = ceil(axis_len / gs)
    padded = n_groups * gs

    codes = unpack_low_bits(pt.packed, pt.bits, pt.axis_len).to(torch.float32)
    if pt.axis_len < padded:             # 补回分组补齐位 (占位, dequant 后截掉)
        codes = torch.cat(
            [codes, codes.new_zeros(*codes.shape[:-1], padded - pt.axis_len)], dim=-1)
    codes_g = codes.reshape(*moved_prefix, n_groups, gs)
    scale = pt.scale.float().unsqueeze(-1)
    mins = pt.mins.float().unsqueeze(-1)
    deq = (codes_g * scale + mins).reshape(*moved_prefix, padded)[..., :pt.axis_len]
    deq = deq.movedim(-1, axis).contiguous()
    return deq.to(torch.float16)
