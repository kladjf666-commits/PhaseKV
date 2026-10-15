"""PhaseKV 阶段 B: 单遍混精度融合 decode 注意力 Triton kernel (CH-2 架构)。

带宽受限的 decode 每步都要把整份 KV 从 HBM 读进来; PhaseKV 把 prompt/冷层以真实低比特
打包 (``packing.PackedTensor``) 驻留 HBM, 本 kernel 一次注意力**只从 HBM 读 packed 字节**,
在片上 (寄存器/共享内存) 反量化后做 flash 式 online-softmax, 不把 KV 反量化回 HBM、不做
dense 回退。因此 decode 的 HBM 读流量按位宽比例下降 —— 这是可报告的加速来源 (阶段 B)。

本文件先交付*正确性打通版*: 单遍读 packed K/V (per-token 分组量化, 沿 head_dim 打包) +
片上反量化 + flash-decode。数值基准为 ``reference.fused_decode_attention_reference``。三层
(prompt+冷+热) 单遍融合与 monkeypatch 端到端测速在其上扩展。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .packing import PackedTensor, _vals_per_byte


@triton.jit
def _decode_attn_packed_kernel(
    q_ptr, o_ptr,
    kp_ptr, ks_ptr, km_ptr,
    vp_ptr, vs_ptr, vm_ptr,
    S, scale,
    stride_qh, stride_qd,
    stride_oh, stride_od,
    stride_kkv, stride_ks, stride_kb,      # packed K: [Hkv, S, HD_PACKED]
    stride_skv, stride_ss, stride_sg,      # K scale/min: [Hkv, S, N_GROUPS]
    stride_vkv, stride_vs, stride_vb,      # packed V: [Hkv, S, HD_PACKED]
    stride_vskv, stride_vss, stride_vsg,   # V scale/min: [Hkv, S, N_GROUPS]
    GROUP: tl.constexpr, BITS: tl.constexpr, VPB: tl.constexpr, GS: tl.constexpr,
    HD: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """每个 program 处理一个 query head 的整段 decode 注意力 (单 query, 单 batch)。"""
    qh = tl.program_id(0)
    kv = qh // GROUP                                  # GQA: query head -> kv head

    d = tl.arange(0, HD)
    mask_d = d < HD
    q = tl.load(q_ptr + qh * stride_qh + d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)

    byte_idx = d // VPB                               # 该 head_dim 位于 packed 的哪个字节
    shift = (d % VPB) * BITS                          # 字节内位移
    grp = d // GS                                     # 该 head_dim 属于哪个量化组
    code_mask = (1 << BITS) - 1

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HD], dtype=tl.float32)

    for start in range(0, S, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)            # 本块的 key token 下标
        mask_n = n < S

        # --- 片上反量化 K: 只从 HBM 读 packed 字节 + 紧凑 scale/min ---
        kp = tl.load(kp_ptr + kv * stride_kkv + n[:, None] * stride_ks + byte_idx[None, :] * stride_kb,
                     mask=mask_n[:, None] & mask_d[None, :], other=0).to(tl.int32)
        code_k = (kp >> shift[None, :]) & code_mask
        ksc = tl.load(ks_ptr + kv * stride_skv + n[:, None] * stride_ss + grp[None, :] * stride_sg,
                      mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        kmn = tl.load(km_ptr + kv * stride_skv + n[:, None] * stride_ss + grp[None, :] * stride_sg,
                      mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        k = code_k.to(tl.float32) * ksc + kmn        # [BLOCK_N, HD] fp32

        s = tl.sum(q[None, :] * k, axis=1) * scale   # [BLOCK_N]
        s = tl.where(mask_n, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)                        # [BLOCK_N]

        # --- 片上反量化 V ---
        vp = tl.load(vp_ptr + kv * stride_vkv + n[:, None] * stride_vs + byte_idx[None, :] * stride_vb,
                     mask=mask_n[:, None] & mask_d[None, :], other=0).to(tl.int32)
        code_v = (vp >> shift[None, :]) & code_mask
        vsc = tl.load(vs_ptr + kv * stride_vskv + n[:, None] * stride_vss + grp[None, :] * stride_vsg,
                      mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        vmn = tl.load(vm_ptr + kv * stride_vskv + n[:, None] * stride_vss + grp[None, :] * stride_vsg,
                      mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = code_v.to(tl.float32) * vsc + vmn        # [BLOCK_N, HD] fp32

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    o = acc / l_i
    tl.store(o_ptr + qh * stride_oh + d * stride_od, o.to(tl.float16), mask=mask_d)


def decode_attention_packed(
    q: torch.Tensor, packed_k: PackedTensor, packed_v: PackedTensor,
    num_attention_heads: int, block_n: int = 64,
) -> torch.Tensor:
    """在片上反量化的单遍 packed-KV decode 注意力 (单 batch, 单 query 步)。

    Args:
        q: 查询 ``[Hq, hd]`` fp16 (单 batch 已 squeeze)。
        packed_k/packed_v: 由 ``quantize_pack(x, bits, axis=-1, gs)`` 得到的 per-token
            打包 KV, 原张量形状 ``[Hkv, S, hd]``。
        num_attention_heads: Hq (GQA 展开由 kernel 内 head 映射完成)。

    Returns:
        注意力输出 ``[Hq, hd]`` fp16。
    """
    hkv, S, hd = packed_k.orig_shape
    if q.shape != (num_attention_heads, hd):
        raise ValueError(f"q 期望 [Hq={num_attention_heads}, hd={hd}], 实际 {tuple(q.shape)}")
    if num_attention_heads % hkv != 0:
        raise ValueError(f"Hq {num_attention_heads} 不是 Hkv {hkv} 的整数倍")
    bits = packed_k.bits
    vpb = _vals_per_byte(bits)
    gs = packed_k.group_size
    if hd % vpb != 0 or hd % gs != 0:
        raise ValueError(f"kernel 要求 head_dim({hd}) 可被 vpb({vpb}) 与 group_size({gs}) 整除")

    q = q.contiguous()
    o = torch.empty((num_attention_heads, hd), dtype=torch.float16, device=q.device)
    scale = 1.0 / (hd ** 0.5)
    n_groups = packed_k.scale.shape[-1]

    kp, ks, km = packed_k.packed.contiguous(), packed_k.scale.contiguous(), packed_k.mins.contiguous()
    vp, vs, vm = packed_v.packed.contiguous(), packed_v.scale.contiguous(), packed_v.mins.contiguous()

    _decode_attn_packed_kernel[(num_attention_heads,)](
        q, o,
        kp, ks, km, vp, vs, vm,
        S, scale,
        q.stride(0), q.stride(1),
        o.stride(0), o.stride(1),
        kp.stride(0), kp.stride(1), kp.stride(2),
        ks.stride(0), ks.stride(1), ks.stride(2),
        vp.stride(0), vp.stride(1), vp.stride(2),
        vs.stride(0), vs.stride(1), vs.stride(2),
        GROUP=num_attention_heads // hkv, BITS=bits, VPB=vpb, GS=gs,
        HD=hd, BLOCK_N=block_n,
    )
    return o


@triton.jit
def _decode_attn_packed_split_kernel(
    q_ptr, pm_ptr, pl_ptr, pacc_ptr,
    kp_ptr, ks_ptr, km_ptr,
    vp_ptr, vs_ptr, vm_ptr,
    S, scale, split_len,
    stride_qh, stride_qd,
    stride_pmh, stride_pms,
    stride_pah, stride_pas, stride_pad,
    stride_kkv, stride_ks, stride_kb,
    stride_skv, stride_ss, stride_sg,
    stride_vkv, stride_vs, stride_vb,
    stride_vskv, stride_vss, stride_vsg,
    GROUP: tl.constexpr, BITS: tl.constexpr, VPB: tl.constexpr, GS: tl.constexpr,
    HD: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """split-K flash-decoding 阶段1: (query head, split) 各算一段 keys 的 partial softmax。"""
    qh = tl.program_id(0)
    sp = tl.program_id(1)
    kv = qh // GROUP
    seg_start = sp * split_len
    seg_end = tl.minimum(seg_start + split_len, S)

    d = tl.arange(0, HD)
    mask_d = d < HD
    q = tl.load(q_ptr + qh * stride_qh + d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)
    byte_idx = d // VPB
    shift = (d % VPB) * BITS
    grp = d // GS
    code_mask = (1 << BITS) - 1

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HD], dtype=tl.float32)

    for start in range(seg_start, seg_end, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        mask_n = n < seg_end
        kpk = tl.load(kp_ptr + kv * stride_kkv + n[:, None] * stride_ks + byte_idx[None, :] * stride_kb,
                      mask=mask_n[:, None] & mask_d[None, :], other=0).to(tl.int32)
        code_k = (kpk >> shift[None, :]) & code_mask
        ksc = tl.load(ks_ptr + kv * stride_skv + n[:, None] * stride_ss + grp[None, :] * stride_sg,
                      mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        kmn = tl.load(km_ptr + kv * stride_skv + n[:, None] * stride_ss + grp[None, :] * stride_sg,
                      mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        k = code_k.to(tl.float32) * ksc + kmn

        s = tl.sum(q[None, :] * k, axis=1) * scale
        s = tl.where(mask_n, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)

        vpk = tl.load(vp_ptr + kv * stride_vkv + n[:, None] * stride_vs + byte_idx[None, :] * stride_vb,
                      mask=mask_n[:, None] & mask_d[None, :], other=0).to(tl.int32)
        code_v = (vpk >> shift[None, :]) & code_mask
        vsc = tl.load(vs_ptr + kv * stride_vskv + n[:, None] * stride_vss + grp[None, :] * stride_vsg,
                      mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        vmn = tl.load(vm_ptr + kv * stride_vskv + n[:, None] * stride_vss + grp[None, :] * stride_vsg,
                      mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = code_v.to(tl.float32) * vsc + vmn

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    tl.store(pm_ptr + qh * stride_pmh + sp * stride_pms, m_i)
    tl.store(pl_ptr + qh * stride_pmh + sp * stride_pms, l_i)
    tl.store(pacc_ptr + qh * stride_pah + sp * stride_pas + d * stride_pad, acc, mask=mask_d)


def _packed_split_launch(
    q: torch.Tensor, packed_k: PackedTensor, packed_v: PackedTensor,
    num_attention_heads: int, num_splits: int, block_n: int,
):
    """启动 split-K 阶段1 kernel, 返回分段 partial (pm, pl, pacc) [Hq, num_splits(, hd)]。"""
    hkv, S, hd = packed_k.orig_shape
    if q.shape != (num_attention_heads, hd):
        raise ValueError(f"q 期望 [Hq={num_attention_heads}, hd={hd}], 实际 {tuple(q.shape)}")
    bits = packed_k.bits
    vpb = _vals_per_byte(bits)
    gs = packed_k.group_size
    num_splits = max(1, min(num_splits, S))
    split_len = triton.cdiv(S, num_splits)

    q = q.contiguous()
    dev = q.device
    pm = torch.full((num_attention_heads, num_splits), -float("inf"), dtype=torch.float32, device=dev)
    pl = torch.zeros((num_attention_heads, num_splits), dtype=torch.float32, device=dev)
    pacc = torch.zeros((num_attention_heads, num_splits, hd), dtype=torch.float32, device=dev)

    kp, ks, km = packed_k.packed.contiguous(), packed_k.scale.contiguous(), packed_k.mins.contiguous()
    vp, vs, vm = packed_v.packed.contiguous(), packed_v.scale.contiguous(), packed_v.mins.contiguous()
    scale = 1.0 / (hd ** 0.5)

    _decode_attn_packed_split_kernel[(num_attention_heads, num_splits)](
        q, pm, pl, pacc,
        kp, ks, km, vp, vs, vm,
        S, scale, split_len,
        q.stride(0), q.stride(1),
        pm.stride(0), pm.stride(1),
        pacc.stride(0), pacc.stride(1), pacc.stride(2),
        kp.stride(0), kp.stride(1), kp.stride(2),
        ks.stride(0), ks.stride(1), ks.stride(2),
        vp.stride(0), vp.stride(1), vp.stride(2),
        vs.stride(0), vs.stride(1), vs.stride(2),
        GROUP=num_attention_heads // hkv, BITS=bits, VPB=vpb, GS=gs,
        HD=hd, BLOCK_N=block_n,
    )
    return pm, pl, pacc


def _reduce_splits(pm, pl, pacc):
    """沿 split 维把分段 partial 合并成整层单个 flash partial (m,l,acc)。

    以层内全局 m 为基准; 返回的 acc/l 未归一化 (供跨层再合并), o = acc/l 才是本层输出。
    """
    m = pm.max(dim=1).values                     # [Hq]
    w = torch.exp(pm - m[:, None])               # [Hq, num_splits]
    l = (w * pl).sum(dim=1)                       # [Hq]
    acc = (w[:, :, None] * pacc).sum(dim=1)       # [Hq, hd]
    return m, l, acc


def decode_attention_packed_splitk(
    q: torch.Tensor, packed_k: PackedTensor, packed_v: PackedTensor,
    num_attention_heads: int, num_splits: int = 8, block_n: int = 64,
) -> torch.Tensor:
    """split-K flash-decoding: 并行度 Hq -> Hq*num_splits 打满 SM, 再合并 partial softmax。"""
    pm, pl, pacc = _packed_split_launch(q, packed_k, packed_v, num_attention_heads, num_splits, block_n)
    _, l_g, acc_g = _reduce_splits(pm, pl, pacc)
    return (acc_g / l_g[:, None]).to(torch.float16)


def _packed_layer_partial(q, packed_k, packed_v, num_attention_heads, num_splits, block_n):
    """一个 packed 层的整层 flash partial (m,l,acc), 供三层融合跨层合并。"""
    pm, pl, pacc = _packed_split_launch(q, packed_k, packed_v, num_attention_heads, num_splits, block_n)
    return _reduce_splits(pm, pl, pacc)


def _fp16_layer_partial(q, hot_k, hot_v, num_attention_heads):
    """FP16 热层的 flash partial (m,l,acc); 热层小, 直接用 torch 算 (无需片上反量化)。"""
    hkv, hot, hd = hot_k.shape
    group = num_attention_heads // hkv
    K = hot_k.repeat_interleave(group, 0).float()          # [Hq, hot, hd]
    V = hot_v.repeat_interleave(group, 0).float()
    scale = 1.0 / (hd ** 0.5)
    s = (q.float().unsqueeze(1) @ K.transpose(-1, -2)).squeeze(1) * scale   # [Hq, hot]
    m = s.max(dim=1).values                                 # [Hq]
    p = torch.exp(s - m[:, None])                           # [Hq, hot]
    l = p.sum(dim=1)                                         # [Hq]
    acc = (p.unsqueeze(1) @ V).squeeze(1)                    # [Hq, hd]
    return m, l, acc


def _merge_partials(partials):
    """跨层 online-softmax 合并多层 flash partial (m,l,acc) -> 最终输出 [Hq, hd] fp16。"""
    ms = torch.stack([p[0] for p in partials], dim=1)       # [Hq, n_layers]
    m = ms.max(dim=1).values                                # [Hq]
    w = torch.exp(ms - m[:, None])                          # [Hq, n_layers]
    l = sum(w[:, i] * partials[i][1] for i in range(len(partials)))          # [Hq]
    acc = sum(w[:, i:i + 1] * partials[i][2] for i in range(len(partials)))  # [Hq, hd]
    return (acc / l[:, None]).to(torch.float16)


@triton.jit
def _fused_three_tier_split_kernel(
    q_ptr, pm_ptr, pl_ptr, pacc_ptr,
    pkp_ptr, pks_ptr, pkm_ptr, pvp_ptr, pvs_ptr, pvm_ptr,   # prompt packed (bits_p)
    ckp_ptr, cks_ptr, ckm_ptr, cvp_ptr, cvs_ptr, cvm_ptr,   # cold packed (bits_c)
    hk_ptr, hv_ptr,                                          # hot fp16
    K0, COLD, HOT, TOTAL, scale, split_len,
    stride_qh, stride_qd,
    stride_pmh, stride_pms,
    stride_pah, stride_pas, stride_pad,
    s_pkp_kv, s_pkp_s, s_pkp_b, s_pks_kv, s_pks_s, s_pks_g,  # prompt K packed / scale strides
    s_ckp_kv, s_ckp_s, s_ckp_b, s_cks_kv, s_cks_s, s_cks_g,  # cold K packed / scale strides
    s_hk_kv, s_hk_s, s_hk_d,                                 # hot K strides
    GROUP: tl.constexpr, BITS_P: tl.constexpr, VPB_P: tl.constexpr,
    BITS_C: tl.constexpr, VPB_C: tl.constexpr, GS: tl.constexpr,
    HD: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """真·单遍三层混精融合: 一个 kernel 内按 key 全局位置选 prompt/cold/hot 层, 谓词化 load。"""
    qh = tl.program_id(0)
    sp = tl.program_id(1)
    kv = qh // GROUP
    seg_start = sp * split_len
    seg_end = tl.minimum(seg_start + split_len, TOTAL)

    d = tl.arange(0, HD)
    mask_d = d < HD
    q = tl.load(q_ptr + qh * stride_qh + d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)
    grp = d // GS
    byte_p = d // VPB_P
    shift_p = (d % VPB_P) * BITS_P
    mask_pc = (1 << BITS_P) - 1
    byte_c = d // VPB_C
    shift_c = (d % VPB_C) * BITS_C
    mask_cc = (1 << BITS_C) - 1

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HD], dtype=tl.float32)

    for start in range(seg_start, seg_end, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        mask_n = n < seg_end
        is_p = n < K0
        is_c = (n >= K0) & (n < K0 + COLD)
        is_h = n >= (K0 + COLD)

        # prompt (bits_p), 局部下标 = n, 谓词化 load (仅 is_p 车道真正读 HBM)
        ip = tl.minimum(tl.maximum(n, 0), tl.maximum(K0 - 1, 0))
        kp = tl.load(pkp_ptr + kv * s_pkp_kv + ip[:, None] * s_pkp_s + byte_p[None, :] * s_pkp_b,
                     mask=is_p[:, None] & mask_d[None, :], other=0).to(tl.int32)
        kpf = ((kp >> shift_p[None, :]) & mask_pc).to(tl.float32) * \
            tl.load(pks_ptr + kv * s_pks_kv + ip[:, None] * s_pks_s + grp[None, :] * s_pks_g,
                    mask=is_p[:, None] & mask_d[None, :], other=0.0).to(tl.float32) + \
            tl.load(pkm_ptr + kv * s_pks_kv + ip[:, None] * s_pks_s + grp[None, :] * s_pks_g,
                    mask=is_p[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        vpf = ((tl.load(pvp_ptr + kv * s_pkp_kv + ip[:, None] * s_pkp_s + byte_p[None, :] * s_pkp_b,
                        mask=is_p[:, None] & mask_d[None, :], other=0).to(tl.int32)
                >> shift_p[None, :]) & mask_pc).to(tl.float32) * \
            tl.load(pvs_ptr + kv * s_pks_kv + ip[:, None] * s_pks_s + grp[None, :] * s_pks_g,
                    mask=is_p[:, None] & mask_d[None, :], other=0.0).to(tl.float32) + \
            tl.load(pvm_ptr + kv * s_pks_kv + ip[:, None] * s_pks_s + grp[None, :] * s_pks_g,
                    mask=is_p[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # cold (bits_c), 局部下标 = n - K0
        ic = tl.minimum(tl.maximum(n - K0, 0), tl.maximum(COLD - 1, 0))
        kc = tl.load(ckp_ptr + kv * s_ckp_kv + ic[:, None] * s_ckp_s + byte_c[None, :] * s_ckp_b,
                     mask=is_c[:, None] & mask_d[None, :], other=0).to(tl.int32)
        kcf = ((kc >> shift_c[None, :]) & mask_cc).to(tl.float32) * \
            tl.load(cks_ptr + kv * s_cks_kv + ic[:, None] * s_cks_s + grp[None, :] * s_cks_g,
                    mask=is_c[:, None] & mask_d[None, :], other=0.0).to(tl.float32) + \
            tl.load(ckm_ptr + kv * s_cks_kv + ic[:, None] * s_cks_s + grp[None, :] * s_cks_g,
                    mask=is_c[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        vcf = ((tl.load(cvp_ptr + kv * s_ckp_kv + ic[:, None] * s_ckp_s + byte_c[None, :] * s_ckp_b,
                        mask=is_c[:, None] & mask_d[None, :], other=0).to(tl.int32)
                >> shift_c[None, :]) & mask_cc).to(tl.float32) * \
            tl.load(cvs_ptr + kv * s_cks_kv + ic[:, None] * s_cks_s + grp[None, :] * s_cks_g,
                    mask=is_c[:, None] & mask_d[None, :], other=0.0).to(tl.float32) + \
            tl.load(cvm_ptr + kv * s_cks_kv + ic[:, None] * s_cks_s + grp[None, :] * s_cks_g,
                    mask=is_c[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # hot (fp16), 局部下标 = n - K0 - COLD
        ih = tl.minimum(tl.maximum(n - K0 - COLD, 0), tl.maximum(HOT - 1, 0))
        khf = tl.load(hk_ptr + kv * s_hk_kv + ih[:, None] * s_hk_s + d[None, :] * s_hk_d,
                      mask=is_h[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        vhf = tl.load(hv_ptr + kv * s_hk_kv + ih[:, None] * s_hk_s + d[None, :] * s_hk_d,
                      mask=is_h[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        k = tl.where(is_p[:, None], kpf, tl.where(is_c[:, None], kcf, khf))
        v = tl.where(is_p[:, None], vpf, tl.where(is_c[:, None], vcf, vhf))

        s = tl.sum(q[None, :] * k, axis=1) * scale
        s = tl.where(mask_n, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    tl.store(pm_ptr + qh * stride_pmh + sp * stride_pms, m_i)
    tl.store(pl_ptr + qh * stride_pmh + sp * stride_pms, l_i)
    tl.store(pacc_ptr + qh * stride_pah + sp * stride_pas + d * stride_pad, acc, mask=mask_d)


def fused_three_tier_decode_single(q, kv, num_attention_heads, num_splits: int = 32, block_n: int = 64):
    """真·单遍三层融合 decode: 一个 kernel 覆盖 prompt+冷+热三层, 单次 launch + 单次 reduce。

    消除分层版 (fused_three_tier_decode) 的多 kernel launch 与 torch 跨层合并开销, 对 GPU
    争用不敏感 (延迟稳定)。prompt/cold 须为 per-token 打包 (axis=-1) PackedTensor, hot 为
    fp16 [Hkv, hot, hd]。

    实测权衡 (已验证正确性 maxerr~1e-4): SIMT 下每个 token 会执行 prompt/cold/hot 三层的
    *全部* 反量化计算指令 (内存访问被谓词化, 但计算指令不被谓词化) => ~3x 计算冗余, 稳定地
    慢于分层版。干净加速需按位宽把不同层分配给不同 program (避免 SIMT 跨层冗余); 在此优化
    落地前, 默认三层融合仍用 fused_three_tier_decode (分层 flash 合并, 数值一致)。
    """
    prompt_k, cold_k, hot_k = kv.prompt_k, kv.cold_k, kv.hot_k
    ref = prompt_k if prompt_k is not None else cold_k
    hkv, _, hd = ref.orig_shape
    dev = q.device
    K0 = prompt_k.orig_shape[1] if prompt_k is not None else 0
    COLD = cold_k.orig_shape[1] if cold_k is not None else 0
    HOT = hot_k.shape[1] if hot_k is not None else 0
    TOTAL = K0 + COLD + HOT
    if TOTAL == 0:
        raise ValueError("三层皆空")
    gs = ref.group_size
    bits_p = prompt_k.bits if prompt_k is not None else 8
    bits_c = cold_k.bits if cold_k is not None else 8
    num_splits = max(1, min(num_splits, TOTAL))
    split_len = triton.cdiv(TOTAL, num_splits)

    def _empty_packed(bits):
        z = torch.zeros((hkv, 1, hd // _vals_per_byte(bits)), dtype=torch.uint8, device=dev)
        sc = torch.zeros((hkv, 1, max(1, hd // gs)), dtype=torch.float16, device=dev)
        return z, sc, sc
    if prompt_k is not None:
        pkp, pks, pkm = prompt_k.packed, prompt_k.scale, prompt_k.mins
        pvp, pvs, pvm = kv.prompt_v.packed, kv.prompt_v.scale, kv.prompt_v.mins
    else:
        pkp, pks, pkm = _empty_packed(bits_p); pvp, pvs, pvm = pkp, pks, pkm
    if cold_k is not None:
        ckp, cks, ckm = cold_k.packed, cold_k.scale, cold_k.mins
        cvp, cvs, cvm = kv.cold_v.packed, kv.cold_v.scale, kv.cold_v.mins
    else:
        ckp, cks, ckm = _empty_packed(bits_c); cvp, cvs, cvm = ckp, cks, ckm
    if hot_k is not None:
        hk, hv = hot_k.contiguous(), kv.hot_v.contiguous()
    else:
        hk = hv = torch.zeros((hkv, 1, hd), dtype=torch.float16, device=dev)

    q = q.contiguous()
    Hq = num_attention_heads
    pm = torch.full((Hq, num_splits), -float("inf"), dtype=torch.float32, device=dev)
    pl = torch.zeros((Hq, num_splits), dtype=torch.float32, device=dev)
    pacc = torch.zeros((Hq, num_splits, hd), dtype=torch.float32, device=dev)
    scale = 1.0 / (hd ** 0.5)

    _fused_three_tier_split_kernel[(Hq, num_splits)](
        q, pm, pl, pacc,
        pkp, pks, pkm, pvp, pvs, pvm,
        ckp, cks, ckm, cvp, cvs, cvm,
        hk, hv,
        K0, COLD, HOT, TOTAL, scale, split_len,
        q.stride(0), q.stride(1),
        pm.stride(0), pm.stride(1),
        pacc.stride(0), pacc.stride(1), pacc.stride(2),
        pkp.stride(0), pkp.stride(1), pkp.stride(2), pks.stride(0), pks.stride(1), pks.stride(2),
        ckp.stride(0), ckp.stride(1), ckp.stride(2), cks.stride(0), cks.stride(1), cks.stride(2),
        hk.stride(0), hk.stride(1), hk.stride(2),
        GROUP=Hq // hkv, BITS_P=bits_p, VPB_P=_vals_per_byte(bits_p),
        BITS_C=bits_c, VPB_C=_vals_per_byte(bits_c), GS=gs, HD=hd, BLOCK_N=block_n,
    )
    _, l_g, acc_g = _reduce_splits(pm, pl, pacc)
    return (acc_g / l_g[:, None]).to(torch.float16)


def fused_three_tier_decode(q, kv, num_attention_heads, num_splits: int = 8, block_n: int = 64):
    """PhaseKV 阶段B 三层单遍混精融合 decode 注意力 (idea 3.3)。

    一次注意力覆盖 [packed prompt] + [packed 冷层] + [FP16 热层]: packed 层经 split-K kernel
    片上反量化算 flash partial, 热层 fp16 直算 partial, 再跨层 online-softmax 合并。数值与
    ``reference.fused_decode_attention_reference`` 一致 (由自检保证), 不回退 dense。

    Args:
        q: ``[Hq, hd]`` fp16。
        kv: ``reference.ThreeTierKV`` (prompt/cold 为 per-token 打包 PackedTensor, hot 为 fp16)。
    Returns:
        ``[Hq, hd]`` fp16。
    """
    partials = []
    if kv.prompt_k is not None:
        partials.append(_packed_layer_partial(q, kv.prompt_k, kv.prompt_v,
                                              num_attention_heads, num_splits, block_n))
    if kv.cold_k is not None:
        partials.append(_packed_layer_partial(q, kv.cold_k, kv.cold_v,
                                              num_attention_heads, num_splits, block_n))
    if kv.hot_k is not None:
        partials.append(_fp16_layer_partial(q, kv.hot_k, kv.hot_v, num_attention_heads))
    if not partials:
        raise ValueError("三层皆空, 无可注意的 KV")
    return _merge_partials(partials)


@triton.jit
def _decode_attn_kivi_split_kernel(
    q_ptr, pm_ptr, pl_ptr, pacc_ptr,
    kp_ptr, ks_ptr, km_ptr, vp_ptr, vs_ptr, vm_ptr,
    S, scale, split_len,
    stride_qh, stride_qd,
    stride_pmh, stride_pms,
    stride_pah, stride_pas, stride_pad,
    stride_kkv, stride_kd, stride_kb,
    stride_kskv, stride_ksd, stride_ksg,
    stride_vkv, stride_vs, stride_vb,
    stride_vskv, stride_vss, stride_vsg,
    GROUP: tl.constexpr, KBITS: tl.constexpr, KVPB: tl.constexpr,
    VBITS: tl.constexpr, VVPB: tl.constexpr,
    KGS: tl.constexpr, VGS: tl.constexpr,
    HD: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """K 按 token 打包、V 按 head_dim 打包的 PhaseKV/KIVI 分段注意力。"""
    qh = tl.program_id(0)
    sp = tl.program_id(1)
    kv = qh // GROUP
    start_token = sp * split_len
    end_token = tl.minimum(start_token + split_len, S)
    d = tl.arange(0, HD)
    q = tl.load(q_ptr + qh * stride_qh + d * stride_qd).to(tl.float32)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HD], dtype=tl.float32)

    for start in range(start_token, end_token, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        valid = n < end_token
        valid_nd = valid[:, None] & (d[None, :] < HD)

        # K: packed [Hkv, hd, ceil(S / KVPB)], metadata [Hkv, hd, ceil(S / KGS)].
        kb = tl.load(
            kp_ptr + kv * stride_kkv + d[None, :] * stride_kd
            + (n[:, None] // KVPB) * stride_kb,
            mask=valid_nd, other=0,
        ).to(tl.int32)
        kc = (kb >> ((n[:, None] % KVPB) * KBITS)) & ((1 << KBITS) - 1)
        kscale = tl.load(
            ks_ptr + kv * stride_kskv + d[None, :] * stride_ksd
            + (n[:, None] // KGS) * stride_ksg,
            mask=valid_nd, other=0.0,
        ).to(tl.float32)
        kmin = tl.load(
            km_ptr + kv * stride_kskv + d[None, :] * stride_ksd
            + (n[:, None] // KGS) * stride_ksg,
            mask=valid_nd, other=0.0,
        ).to(tl.float32)
        k = kc.to(tl.float32) * kscale + kmin
        score = tl.sum(q[None, :] * k, axis=1) * scale
        score = tl.where(valid, score, -float("inf"))
        new_m = tl.maximum(m_i, tl.max(score, axis=0))
        alpha = tl.exp(m_i - new_m)
        prob = tl.exp(score - new_m)

        # V: packed [Hkv, S, ceil(hd / VVPB)], metadata [Hkv, S, ceil(hd / VGS)].
        vb = tl.load(
            vp_ptr + kv * stride_vkv + n[:, None] * stride_vs
            + (d[None, :] // VVPB) * stride_vb,
            mask=valid_nd, other=0,
        ).to(tl.int32)
        vc = (vb >> ((d[None, :] % VVPB) * VBITS)) & ((1 << VBITS) - 1)
        vscale = tl.load(
            vs_ptr + kv * stride_vskv + n[:, None] * stride_vss
            + (d[None, :] // VGS) * stride_vsg,
            mask=valid_nd, other=0.0,
        ).to(tl.float32)
        vmin = tl.load(
            vm_ptr + kv * stride_vskv + n[:, None] * stride_vss
            + (d[None, :] // VGS) * stride_vsg,
            mask=valid_nd, other=0.0,
        ).to(tl.float32)
        v = vc.to(tl.float32) * vscale + vmin
        acc = acc * alpha + tl.sum(prob[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(prob, axis=0)
        m_i = new_m

    tl.store(pm_ptr + qh * stride_pmh + sp * stride_pms, m_i)
    tl.store(pl_ptr + qh * stride_pmh + sp * stride_pms, l_i)
    tl.store(pacc_ptr + qh * stride_pah + sp * stride_pas + d * stride_pad, acc)


def _kivi_layer_partial(q, key: PackedTensor, value: PackedTensor,
                        num_attention_heads: int, num_splits: int, block_n: int):
    """读取一个真实 PhaseKV packed 段，返回可跨段合并的 softmax partial。"""
    if key.axis != 1 or value.axis != 2 or len(key.orig_shape) != 3:
        raise ValueError("K 应沿 token 轴、V 应沿 head_dim 轴打包，且 batch=1 已移除")
    hkv, seq_len, head_dim = key.orig_shape
    if value.orig_shape != key.orig_shape or seq_len <= 0:
        raise ValueError("packed K/V 形状不一致或为空")
    if q.shape != (num_attention_heads, head_dim) or q.dtype != torch.float16:
        raise ValueError("query 必须为 [Hq, head_dim] float16")
    if num_attention_heads % hkv or head_dim & (head_dim - 1):
        raise ValueError("Hq 必须是 Hkv 的整数倍，head_dim 必须为 2 的幂")
    if q.device != key.packed.device or q.device != value.packed.device:
        raise ValueError("query 与 packed K/V 必须在同一设备")
    if num_splits <= 0 or block_n <= 0 or block_n & (block_n - 1):
        raise ValueError("num_splits 必须为正，block_n 必须为 2 的幂")
    hq = num_attention_heads
    splits = min(num_splits, seq_len)
    split_len = triton.cdiv(seq_len, splits)
    q = q.contiguous()
    pm = torch.empty((hq, splits), device=q.device, dtype=torch.float32)
    pl = torch.empty((hq, splits), device=q.device, dtype=torch.float32)
    pacc = torch.empty((hq, splits, head_dim), device=q.device, dtype=torch.float32)
    kp, ks, km = key.packed.contiguous(), key.scale.contiguous(), key.mins.contiguous()
    vp, vs, vm = value.packed.contiguous(), value.scale.contiguous(), value.mins.contiguous()
    _decode_attn_kivi_split_kernel[(hq, splits)](
        q, pm, pl, pacc, kp, ks, km, vp, vs, vm,
        seq_len, 1.0 / (head_dim ** 0.5), split_len,
        *q.stride(), *pm.stride(), *pacc.stride(),
        *kp.stride(), *ks.stride(), *vp.stride(), *vs.stride(),
        GROUP=hq // hkv, KBITS=key.bits, KVPB=_vals_per_byte(key.bits),
        VBITS=value.bits, VVPB=_vals_per_byte(value.bits),
        KGS=key.group_size, VGS=value.group_size,
        HD=head_dim, BLOCK_N=block_n,
    )
    return _reduce_splits(pm, pl, pacc)


#该函数只能一次计算一个token的注意力
def fused_phasekv_decode(q: torch.Tensor, tiers, num_attention_heads: int,
                         num_splits: int = 8, block_n: int = 64) -> torch.Tensor:
    """batch=1 的 PhaseKV 注意力：逐段读取 packed KV，再合并 softmax。

    ``tiers`` 是 ``PhaseKVCache.layer_tiers`` 的返回值。这里不解压整份 KV，
    但冷层有多个冻结段时会产生多个 Triton launch；后续可进一步融合 launch。
    """
    from dataclasses import replace

    def remove_batch(pt: PackedTensor) -> PackedTensor:
        if pt.orig_shape[0] != 1:
            raise ValueError("当前 PhaseKV Triton 路径只支持 batch=1")
        return replace(pt, packed=pt.packed[0], scale=pt.scale[0], mins=pt.mins[0],
                       orig_shape=pt.orig_shape[1:], axis=pt.axis - 1)

    partials = []
    for segment in [tiers.prompt, *tiers.prompt_extensions, *tiers.cold]:
        partials.append(_kivi_layer_partial(
            q, remove_batch(segment.key), remove_batch(segment.value),
            num_attention_heads, num_splits, block_n))
    if tiers.hot_len:
        hot_k, hot_v = tiers.hot_views()
        partials.append(_fp16_layer_partial(q, hot_k[0], hot_v[0], num_attention_heads))
    return _merge_partials(partials)
