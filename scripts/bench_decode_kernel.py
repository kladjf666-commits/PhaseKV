"""PhaseKV 阶段 B kernel 级 decode 注意力真实测速 (CH-2 架构)。

对比同一 decode 单步注意力的三条路径, 全部真实 GPU 测量 (warmup + CUDA events + 多次中位数):
  - fp16-SDPA      : PyTorch scaled_dot_product_attention 读全精度 KV (模型现状基线)。
  - pk{bits}-single: PhaseKV packed 低比特 KV + 片上反量化, 单 program/head (未优化)。
  - pk{bits}-splitK: 同上 + split-K flash-decoding (并行度 Hq*num_splits 打满 SM)。

红线 (遵守项目约定): 数字全部来自真实 kernel 端到端计时, 无模拟; 每条同时记录 packed 与
fp16 的 KV 读字节, 使"加速"可对照"带宽节省"归因。这是 kernel 级 microbench (单 batch/单
query 步/单层), 模型端到端 TTFT/TPOT 由 monkeypatch 后另测, 不用本结果冒充端到端加速比。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phasekv.kernels import packing as pk
from phasekv.kernels.packing import unpack_dequantize
from phasekv.kernels.attention_triton import (
    decode_attention_packed, decode_attention_packed_splitk)


def _gpu_util() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def _reference(q, pkK, pkV, group, hd):
    Kd = unpack_dequantize(pkK).repeat_interleave(group, 0).float()
    Vd = unpack_dequantize(pkV).repeat_interleave(group, 0).float()
    s = (q.float().unsqueeze(1) @ Kd.transpose(-1, -2)).squeeze(1) * (1.0 / hd ** 0.5)
    return (s.softmax(-1).unsqueeze(1) @ Vd).squeeze(1).half()


def run(hkv, hq, hd, gs, seqs, bits_list, splits_grid, warmup, iters, seed):
    dev = "cuda"
    torch.manual_seed(seed)
    group = hq // hkv
    rows = []
    for S in seqs:
        K = torch.randn(hkv, S, hd, dtype=torch.float16, device=dev)
        V = torch.randn(hkv, S, hd, dtype=torch.float16, device=dev)
        q = torch.randn(hq, hd, dtype=torch.float16, device=dev)
        qb = q.view(1, hq, 1, hd)
        Kb = K.repeat_interleave(group, 0).view(1, hq, S, hd)
        Vb = V.repeat_interleave(group, 0).view(1, hq, S, hd)
        t_sdpa = _bench(lambda: F.scaled_dot_product_attention(qb, Kb, Vb), warmup, iters)
        fp16_kv_bytes = 2 * hkv * S * hd * 2

        for bits in bits_list:
            pkK = pk.quantize_pack(K, bits, -1, gs)
            pkV = pk.quantize_pack(V, bits, -1, gs)
            packed_kv_bytes = pkK.nbytes + pkV.nbytes
            o_ref = _reference(q, pkK, pkV, group, hd)
            err_single = (decode_attention_packed(q, pkK, pkV, hq).float()
                          - o_ref.float()).abs().max().item()
            t_single = _bench(lambda: decode_attention_packed(q, pkK, pkV, hq), warmup, iters)

            best_t, best_ns, best_err = 1e18, 0, 0.0
            for ns in splits_grid:
                o_sp = decode_attention_packed_splitk(q, pkK, pkV, hq, num_splits=ns)
                t = _bench(lambda: decode_attention_packed_splitk(q, pkK, pkV, hq, num_splits=ns),
                           warmup, iters)
                if t < best_t:
                    best_t, best_ns = t, ns
                    best_err = (o_sp.float() - o_ref.float()).abs().max().item()

            rows.append({
                "seq_len": S, "bits": bits, "num_kv_heads": hkv, "num_q_heads": hq, "head_dim": hd,
                "group_size": gs,
                "sdpa_fp16_ms": round(t_sdpa, 5),
                "packed_single_ms": round(t_single, 5),
                "packed_splitk_ms": round(best_t, 5),
                "best_num_splits": best_ns,
                "speedup_splitk_vs_single": round(t_single / best_t, 3),
                "speedup_splitk_vs_sdpa": round(t_sdpa / best_t, 3),
                "kv_read_bytes_fp16": fp16_kv_bytes,
                "kv_read_bytes_packed": packed_kv_bytes,
                "kv_bytes_saving_pct": round(100.0 * (1 - packed_kv_bytes / fp16_kv_bytes), 2),
                "max_abs_err_single_vs_ref": round(err_single, 6),
                "max_abs_err_splitk_vs_ref": round(best_err, 6),
                "num_samples": iters,
            })
            print(f"[bench] S={S} bits={bits}: SDPA={t_sdpa:.4f} single={t_single:.4f} "
                  f"splitK={best_t:.4f}(ns={best_ns}) vsSDPA={t_sdpa/best_t:.2f}x "
                  f"saving={rows[-1]['kv_bytes_saving_pct']}% err={best_err:.1e}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--num-q-heads", type=int, default=28)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--seqs", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--splits", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default="results/kernel_decode_bench")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU 做真实测速")

    gpu = _gpu_util()
    rows = run(args.num_kv_heads, args.num_q_heads, args.head_dim, args.group_size,
               args.seqs, args.bits, args.splits, args.warmup, args.iters, args.seed)

    meta = {
        "experiment": "stage_b_decode_kernel_microbench",
        "gpu": gpu,
        "note": ("kernel 级 microbench(单 batch/单 query 步/单层, per-token 分组量化); "
                 "真实 CUDA events 计时, 中位数; 测量时 GPU 可能与其他作业共享(见 gpu 字段 util)。"
                 "非模型端到端 TTFT/TPOT, 不作端到端加速比使用。"),
        "torch": torch.__version__,
        "warmup": args.warmup, "iters_median": args.iters,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, ensure_ascii=False, indent=2)

    lines = ["# 阶段B decode kernel microbench (真实测量)", "",
             f"- GPU: {gpu}", f"- 计时: warmup={args.warmup}, 中位数 of {args.iters}",
             "- 口径: 单batch/单query步/单层 kernel microbench; 非端到端 TTFT/TPOT", "",
             "| S | bits | SDPA-fp16(ms) | packed-single(ms) | packed-splitK(ms) | best_ns | vs SDPA | KV省% | err |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seq_len']} | {r['bits']} | {r['sdpa_fp16_ms']} | "
                     f"{r['packed_single_ms']} | {r['packed_splitk_ms']} | {r['best_num_splits']} | "
                     f"{r['speedup_splitk_vs_sdpa']}x | {r['kv_bytes_saving_pct']}% | "
                     f"{r['max_abs_err_splitk_vs_ref']:.1e} |")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[bench] 落盘 {args.out}.json / .md ({len(rows)} 行)")


if __name__ == "__main__":
    main()
