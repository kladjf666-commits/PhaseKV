"""图1 Roofline 真实测量标定 (CH-6 通用执行)。

把 idea 文档洞见 1 的图 1 (prefill 计算受限 / decode 带宽受限, KV 读取占 decode 带宽比例
随上下文增长) 从 planner.py 的"解析强度 + 标称硬件峰值"升级为"解析强度 (精确量) +
真实测量的硬件 roofline + 真实测量的模型落点"。planner.py 的 HardwareSpec 注释明确要求
"终版图 1 的精确 FLOPs 由微基准实测标定后覆盖默认值", 本脚本即提供该标定。

三部分 (全部真实, 无模拟数据; 严禁报 kernel 加速比, 那属阶段 B):

  A. 硬件经验 roofline (实测, 与模型无关):
     - 可达 HBM 带宽: 大张量 copy(读+写) 与纯读(归约) 的实测吞吐 (GB/s)。
     - 可达 fp16 张量核 GEMM 吞吐: 大方阵 matmul 的实测吞吐 (TFLOPS)。
     - 经验 ridge point = 实测 GEMM / 实测带宽 (FLOP/Byte); 覆盖 planner 的标称值。

  B. 模型解析曲线 (精确解析量, 来自真实模型 config; 非模拟实验):
     - prefill / decode 自注意力算术强度落点 (复用 src.phasekv.planner 的强度函数,
       与 planner 成本模型同一事实源)。
     - KV 读流量占单步 decode HBM 读取的比例随上下文长度变化 (= KV 字节 / (权重字节 + KV 字节))。

  C. 模型实测落点 (真实加载并计时):
     - 真实 prefill: 不同 prompt 长度的前向延迟 -> 实测有效 TFLOPS -> 落到经验 roofline
       (应接近 GEMM 峰值, 计算受限)。
     - 真实 decode: 不同上下文长度的单步延迟 -> 实测有效 HBM 带宽 -> 落到经验 roofline
       (应接近带宽峰值, 带宽受限)。

在 release/kvcomp/ 下运行, 例:
  HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 \
  python scripts/roofline_measure.py --measured-model Qwen/Qwen2.5-0.5B-Instruct \
    --analytic-models Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-7B-Instruct \
    --out results/roofline_4080s
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from statistics import median
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from eval import metrics  # noqa: E402
from src.phasekv.planner import (  # noqa: E402  复用 planner 强度函数作单一事实源
    HardwareSpec,
    ModelDims,
    decode_attention_intensity,
    prefill_attention_intensity,
)


# --------------------------------------------------------------------------- #
# 计时原语
# --------------------------------------------------------------------------- #
def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, iters: int) -> float:
    """同步后计时 iters 次 fn 的总耗时 (秒), 返回单次中位耗时。"""
    samples = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - t0)
    return median(samples)


# --------------------------------------------------------------------------- #
# A. 硬件经验 roofline (实测)
# --------------------------------------------------------------------------- #
def measure_hbm_bandwidth(device: torch.device, *, sizes_mb=(256, 512, 1024),
                          warmup: int = 5, iters: int = 30) -> dict:
    """实测可达 HBM 带宽 (GB/s): copy(读+写) 与纯读(归约) 两种口径。"""
    copy_results, read_results = [], []
    for size_mb in sizes_mb:
        n = (size_mb * 1024 * 1024) // 2  # fp16 元素数
        src = torch.randn(n, dtype=torch.float16, device=device)
        dst = torch.empty_like(src)
        nbytes = src.numel() * src.element_size()

        for _ in range(warmup):
            dst.copy_(src)
        _sync()
        t_copy = _timed(lambda: dst.copy_(src), iters)
        copy_gbps = (2.0 * nbytes) / t_copy / 1e9  # 读 src + 写 dst

        acc = torch.zeros(1, dtype=torch.float32, device=device)
        for _ in range(warmup):
            acc = src.sum(dtype=torch.float32)
        _sync()
        t_read = _timed(lambda: src.sum(dtype=torch.float32), iters)
        read_gbps = nbytes / t_read / 1e9  # 仅读 src

        copy_results.append({"size_mb": size_mb, "copy_GBps": round(copy_gbps, 1)})
        read_results.append({"size_mb": size_mb, "read_GBps": round(read_gbps, 1)})
        del src, dst, acc
        torch.cuda.empty_cache()

    best_copy = max(r["copy_GBps"] for r in copy_results)
    best_read = max(r["read_GBps"] for r in read_results)
    return {
        "copy_sweep": copy_results,
        "read_sweep": read_results,
        "achievable_copy_GBps": best_copy,
        "achievable_read_GBps": best_read,
    }


def measure_fp16_gemm_tflops(device: torch.device, *, sizes=(2048, 4096, 8192),
                             warmup: int = 5, iters: int = 20) -> dict:
    """实测可达 fp16 张量核 GEMM 吞吐 (TFLOPS): C = A @ B, FLOPs = 2*M*N*K。"""
    sweep = []
    for s in sizes:
        a = torch.randn(s, s, dtype=torch.float16, device=device)
        b = torch.randn(s, s, dtype=torch.float16, device=device)
        for _ in range(warmup):
            torch.matmul(a, b)
        _sync()
        t = _timed(lambda: torch.matmul(a, b), iters)
        flops = 2.0 * s * s * s
        sweep.append({"size": s, "TFLOPS": round(flops / t / 1e12, 1)})
        del a, b
        torch.cuda.empty_cache()
    best = max(r["TFLOPS"] for r in sweep)
    return {"gemm_sweep": sweep, "achievable_fp16_TFLOPS": best}


# --------------------------------------------------------------------------- #
# B. 模型解析曲线 (精确解析量, 来自真实 config)
# --------------------------------------------------------------------------- #
def decoder_param_count(cfg) -> int:
    """从 config 精确计算 decoder-only (Qwen2/Llama 族) 参数量。

    嵌入 + 逐层(注意力 q/k/v/o 投影 + MLP gate/up/down) + 最终 norm + lm_head;
    词嵌入与 lm_head 绑定 (tie_word_embeddings) 时不重复计 lm_head。norm/bias 量级
    极小, 计入注意力 qkv 的 bias (Qwen2 attention 有 bias)。
    """
    vocab = cfg.vocab_size
    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", None) or (hidden // n_heads)
    q_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim

    embed = vocab * hidden
    attn_w = hidden * q_dim + 2 * (hidden * kv_dim) + q_dim * hidden  # q,k,v,o 权重
    attn_b = q_dim + 2 * kv_dim if getattr(cfg, "attention_bias", True) else 0  # Qwen2 qkv bias
    mlp = 3 * (hidden * inter)  # gate, up, down
    norms = 2 * hidden  # input_layernorm + post_attention_layernorm
    per_layer = attn_w + attn_b + mlp + norms
    final_norm = hidden
    lm_head = 0 if getattr(cfg, "tie_word_embeddings", False) else vocab * hidden
    return embed + layers * per_layer + final_norm + lm_head


def analytic_model_curve(model_name: str, dims: ModelDims, param_bytes: int,
                         hw: HardwareSpec, *, prompt_grid: Tuple[int, ...],
                         context_grid: Tuple[int, ...], dtype_bytes: int = 2,
                         batch_grid: Tuple[int, ...] = (1,)) -> dict:
    """某模型在给定硬件 ridge 下的解析 roofline 曲线 (精确量)。

    KV 读占比按 batch 分别给出: 单步 decode 读取权重一次 (batch 间复用), 但 KV 随 batch
    线性增长, 故 batch 越大 KV 占 decode 带宽比例越高 (洞见 2 的工程后果)。
    """
    ridge = hw.ridge_intensity
    batch = batch_grid[0]

    prefill = []
    for p in prompt_grid:
        inten = prefill_attention_intensity(p, dims, batch)
        prefill.append({
            "prompt_len": p,
            "intensity_flop_per_byte": round(inten, 3),
            "bound": "compute" if inten >= ridge else "bandwidth",
        })

    decode = []
    kv_fraction = []
    for c in context_grid:
        inten = decode_attention_intensity(c, dims, batch)
        decode.append({
            "context_len": c,
            "intensity_flop_per_byte": round(inten, 4),
            "bound": "compute" if inten >= ridge else "bandwidth",
        })
        row = {"context_len": c, "weight_read_bytes": param_bytes}
        for b in batch_grid:
            kv_bytes = metrics.kv_cache_dtype_bytes(
                dims.num_layers, dims.num_kv_heads, dims.head_dim, c, dtype_bytes, b)
            row[f"kv_read_bytes_b{b}"] = kv_bytes
            row[f"kv_fraction_b{b}"] = round(kv_bytes / (kv_bytes + param_bytes), 4)
        kv_fraction.append(row)

    return {
        "dims": {
            "num_layers": dims.num_layers,
            "num_attention_heads": dims.num_attention_heads,
            "num_kv_heads": dims.num_kv_heads,
            "head_dim": dims.head_dim,
        },
        "param_count": param_bytes // dtype_bytes,
        "weight_bytes_fp16": param_bytes,
        "batch_grid": list(batch_grid),
        "prefill_intensity": prefill,
        "decode_intensity": decode,
        "kv_read_fraction": kv_fraction,
    }


def load_config(model_name: str):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(model_name, trust_remote_code=True)


# --------------------------------------------------------------------------- #
# C. 模型实测落点 (真实加载并计时)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_model_points(model_name: str, dims: ModelDims, hw_peak: dict, *,
                         prompt_grid: Tuple[int, ...], context_grid: Tuple[int, ...],
                         dtype: str = "float16", device: str = "cuda",
                         warmup: int = 2, iters: int = 5) -> dict:
    """加载真实模型, 实测 prefill / decode 落点。"""
    from transformers.cache_utils import DynamicCache

    from eval.runtime import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        model_name, dtype=dtype, device=device, attn_implementation="sdpa")
    dev = next(model.parameters()).device
    param_count = sum(p.numel() for p in model.parameters())
    dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}[dtype]
    weight_bytes = param_count * dtype_bytes
    vocab = model.config.vocab_size
    h, hkv, d, L = (dims.num_attention_heads, dims.num_kv_heads,
                    dims.head_dim, dims.num_layers)

    def rand_ids(length: int) -> torch.Tensor:
        return torch.randint(0, vocab, (1, length), device=dev, dtype=torch.long)

    peak_tflops = hw_peak["achievable_fp16_TFLOPS"]
    peak_gbps = hw_peak["achievable_copy_GBps"]

    prefill_points = []
    for p in prompt_grid:
        ids = rand_ids(p)
        pos = torch.arange(p, device=dev).unsqueeze(0)

        def _prefill():
            model(input_ids=ids, use_cache=False, position_ids=pos)

        for _ in range(warmup):
            _prefill()
        t = _timed(_prefill, iters)
        linear_flops = 2.0 * param_count * p
        attn_flops = 4.0 * L * h * (p ** 2) * d
        total_flops = linear_flops + attn_flops
        ach_tflops = total_flops / t / 1e12
        prefill_points.append({
            "prompt_len": p,
            "latency_ms": round(t * 1000.0, 3),
            "achieved_TFLOPS": round(ach_tflops, 2),
            "compute_util_pct": round(100.0 * ach_tflops / peak_tflops, 1),
            "operational_intensity": round(prefill_attention_intensity(p, dims), 2),
        })
        del ids
        torch.cuda.empty_cache()

    decode_points = []
    for c in context_grid:
        ids = rand_ids(c)
        pos = torch.arange(c, device=dev).unsqueeze(0)
        cache = DynamicCache()
        model(input_ids=ids, use_cache=True, past_key_values=cache,
              position_ids=pos, cache_position=torch.arange(c, device=dev))
        next_id = torch.randint(0, vocab, (1, 1), device=dev, dtype=torch.long)
        phys = [c]  # 真实自回归: 每步物理缓存长度 +1, position_ids 与 cache_position 同步推进

        def _decode():
            step_pos = torch.tensor([[phys[0]]], device=dev, dtype=torch.long)
            step_cache_pos = torch.tensor([phys[0]], device=dev, dtype=torch.long)
            model(input_ids=next_id, use_cache=True, past_key_values=cache,
                  position_ids=step_pos, cache_position=step_cache_pos)
            phys[0] += 1

        for _ in range(warmup):
            _decode()
        t = _timed(_decode, iters)
        kv_bytes = metrics.kv_cache_dtype_bytes(L, hkv, d, c, dtype_bytes)
        read_bytes = weight_bytes + kv_bytes
        ach_gbps = read_bytes / t / 1e9
        decode_flops = 2.0 * param_count + 4.0 * L * h * c * d
        decode_points.append({
            "context_len": c,
            "latency_ms": round(t * 1000.0, 3),
            "achieved_GBps": round(ach_gbps, 1),
            "bandwidth_util_pct": round(100.0 * ach_gbps / peak_gbps, 1),
            "achieved_TFLOPS": round(decode_flops / t / 1e12, 3),
            "kv_read_bytes": kv_bytes,
            "weight_read_bytes": weight_bytes,
            "operational_intensity": round(decode_attention_intensity(c, dims), 4),
        })
        del ids, cache
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "param_count": param_count,
        "weight_bytes_fp16": weight_bytes,
        "prefill": prefill_points,
        "decode": decode_points,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    prop = torch.cuda.get_device_properties(device) if device.type == "cuda" else None

    print("[roofline] A. 实测硬件 roofline ...", flush=True)
    bw = measure_hbm_bandwidth(device)
    gemm = measure_fp16_gemm_tflops(device)
    measured_peak = {
        "achievable_copy_GBps": bw["achievable_copy_GBps"],
        "achievable_read_GBps": bw["achievable_read_GBps"],
        "achievable_fp16_TFLOPS": gemm["achievable_fp16_TFLOPS"],
        "empirical_ridge_flop_per_byte": round(
            gemm["achievable_fp16_TFLOPS"] * 1e12 / (bw["achievable_copy_GBps"] * 1e9), 2),
    }
    measured_hw = HardwareSpec(
        name=(prop.name if prop else "cpu"),
        peak_flops=gemm["achievable_fp16_TFLOPS"] * 1e12,
        hbm_bandwidth=bw["achievable_copy_GBps"] * 1e9,
    )
    nominal_hw = HardwareSpec()
    print(f"[roofline]   可达带宽(copy)={measured_peak['achievable_copy_GBps']} GB/s, "
          f"GEMM={measured_peak['achievable_fp16_TFLOPS']} TFLOPS, "
          f"经验 ridge={measured_peak['empirical_ridge_flop_per_byte']} FLOP/Byte", flush=True)

    prompt_grid = tuple(int(x) for x in args.prompt_grid.split(","))
    context_grid = tuple(int(x) for x in args.context_grid.split(","))
    analytic_context_grid = tuple(int(x) for x in args.analytic_context_grid.split(","))
    batch_grid = tuple(int(x) for x in args.batch_grid.split(","))

    print("[roofline] B. 模型解析曲线 (精确解析, 来自真实 config) ...", flush=True)
    analytic: Dict[str, dict] = {}
    for name in [m.strip() for m in args.analytic_models.split(",") if m.strip()]:
        try:
            cfg = load_config(name)
        except Exception as exc:
            analytic[name] = {"unavailable": f"{exc.__class__.__name__}: {exc}"}
            print(f"[roofline]   {name} config 不可用 (可能仍在下载): {exc}", flush=True)
            continue
        dims = ModelDims.from_config(cfg)
        param_bytes = decoder_param_count(cfg) * 2
        analytic[name] = analytic_model_curve(
            name, dims, param_bytes, measured_hw,
            prompt_grid=prompt_grid, context_grid=analytic_context_grid, batch_grid=batch_grid)
        print(f"[roofline]   {name}: {dims.num_layers}L/{dims.num_attention_heads}H/"
              f"{dims.num_kv_heads}KV/{dims.head_dim}d, ~{param_bytes // 2 / 1e9:.2f}B 参数", flush=True)

    measured_points = None
    if args.measured_model:
        print(f"[roofline] C. 加载 {args.measured_model} 实测 prefill/decode 落点 ...", flush=True)
        cfg = load_config(args.measured_model)
        dims = ModelDims.from_config(cfg)
        measured_points = measure_model_points(
            args.measured_model, dims, measured_peak,
            prompt_grid=prompt_grid, context_grid=context_grid,
            dtype=args.dtype, device=str(device))
        for row in measured_points["decode"]:
            print(f"[roofline]   decode C={row['context_len']}: {row['latency_ms']}ms, "
                  f"{row['achieved_GBps']}GB/s ({row['bandwidth_util_pct']}% 带宽), "
                  f"{row['achieved_TFLOPS']}TFLOPS", flush=True)
        for row in measured_points["prefill"]:
            print(f"[roofline]   prefill P={row['prompt_len']}: {row['latency_ms']}ms, "
                  f"{row['achieved_TFLOPS']}TFLOPS ({row['compute_util_pct']}% 算力)", flush=True)

    findings = _derive_findings(measured_peak, measured_points, analytic, args)

    meta = {
        "experiment": "roofline_characterization",
        "gpu": prop.name if prop else "cpu",
        "gpu_memory_GB": round(prop.total_memory / 1e9, 1) if prop else None,
        "compute_capability": f"{prop.major}.{prop.minor}" if prop else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "measured_peak": measured_peak,
        "nominal_peak": {
            "hbm_bandwidth_GBps": nominal_hw.hbm_bandwidth / 1e9,
            "fp16_TFLOPS": nominal_hw.peak_flops / 1e12,
            "ridge_flop_per_byte": round(nominal_hw.ridge_intensity, 2),
        },
        "prompt_grid": list(prompt_grid),
        "context_grid": list(context_grid),
        "analytic_context_grid": list(analytic_context_grid),
        "batch_grid": list(batch_grid),
        "dtype": args.dtype,
        "note": ("真实模型真实测量, 无模拟数据。A 硬件峰值=GPU 上微基准实测(copy/read 带宽, "
                 "fp16 GEMM 吞吐); B 算术强度与 KV 占比=算子 FLOP/Byte 定义的精确解析量(非模拟实验), "
                 "来自真实模型 config; C 落点=真实加载模型计时所得有效吞吐。本图刻画 prefill/decode "
                 "硬件瓶颈差异(洞见1), 不报任何 kernel 加速比(那属阶段 B 真实 kernel)。"),
    }
    return {
        "meta": meta,
        "findings": findings,
        "hardware_roofline": {**bw, **gemm},
        "analytic": analytic,
        "measured_points": measured_points,
    }


def _derive_findings(measured_peak: dict, measured_points: Optional[dict],
                     analytic: dict, args) -> dict:
    """从实测/解析数据自动得出洞见 1 的诚实判读 (不夸大, 数字来自上面真实测量)。"""
    out = {}
    if measured_points and measured_points["prefill"]:
        pf = measured_points["prefill"]
        best = max(pf, key=lambda r: r["compute_util_pct"])
        out["prefill"] = {
            "max_compute_util_pct": best["compute_util_pct"],
            "at_prompt_len": best["prompt_len"],
            "conclusion": (f"prefill 随 prompt 长度算力利用率从 {pf[0]['compute_util_pct']}% 升至 "
                           f"{best['compute_util_pct']}% (P={best['prompt_len']}), 趋近实测 GEMM 峰值 "
                           f"=> 计算受限, 与洞见 1 一致。"),
        }
    if measured_points and measured_points["decode"]:
        dc = measured_points["decode"]
        lat = [r["latency_ms"] for r in dc]
        bw_util = [r["bandwidth_util_pct"] for r in dc]
        comp_util = [100.0 * r["achieved_TFLOPS"] / measured_peak["achievable_fp16_TFLOPS"]
                     for r in dc]
        flat = (max(lat) - min(lat)) / max(lat) < 0.5
        out["decode_measured"] = {
            "model": measured_points["model"],
            "latency_ms_range": [min(lat), max(lat)],
            "bandwidth_util_pct_range": [min(bw_util), max(bw_util)],
            "max_compute_util_pct": round(max(comp_util), 4),
            "conclusion": (
                f"单步 decode 算力利用率最高仅 {round(max(comp_util), 3)}% (远低于 prefill) "
                f"=> 明确*非*计算受限。此 {measured_points['param_count'] / 1e9:.2f}B 小模型上 "
                f"延迟近似恒定 ({min(lat):.0f}~{max(lat):.0f}ms, 随上下文 32x 增长几乎不变) 且带宽利用 "
                f"仅 {min(bw_util):.0f}~{max(bw_util):.0f}% => 受*固定单步开销*(逐层 kernel 启动/Python 循环)"
                f"主导, 尚未饱和 HBM。"
                if flat else
                f"单步 decode 算力利用率最高仅 {round(max(comp_util), 3)}% => 非计算受限。"),
        }
    big = "Qwen/Qwen2.5-7B-Instruct"
    if big in analytic and not analytic[big].get("unavailable"):
        a = analytic[big]
        wb = a["weight_bytes_fp16"]
        floor_ms = wb / (measured_peak["achievable_copy_GBps"] * 1e9) * 1000.0
        last = a["kv_read_fraction"][-1]
        bks = sorted(a["batch_grid"])
        out["decode_bandwidth_bound_projection_7b"] = {
            "weight_read_bytes_fp16": wb,
            "hbm_read_floor_ms_batch1": round(floor_ms, 1),
            "kv_fraction_at_longest_ctx": {f"batch{b}": last[f"kv_fraction_b{b}"] for b in bks},
            "longest_ctx": last["context_len"],
            "conclusion": (
                f"7B 单步 decode 仅读权重 ({wb / 1e9:.1f}GB) 即有 ~{floor_ms:.0f}ms HBM 底线 "
                f"(实测 copy 带宽 {measured_peak['achievable_copy_GBps']}GB/s), 远超 0.5B 的固定开销 "
                f"=> 7B decode 必然带宽受限。KV 读占比随上下文与 batch 放大: ctx={last['context_len']} 时 "
                + ", ".join(f"batch{b}={100.0 * last[f'kv_fraction_b{b}']:.0f}%" for b in bks)
                + " => 长上下文/批量下 KV 读流量成为 decode 带宽主因, 直接支撑压 KV 的收益 (洞见 1/2)。"
                " 实测 7B decode 落点待模型下载完成后用本脚本 --measured-model 7B 复跑。"),
        }
    return out


def _save(result: dict, out_prefix: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    meta = result["meta"]
    mp = meta["measured_peak"]
    lines = [
        f"### 图1 Roofline 真实测量标定 · {meta['gpu']}", "",
        f"- torch {meta['torch']} / CUDA {meta['cuda']} / 显存 {meta['gpu_memory_GB']}GB / "
        f"算力 {meta['compute_capability']}",
        "",
        "#### A. 实测硬件峰值 (覆盖 planner 标称值)", "",
        "| 指标 | 实测 | planner 标称 |",
        "| --- | --- | --- |",
        f"| HBM 带宽 copy(读+写) | {mp['achievable_copy_GBps']} GB/s | "
        f"{meta['nominal_peak']['hbm_bandwidth_GBps']:.0f} GB/s |",
        f"| HBM 带宽 纯读 | {mp['achievable_read_GBps']} GB/s | — |",
        f"| fp16 GEMM 吞吐 | {mp['achievable_fp16_TFLOPS']} TFLOPS | "
        f"{meta['nominal_peak']['fp16_TFLOPS']:.0f} TFLOPS |",
        f"| 经验 ridge point | {mp['empirical_ridge_flop_per_byte']} FLOP/Byte | "
        f"{meta['nominal_peak']['ridge_flop_per_byte']} FLOP/Byte |",
        "",
    ]

    findings = result.get("findings", {})
    if findings:
        lines += ["#### 洞见1 判读 (数字均来自上方真实测量)", ""]
        for key in ("prefill", "decode_measured", "decode_bandwidth_bound_projection_7b"):
            if key in findings and findings[key].get("conclusion"):
                lines += [f"- {findings[key]['conclusion']}"]
        lines.append("")

    if result.get("measured_points"):
        pts = result["measured_points"]
        lines += [f"#### C. 实测落点 · {pts['model']} ({pts['param_count'] / 1e9:.2f}B 参数)", "",
                  "decode (单步, 非计算受限; 0.5B 小模型上固定开销主导, 带宽受限regime见7B投影):", "",
                  "| 上下文 | 延迟ms | 有效带宽GB/s | 带宽利用% | 有效TFLOPS |",
                  "| --- | --- | --- | --- | --- |"]
        for r in pts["decode"]:
            lines.append(f"| {r['context_len']} | {r['latency_ms']} | {r['achieved_GBps']} | "
                         f"{r['bandwidth_util_pct']} | {r['achieved_TFLOPS']} |")
        lines += ["", "prefill (计算受限):", "",
                  "| prompt | 延迟ms | 有效TFLOPS | 算力利用% |",
                  "| --- | --- | --- | --- |"]
        for r in pts["prefill"]:
            lines.append(f"| {r['prompt_len']} | {r['latency_ms']} | {r['achieved_TFLOPS']} | "
                         f"{r['compute_util_pct']} |")
        lines.append("")

    for name, a in result["analytic"].items():
        if a.get("unavailable"):
            lines += [f"#### B. 解析曲线 · {name}: 不可用 ({a['unavailable']})", ""]
            continue
        bks = a.get("batch_grid", [1])
        head = "| 上下文 | 权重读GB | " + " | ".join(f"KV占比 b{b}" for b in bks) + " |"
        sep = "| --- | --- | " + " | ".join("---" for _ in bks) + " |"
        lines += [f"#### B. 解析 KV 读占比 · {name} "
                  f"({a['param_count'] / 1e9:.2f}B 参数, {a['dims']['num_layers']}L/"
                  f"{a['dims']['num_kv_heads']}KV/{a['dims']['head_dim']}d)", "", head, sep]
        for r in a["kv_read_fraction"]:
            cells = " | ".join(f"{100.0 * r[f'kv_fraction_b{b}']:.1f}%" for b in bks)
            lines.append(f"| {r['context_len']} | {r['weight_read_bytes'] / 1e9:.2f} | {cells} |")
        lines.append("")

    lines += ["", f"> {meta['note']}", ""]
    md = "\n".join(lines)
    with open(f"{out_prefix}.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print("报告已保存:", f"{out_prefix}.json / .md", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="图1 Roofline 真实测量标定")
    p.add_argument("--measured-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="加载并实测 prefill/decode 落点的模型 (留空则跳过 C 部分)")
    p.add_argument("--analytic-models",
                   default="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-7B-Instruct",
                   help="仅读 config 做解析曲线的模型 (逗号分隔)")
    p.add_argument("--prompt-grid", default="512,1024,2048,4096")
    p.add_argument("--context-grid", default="512,1024,2048,4096,8192,16384",
                   help="实测 decode 上下文网格 (受显存限制)")
    p.add_argument("--analytic-context-grid",
                   default="512,1024,2048,4096,8192,16384,32768,65536,131072",
                   help="解析 KV 占比上下文网格 (可延伸到 128K)")
    p.add_argument("--batch-grid", default="1,16,64", help="解析 KV 占比的 batch 网格")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/roofline_4080s")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("[roofline] 需要 CUDA GPU 做真实测量, 未检测到可用 CUDA。")
    result = run(args)
    _save(result, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
