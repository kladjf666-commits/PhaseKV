"""accuracy-per-byte 拆分实验 (QA / CH-3 维护) —— 支撑 idea 文档第 2 节图2。

在 *同一* 注意力 heavy-hitter 选择机制下 (eval.compressors.segment_ablation),
分别对 prompt 段与 generated 段独立做预算扫描, 度量"每字节精度"随段的差异:
  - 曲线 A 『只压 prompt-KV』: 固定保留全部 generated, 扫描 prompt 保留比例。
  - 曲线 B 『只压 generated-KV』: 固定保留全部 prompt, 扫描 generated 保留比例。

精度信号 (smoke 尺度, 离线无联网): token_match_vs_full —— 压缩输出相对 full-cache 输出
的逐位置保真度 (越高越好), 模型/任务无关, 适合小模型快速扫描。M1 (Qwen2.5-7B +
LongBench 子集) 将以任务分数替换, 行结构保持一致。

输出 results/<out>.{md,csv,json}; JSON 为 {meta, rows[]}, 每行除标准指标外含额外字段
segment / keep_ratio / kept_cache_bytes / budget_tokens, 供前端直接画两条曲线。

在 release/kvcomp/ 下运行:
  python -m eval.accuracy_per_byte --model Qwen/Qwen2.5-0.5B-Instruct \
      --prompt-tokens 768 --max-new-tokens 128 --out results/accuracy_per_byte
"""
from __future__ import annotations

import argparse
import os
from typing import List

from . import datasets, metrics, report, runtime
from .compressors import build_compressor


def _kv_dims(model_config):
    num_layers = getattr(model_config, "num_hidden_layers")
    num_heads = getattr(model_config, "num_attention_heads")
    num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        head_dim = model_config.hidden_size // num_heads
    return num_layers, num_kv_heads, head_dim


def run_accuracy_per_byte(model_name: str, *, prompt_tokens: int = 768,
                          max_new_tokens: int = 128,
                          segments: List[str] = ("prompt", "generated"),
                          grid: List[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
                          recent_window: int = 8, dtype: str = "float16",
                          device: str = "cuda", seed: int = 1234) -> dict:
    model, tokenizer = runtime.load_model_and_tokenizer(
        model_name, dtype=dtype, device=device, attn_implementation="eager")
    num_layers, num_kv_heads, head_dim = _kv_dims(model.config)

    prompt = datasets.build_long_prompt_from_text(
        datasets.BUILTIN_SAMPLE_TEXT, tokenizer, prompt_tokens)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    full_gen = runtime.generate_with_compression(
        model, tokenizer, input_ids, build_compressor("full"),
        max_new_tokens=max_new_tokens, greedy=True, seed=seed)
    ref_ids = full_gen["generated_ids"]

    rows = []
    for segment in segments:
        ratio_key = "prefill_keep_ratio" if segment == "prompt" else "decode_keep_ratio"
        for ratio in grid:
            params = {"segment": segment, "recent_window": recent_window, ratio_key: float(ratio)}
            compressor = build_compressor("segment_ablation", **params)
            gen = runtime.generate_with_compression(
                model, tokenizer, input_ids, compressor,
                max_new_tokens=max_new_tokens, greedy=True, seed=seed)
            total_tokens = prompt_len + gen["latency"].num_generated
            full_ref_bytes = metrics.kv_cache_dtype_bytes(
                num_layers, num_kv_heads, head_dim, total_tokens, dtype_bytes=2)
            method_bytes = gen["final_cache_bytes"]
            budget = (int(round(prompt_len * ratio)) if segment == "prompt"
                      else max(recent_window, int(round(gen["latency"].num_generated * ratio))))
            rows.append({
                "method": f"{segment}@{ratio:g}",
                "segment": segment,
                "keep_ratio": float(ratio),
                "budget_tokens": budget,
                "kv_saving_pct": metrics.memory_saving_ratio(method_bytes, full_ref_bytes) * 100.0,
                "compression_ratio": metrics.compression_ratio(method_bytes, full_ref_bytes),
                "kept_cache_bytes": method_bytes,
                "full_ref_cache_bytes": full_ref_bytes,
                "peak_gpu_mb": gen["peak_gpu_bytes"] / 1e6,
                "ttft_ms": gen["latency"].ttft_s * 1000.0,
                "tpot_ms": gen["latency"].tpot_s * 1000.0,
                "decode_tps": gen["latency"].decode_throughput,
                "token_match_vs_full": metrics.token_match_rate(gen["generated_ids"], ref_ids),
                "perplexity": None,
                "num_generated": gen["latency"].num_generated,
            })

    meta = {
        "model": model_name,
        "experiment": "accuracy_per_byte",
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "segments": list(segments),
        "grid": [float(r) for r in grid],
        "recent_window": recent_window,
        "kv_dims": {"num_layers": num_layers, "num_kv_heads": num_kv_heads, "head_dim": head_dim},
        "score_metric": "token_match_vs_full",
        "score_direction": "higher_better",
        "note": ("smoke 尺度(0.5B + 内置离线语料); 精度信号=对 full-cache 输出的逐位置保真度 "
                 "token_match_vs_full(越高越好), 模型/任务无关。M1 将在 Qwen2.5-7B + LongBench "
                 "子集上以任务分数替换 score, 行结构(segment/keep_ratio/kept_cache_bytes)保持一致。"),
    }
    return {"rows": rows, "meta": meta}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="accuracy-per-byte 拆分实验 (图2)")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-tokens", type=int, default=768)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--segments", default="prompt,generated")
    p.add_argument("--grid", default="0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--recent-window", type=int, default=8)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/accuracy_per_byte")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    segments = [s.strip() for s in args.segments.split(",") if s.strip()]
    grid = [float(x) for x in args.grid.split(",") if x.strip()]
    result = run_accuracy_per_byte(
        args.model, prompt_tokens=args.prompt_tokens, max_new_tokens=args.max_new_tokens,
        segments=segments, grid=grid, recent_window=args.recent_window,
        dtype=args.dtype, device=args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    paths = report.save_all(
        result["rows"], args.out,
        title=f"accuracy-per-byte 拆分 · {args.model}", meta=result["meta"])
    print(report.to_markdown(result["rows"], title=f"accuracy-per-byte 拆分 · {args.model}"))
    print("\n报告已保存:", paths)


if __name__ == "__main__":
    main()
