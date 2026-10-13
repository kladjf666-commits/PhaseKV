"""隔离测试真实 PhaseKVCache 的 prompt 位宽与 token 保留率。

两条轴：保留全部 prompt 时测 2/4/8 bit；固定 8 bit 时测
100%/75%/50%/25% 保留率。另附当前默认的 2 bit + 50% 作为参照。
只比较确定性贪心回答和缓存大小，不把首次 JIT 编译时间当作速度结果。
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_qwen2_phasekv import (
    builtin_cases, full_sequence_token_match, normalize_cases, run_case,
)
import qwen2
from src.phasekv import install_qwen2_phasekv_attention


CONFIGS = (
    ("2bit_100pct", 2, 1.0),
    ("4bit_100pct", 4, 1.0),
    ("8bit_100pct", 8, 1.0),
    ("8bit_75pct", 8, 0.75),
    ("8bit_50pct", 8, 0.5),
    ("8bit_25pct", 8, 0.25),
    ("2bit_50pct_default", 2, 0.5),
)


def summarize(rows: list[dict], full_rows: list[dict]) -> dict:
    labeled = [row["answer_hit"] for row in rows if row["answer_hit"] is not None]
    return {
        "hits": sum(labeled), "labeled_cases": len(labeled),
        "median_kv_mib": statistics.median(row["kv_bytes"] for row in rows) / 2**20,
        "median_token_match_to_full": statistics.median(
            full_sequence_token_match(row["generated_ids"], baseline["generated_ids"])
            for row, baseline in zip(rows, full_rows)
        ),
    }


def write_report(prefix: Path, payload: dict) -> tuple[Path, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Qwen2 PhaseKV Prompt 消融实验", "",
        "固定 cold=4 bit、decode 保留率=50%、recent_window=32、贪心解码。",
        "命中只表示回答包含预设答案；内置六题中五题有标准答案。",
        "本实验没有逐配置预热，计时不用于速度比较。", "",
        "| 配置 | Prompt 位宽 | Prompt 保留率 | 命中 | KV 中位数 MiB | 与 full 的 token 一致率中位数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    full = payload["full_summary"]
    lines.append(f"| full | BF16 | 100% | {full['hits']}/{full['labeled_cases']} | {full['median_kv_mib']:.2f} | 1.000 |")
    for name, bits, keep in CONFIGS:
        result = payload["variants"][name]
        s = result["summary"]
        lines.append(f"| {name} | {bits} | {keep:.0%} | {s['hits']}/{s['labeled_cases']} | "
                     f"{s['median_kv_mib']:.2f} | {s['median_token_match_to_full']:.3f} |")
    lines += ["", "## 逐题命中", "",
              "| 题目 | full | " + " | ".join(name for name, _, _ in CONFIGS) + " |",
              "|---|---:|" + "---:|" * len(CONFIGS)]
    label = lambda value: "—" if value is None else ("✓" if value else "✗")
    for index, case in enumerate(payload["cases"]):
        values = [label(payload["variants"][name]["rows"][index]["answer_hit"])
                  for name, _, _ in CONFIGS]
        lines.append(f"| {case['id']} | {label(payload['full_rows'][index]['answer_hit'])} | "
                     + " | ".join(values) + " |")
    lines += ["", "完整回答、实际保留的 prompt token 数和逐题 KV 大小见 JSON 文件。", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="PhaseKV prompt 位宽与保留率消融")
    parser.add_argument("--model", default=qwen2.MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--out", default="results/qwen2_phasekv_prompt_ablation")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens 必须为正")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("此实验需要支持 BF16 的 CUDA GPU")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    install_qwen2_phasekv_attention(model)
    cases = normalize_cases(builtin_cases())
    full_rows: list[dict] = []
    variants = {name: {"prompt_bits": bits, "prefill_keep_ratio": keep, "rows": []}
                for name, bits, keep in CONFIGS}
    for case in cases:
        gc.collect()
        torch.cuda.empty_cache()
        baseline = run_case(model, tokenizer, case, method="full",
                            max_new_tokens=args.max_new_tokens, sample=False, seed=1234,
                            phase_kwargs={})
        full_rows.append(baseline)
        print(f"[{case['id']}] full: {baseline['answer_hit']}", flush=True)
        for name, bits, keep in CONFIGS:
            gc.collect()
            torch.cuda.empty_cache()
            row = run_case(model, tokenizer, case, method="phasekv",
                           max_new_tokens=args.max_new_tokens, sample=False, seed=1234,
                           phase_kwargs={
                               "prompt_bits": bits,
                               "prefill_keep_ratio": keep,
                               "cold_bits": 4,
                               "decode_keep_ratio": 0.5,
                               "recent_window": 32,
                           })
            variants[name]["rows"].append(row)
            print(f"[{case['id']}] {name}: {row['answer_hit']} "
                  f"prompt={row['retained_prompt_tokens']}/{row['prompt_tokens']} "
                  f"cold={row['cold_tokens']}", flush=True)

    for variant in variants.values():
        variant["summary"] = summarize(variant["rows"], full_rows)
    payload = {
        "meta": {"model": args.model, "max_new_tokens": args.max_new_tokens,
                 "mode": "greedy", "cold_bits": 4, "decode_keep_ratio": 0.5,
                 "recent_window": 32, "gpu": torch.cuda.get_device_name(0)},
        "cases": cases, "full_rows": full_rows,
        "full_summary": summarize(full_rows, full_rows), "variants": variants,
    }
    json_path, md_path = write_report(Path(args.out), payload)
    print(f"报告已保存：{json_path}、{md_path}", flush=True)


if __name__ == "__main__":
    main()
