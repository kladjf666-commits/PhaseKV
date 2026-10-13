"""命令行入口: 运行 KV 压缩评估并产出报告 (QA / CH-3 维护)。

示例:
  # 用内置离线语料做 smoke (小模型, 不联网精度评测)
  python -m eval.run_benchmark --model Qwen/Qwen2.5-0.5B-Instruct \
      --preset core --prompt-tokens 512 --max-new-tokens 48 --ppl-source builtin \
      --out ../results/smoke

  # demo 主力模型 + wikitext 困惑度
  HF_ENDPOINT=https://hf-mirror.com python -m eval.run_benchmark \
      --model Qwen/Qwen2.5-7B-Instruct --preset full --prompt-tokens 4096 \
      --max-new-tokens 128 --ppl-source wikitext --out ../results/qwen7b
"""
from __future__ import annotations

import argparse
import json
import os

from . import datasets, report
from .compressors import available
from .runner import BenchmarkConfig, run_benchmark


PRESETS = {
    "core": [
        {"method": "full"},
        {"method": "recent", "recent_window": 256},
        {"method": "kvquant", "bits": 4},
    ],
    "full": [
        {"method": "full"},
        {"method": "recent", "recent_window": 256},
        {"method": "streaming", "sink_tokens": 4, "recent_window": 256},
        {"method": "kvquant", "bits": 8},
        {"method": "kvquant", "bits": 4},
    ],
    "attention": [
        {"method": "full"},
        {"method": "streaming", "sink_tokens": 4, "recent_window": 256},
        {"method": "h2o", "prefill_keep_ratio": 0.3, "sink_tokens": 4, "recent_window": 128},
        {"method": "hetero", "prefill_keep_ratio": 0.3, "sink_tokens": 4, "recent_window": 128},
    ],
}


def _build_prompt(args, tokenizer_text: str) -> str:
    if args.prompt is not None:
        return args.prompt
    base = datasets.BUILTIN_SAMPLE_TEXT
    if args.prompt_tokens is None:
        return base
    # 近似按字符放大到目标 token 数 (避免提前加载 tokenizer)
    return (base * ((args.prompt_tokens // 64) + 1))[: args.prompt_tokens * 6]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="KV Cache 压缩评估")
    p.add_argument("--model", required=True)
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="core")
    p.add_argument("--methods-json", default=None, help="覆盖 preset 的方法列表 (JSON)")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--ppl-source", choices=["none", "builtin", "wikitext"], default="builtin")
    p.add_argument("--ppl-chunk-size", type=int, default=512)
    p.add_argument("--out", default="../results/benchmark")
    p.add_argument("--list-methods", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list_methods:
        print("可用方法:", available())
        return

    methods = json.loads(args.methods_json) if args.methods_json else PRESETS[args.preset]

    ppl_text = None
    if args.ppl_source == "builtin":
        ppl_text = datasets.BUILTIN_SAMPLE_TEXT
    elif args.ppl_source == "wikitext":
        ppl_text = datasets.load_wikitext(split="test", n_chars=20000)

    cfg = BenchmarkConfig(
        model_name=args.model,
        methods=methods,
        prompt=_build_prompt(args, ""),
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        device=args.device,
        ppl_text=ppl_text,
        ppl_chunk_size=args.ppl_chunk_size,
        extra_meta={"preset": args.preset if not args.methods_json else "custom"},
    )

    result = run_benchmark(cfg)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    paths = report.save_all(
        result["rows"], args.out,
        title=f"KV 压缩评估 · {args.model}", meta=result["meta"])

    print(report.to_markdown(result["rows"], title=f"KV 压缩评估 · {args.model}"))
    print("\n报告已保存:", paths)


if __name__ == "__main__":
    main()
