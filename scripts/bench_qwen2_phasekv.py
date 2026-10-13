"""真实 PhaseKVCache 与普通 DynamicCache 的端到端 Qwen2 对比。

运行: .venv/bin/python scripts/bench_qwen2_phasekv.py
自定义题目: --cases-jsonl path/to/cases.jsonl
JSONL 每行: {"id":"...","messages":[{"role":"user","content":"..."}],"answers":["..."]}
也可用 "prompt" 代替 "messages"；answers 可省略，此时只记录输出与性能。
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

import qwen2
from eval.datasets import make_needle_haystack
from eval.metrics import first_divergence_index
from src.phasekv import install_qwen2_phasekv_attention


def builtin_cases() -> list[dict]:
    cases = [
        {"id": "arithmetic", "prompt": "17 加 25 等于多少？只回答数字。", "answers": ["42"]},
        {"id": "extract", "prompt": "记录：订单 A 的验证码是 739284，订单 B 的验证码是 815603。订单 A 的验证码是什么？只回答数字。", "answers": ["739284"]},
        {"id": "history", "messages": [
            {"role": "user", "content": "请记住我的代号是蓝鲸七号。"},
            {"role": "assistant", "content": "好的，我会记住。"},
            {"role": "user", "content": "我的代号是什么？只回答代号。"},
        ], "answers": ["蓝鲸七号"]},
        {"id": "long_followup", "messages": [
            {"role": "user", "content": "为我介绍一下大模型。"},
            {"role": "assistant", "content": "您好，有什么问题我可以帮助您解答吗？"},
            {"role": "user", "content": "请详细介绍大模型的原理、用途和局限，至少写三段。"},
        ], "answers": []},
    ]
    for index, item in enumerate(make_needle_haystack(2, haystack_sentences=10, seed=2026)):
        cases.append({
            "id": f"needle_{index}",
            "prompt": f"{item['context']}\n\n{item['question']}",
            "answers": item["answers"],
        })
    return cases


def normalize_cases(raw_cases: list[dict]) -> list[dict]:
    cases = []
    seen_ids = set()
    for index, raw in enumerate(raw_cases):
        case_id = str(raw.get("id", f"case_{index}"))
        if case_id in seen_ids:
            raise ValueError(f"题目 id 重复: {case_id}")
        seen_ids.add(case_id)
        if "messages" in raw:
            messages = raw["messages"]
        elif "prompt" in raw:
            messages = [{"role": "user", "content": raw["prompt"]}]
        else:
            raise ValueError(f"{case_id} 缺少 messages 或 prompt")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{case_id} 的 messages 必须是非空列表")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"{case_id} 的消息 role 无效")
            if not isinstance(message.get("content"), str):
                raise ValueError(f"{case_id} 的消息 content 必须是文本")
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": "You are a helpful assistant."}, *messages]
        answers = raw.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        if not isinstance(answers, list) or not all(isinstance(a, str) and a for a in answers):
            raise ValueError(f"{case_id} 的 answers 必须是非空字符串列表或省略")
        cases.append({"id": case_id, "messages": messages, "answers": answers})
    if not cases:
        raise ValueError("题目列表为空")
    return cases


def load_cases(path: str | None) -> list[dict]:
    if path is None:
        return normalize_cases(builtin_cases())
    with open(path, encoding="utf-8") as handle:
        raw = [json.loads(line) for line in handle if line.strip()]
    return normalize_cases(raw)


def answer_hit(text: str, answers: list[str]) -> bool | None:
    if not answers:
        return None
    lowered = text.casefold()
    for answer in answers:
        target = answer.casefold()
        if target.isdecimal():
            if re.search(rf"(?<!\d){re.escape(target)}(?!\d)", lowered):
                return True
        elif target in lowered:
            return True
    return False


def repeated_fourgram_fraction(token_ids: list[int]) -> float:
    if len(token_ids) < 4:
        return 0.0
    grams = [tuple(token_ids[i:i + 4]) for i in range(len(token_ids) - 3)]
    return (len(grams) - len(set(grams))) / len(grams)


def full_sequence_token_match(pred_ids: list[int], ref_ids: list[int]) -> float:
    """把提前结束也计为不一致，避免短前缀得到虚假的 100% 一致率。"""
    length = max(len(pred_ids), len(ref_ids))
    if length == 0:
        return 1.0
    matches = sum(a == b for a, b in zip(pred_ids, ref_ids))
    return matches / length


def dense_cache_bytes(cache: DynamicCache) -> int:
    return sum(t.numel() * t.element_size() for t in [*cache.key_cache, *cache.value_cache])


@torch.inference_mode()
def run_case(model, tokenizer, case: dict, *, method: str, max_new_tokens: int,
             sample: bool, seed: int, phase_kwargs: dict,
             stop_on_eos: bool = True) -> dict:
    text = tokenizer.apply_chat_template(case["messages"], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to("cuda")
    prompt_len = input_ids.shape[1]
    max_new_tokens = min(max_new_tokens, model.config.max_position_embeddings - prompt_len)
    if max_new_tokens <= 0:
        raise ValueError(f"{case['id']} 的 prompt 已超出模型上下文")
    positions = torch.arange(prompt_len, device=input_ids.device)
    eos_ids = qwen2.eos_token_ids(model.generation_config)
    sample_options = qwen2.sampling_options(model.generation_config)
    generator = torch.Generator(device=input_ids.device).manual_seed(seed) if sample else None
    seen_token_ids = input_ids

    def choose(logits: torch.Tensor) -> torch.Tensor:
        if not sample:
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"{case['id']} 的 {method} logits 含 NaN/Inf")
            return logits.float().argmax(dim=-1)
        return qwen2.sample_token(
            logits, seen_token_ids=seen_token_ids, generator=generator, **sample_options)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    prefill = model(
        input_ids=input_ids, past_key_values=DynamicCache(), use_cache=True,
        output_attentions=(method == "phasekv"), position_ids=positions.unsqueeze(0),
        cache_position=positions,
    )
    cache = (qwen2.build_phasekv_cache(prefill, model, prompt_len, **phase_kwargs)
             if method == "phasekv" else prefill.past_key_values)
    next_token = choose(prefill.logits[:, -1, :])
    del prefill
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - start) * 1000

    generated_ids: list[int] = []
    step_ms: list[float] = []
    stop_reason = "max_new_tokens"
    for step in range(max_new_tokens):
        token_id = int(next_token.item())
        if stop_on_eos and token_id in eos_ids:
            stop_reason = "eos"
            break
        generated_ids.append(token_id)
        if step == max_new_tokens - 1:
            break
        seen_token_ids = torch.cat((seen_token_ids, next_token.view(1, 1)), dim=1)
        true_position = prompt_len + step
        physical_position = cache.get_seq_length()
        begin_step = time.perf_counter()
        output = model(
            input_ids=next_token.view(1, 1), past_key_values=cache,
            use_cache=True, output_attentions=False,
            position_ids=torch.tensor([[true_position]], device=input_ids.device),
            cache_position=torch.tensor([physical_position], device=input_ids.device),
        )
        if method == "phasekv":
            cache.finish_decode_step()
        next_token = choose(output.logits[:, -1, :])
        del output
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - begin_step) * 1000)

    peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
    kv_bytes = (cache.physical_nbytes() if method == "phasekv" else dense_cache_bytes(cache))
    retained_prompt_tokens = cache.layer_tiers(0).prompt.length if method == "phasekv" else prompt_len
    cold_tokens = cache.layer_tiers(0).cold_len if method == "phasekv" else None
    hot_tokens = cache.layer_tiers(0).hot_len if method == "phasekv" else None
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "case_id": case["id"], "method": method, "seed": seed,
        "prompt_tokens": prompt_len, "retained_prompt_tokens": retained_prompt_tokens,
        "generated_tokens": len(generated_ids),
        "stop_reason": stop_reason, "answer_hit": answer_hit(response, case["answers"]),
        "repeat4_fraction": repeated_fourgram_fraction(generated_ids),
        "setup_ms": setup_ms, "tpot_ms": statistics.mean(step_ms) if step_ms else 0.0,
        "total_ms": setup_ms + sum(step_ms), "peak_gpu_bytes": peak_gpu_bytes,
        "kv_bytes": kv_bytes, "cold_tokens": cold_tokens, "hot_tokens": hot_tokens,
        "text": response, "generated_ids": generated_ids,
    }


def summarize(rows: list[dict], method: str) -> dict:
    selected = [row for row in rows if row["method"] == method]
    labeled = [row["answer_hit"] for row in selected if row["answer_hit"] is not None]
    median = lambda key: statistics.median(row[key] for row in selected)
    return {
        "runs": len(selected),
        "answer_hit_rate": sum(labeled) / len(labeled) if labeled else None,
        "mean_repeat4_fraction": statistics.mean(row["repeat4_fraction"] for row in selected),
        "max_token_runs": sum(row["stop_reason"] == "max_new_tokens" for row in selected),
        "median_setup_ms": median("setup_ms"), "median_tpot_ms": median("tpot_ms"),
        "median_total_ms": median("total_ms"),
        "median_peak_gpu_mib": median("peak_gpu_bytes") / (1024 ** 2),
        "median_kv_mib": median("kv_bytes") / (1024 ** 2),
    }


def write_report(out_prefix: Path, payload: dict) -> tuple[Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Qwen2 真实 PhaseKVCache 对比", "",
             "普通 DynamicCache 与真实打包 PhaseKVCache 使用同一模型、prompt 和解码策略。",
             "命中率只检查预设答案是否出现在回答中；token 一致率只衡量输出偏移，不代表语义质量。",
             "峰值显存含模型权重、prefill 中间张量和缓存；PhaseKV 的 setup 含注意力收集与打包。", "",
             "| 方法 | 题目命中率 | 平均重复 4-gram | 达到 token 上限 | setup 中位数 ms | TPOT 中位数 ms | 峰值显存中位数 MiB | KV 中位数 MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ("full", "phasekv"):
        s = payload["summary"][method]
        hit = "—" if s["answer_hit_rate"] is None else f"{s['answer_hit_rate']:.1%}"
        lines.append(f"| {method} | {hit} | {s['mean_repeat4_fraction']:.3f} | {s['max_token_runs']}/{s['runs']} | "
                     f"{s['median_setup_ms']:.1f} | {s['median_tpot_ms']:.1f} | "
                     f"{s['median_peak_gpu_mib']:.1f} | {s['median_kv_mib']:.2f} |")
    lines += ["", "## 逐题结果", "",
              "| 题目 | 种子 | full 命中 | PhaseKV 命中 | full 重复率 | PhaseKV 重复率 | token 一致率 | KV 压缩比 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for pair in payload["pairs"]:
        full, phase = pair["full"], pair["phasekv"]
        label = lambda value: "—" if value is None else ("✓" if value else "✗")
        lines.append(f"| {pair['case_id']} | {pair['seed']} | {label(full['answer_hit'])} | "
                     f"{label(phase['answer_hit'])} | {full['repeat4_fraction']:.3f} | "
                     f"{phase['repeat4_fraction']:.3f} | {pair['token_match']:.3f} | "
                     f"{pair['kv_compression_ratio']:.2f}× |")
    lines += ["", "完整题目、回答和逐次计时保存在同名 JSON 文件。", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="真实 PhaseKVCache vs 无压缩 Qwen2 自动基准")
    parser.add_argument("--model", default=qwen2.MODEL_ID)
    parser.add_argument("--cases-jsonl", help="每行一个题目，字段见文件顶部说明")
    parser.add_argument("--case-id", action="append", help="只跑指定题目 id；可重复传入")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题；0 表示全部")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--seeds", type=int, default=1, help="sample 模式每题的随机种子数")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-warmup", action="store_true", help="跳过每题同配置预热；计时会包含 Triton 首次编译")
    parser.add_argument("--prompt-bits", type=int, default=qwen2.PROMPT_BITS)
    parser.add_argument("--cold-bits", type=int, default=qwen2.COLD_BITS)
    parser.add_argument("--prefill-keep-ratio", type=float, default=qwen2.PREFILL_KEEP_RATIO)
    parser.add_argument("--decode-keep-ratio", type=float, default=qwen2.DECODE_KEEP_RATIO)
    parser.add_argument("--recent-window", type=int, default=qwen2.RECENT_WINDOW)
    parser.add_argument("--out", default="results/qwen2_phasekv_comparison")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.max_new_tokens <= 0 or args.seeds <= 0 or args.limit < 0:
        raise ValueError("token 数、seeds 必须为正；limit 不能为负")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("此基准需要支持 BF16 的 CUDA GPU")
    cases = load_cases(args.cases_jsonl)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"找不到题目 id: {sorted(missing)}")
    if args.limit:
        cases = cases[:args.limit]
    phase_kwargs = {
        "prompt_bits": args.prompt_bits, "cold_bits": args.cold_bits,
        "prefill_keep_ratio": args.prefill_keep_ratio,
        "decode_keep_ratio": args.decode_keep_ratio,
        "recent_window": args.recent_window,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    install_qwen2_phasekv_attention(model)
    rows = []
    pairs = []
    seeds_per_case = args.seeds if args.mode == "sample" else 1
    for case in cases:
        for offset in range(seeds_per_case):
            seed = args.seed + offset
            pair_rows = {}
            for method in ("full", "phasekv"):
                if not args.no_warmup:
                    run_case(model, tokenizer, case, method=method,
                             max_new_tokens=args.max_new_tokens,
                             sample=(args.mode == "sample"), seed=seed,
                             phase_kwargs=phase_kwargs)
                gc.collect()
                torch.cuda.empty_cache()
                row = run_case(model, tokenizer, case, method=method,
                               max_new_tokens=args.max_new_tokens,
                               sample=(args.mode == "sample"), seed=seed,
                               phase_kwargs=phase_kwargs)
                rows.append(row)
                pair_rows[method] = row
                print(f"[{case['id']} seed={seed}] {method}: answer={row['answer_hit']} "
                      f"tokens={row['generated_tokens']} repeat4={row['repeat4_fraction']:.3f} "
                      f"TPOT={row['tpot_ms']:.1f}ms", flush=True)
            full, phase = pair_rows["full"], pair_rows["phasekv"]
            pairs.append({
                "case_id": case["id"], "seed": seed, "full": full, "phasekv": phase,
                "token_match": full_sequence_token_match(
                    phase["generated_ids"], full["generated_ids"]),
                "first_divergence": first_divergence_index(
                    phase["generated_ids"], full["generated_ids"]),
                "kv_compression_ratio": full["kv_bytes"] / phase["kv_bytes"],
            })
    payload = {
        "meta": {"model": args.model, "mode": args.mode, "seed": args.seed,
                 "seeds_per_case": seeds_per_case, "max_new_tokens": args.max_new_tokens,
                 "warmup_per_case": not args.no_warmup, "phase_kwargs": phase_kwargs,
                 "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                 "gpu": torch.cuda.get_device_name(0)},
        "cases": cases, "pairs": pairs,
        "summary": {method: summarize(rows, method) for method in ("full", "phasekv")},
    }
    json_path, md_path = write_report(Path(args.out), payload)
    print(f"报告已保存：{json_path}、{md_path}")


if __name__ == "__main__":
    main()
