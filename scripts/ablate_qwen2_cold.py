"""在 Qwen2 真实 PhaseKVCache 上隔离 decode 冷层对回答的影响。

同一 prompt 和同一段 teacher-forced assistant 前缀用于所有配置。前缀开头的
编号会在 32-token 热窗口之后进入冷层；随后由模型贪心补全编号。这样输出
差异来自缓存执行路径，而不是各配置此前生成了不同的文本。
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

import qwen2
from bench_qwen2_phasekv import (
    answer_hit, builtin_cases, full_sequence_token_match, normalize_cases,
    run_case as run_natural_case,
)
from src.phasekv import install_qwen2_phasekv_attention


FILLER = (
    "先解释这份记录的背景：它用于核对一项普通的资料登记。"
    "登记流程包含填写、复核、归档三个步骤，各步骤都需要仔细检查。"
    "这段说明只提供背景，不引入新的编号，也不修改开头给出的编号。"
)

CASES = (
    ("record_a", "739284"),
    ("record_b", "815603"),
    ("record_c", "224906"),
)

CONFIGS = (
    ("full", None, None, None),
    ("prompt4_no_cold", None, 1.0, None),
    ("prompt4_cold8_keepall", 8, 1.0, 32),
    ("prompt4_cold4_keepall", 4, 1.0, 32),
    ("prompt4_cold4_keep50", 4, 0.5, 32),
)


def make_case(tokenizer, case_id: str, code: str, filler_repeats: int) -> dict:
    initial = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请按已写出的内容继续作答，最后复述开头的六位编号。"},
    ]
    forced = f"记录开头的编号是 {code}。" + FILLER * filler_repeats + "\n最后复述开头的六位编号："
    prefix_ids = tokenizer.apply_chat_template(
        initial, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(
        [*initial, {"role": "assistant", "content": forced}],
        tokenize=True, add_generation_prompt=False)
    ending = full_ids[-2:]
    if ending[0] != tokenizer.eos_token_id or tokenizer.decode(ending[1:]) != "\n":
        raise ValueError("预期 chat 模板以 assistant 消息结束符和换行收尾")
    full_ids = full_ids[:-2]
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("chat 模板的 prefill 与完整消息 token 前缀不一致")
    return {
        "id": case_id, "answer": code,
        "prompt_ids": prefix_ids,
        "forced_ids": full_ids[len(prefix_ids):],
        "forced_text": forced,
    }


@torch.inference_mode()
def run_case(model, tokenizer, case: dict, *, name: str, cold_bits: int | None,
             decode_keep_ratio: float | None, recent_window: int | None,
             max_new_tokens: int) -> dict:
    device = next(model.parameters()).device
    prompt_ids = torch.tensor([case["prompt_ids"]], device=device)
    prompt_len = prompt_ids.shape[1]
    positions = torch.arange(prompt_len, device=device)
    phase = name != "full"
    prefill = model(
        input_ids=prompt_ids, past_key_values=DynamicCache(), use_cache=True,
        output_attentions=phase, position_ids=positions.unsqueeze(0),
        cache_position=positions,
    )
    if phase:
        window = (len(case["forced_ids"]) + max_new_tokens + 1
                  if recent_window is None else recent_window)
        cache = qwen2.build_phasekv_cache(
            prefill, model, prompt_len, prompt_bits=4, cold_bits=cold_bits or 4,
            prefill_keep_ratio=1.0, decode_keep_ratio=decode_keep_ratio,
            recent_window=window,
        )
        if cache.layer_tiers(0).prompt.length != prompt_len:
            raise AssertionError("Prompt 未全部保留")
    else:
        cache = prefill.past_key_values
    del prefill

    logits = None
    for index, token_id in enumerate(case["forced_ids"]):
        true_position = prompt_len + index
        physical_position = cache.get_seq_length()
        out = model(
            input_ids=torch.tensor([[token_id]], device=device),
            past_key_values=cache, use_cache=True, output_attentions=False,
            position_ids=torch.tensor([[true_position]], device=device),
            cache_position=torch.tensor([physical_position], device=device),
        )
        if phase:
            cache.finish_decode_step()
        logits = out.logits[:, -1, :]
        del out

    cold_at_answer = cache.layer_tiers(0).cold_len if phase else None
    if phase and recent_window is not None and cold_at_answer == 0:
        raise AssertionError("测试未触发冷层；请增加 --filler-repeats")
    eos_ids = qwen2.eos_token_ids(model.generation_config)
    generated: list[int] = []
    for index in range(max_new_tokens):
        token_id = int(logits.float().argmax(dim=-1).item())
        if token_id in eos_ids:
            break
        generated.append(token_id)
        if index == max_new_tokens - 1:
            break
        true_position = prompt_len + len(case["forced_ids"]) + index
        physical_position = cache.get_seq_length()
        out = model(
            input_ids=torch.tensor([[token_id]], device=device),
            past_key_values=cache, use_cache=True, output_attentions=False,
            position_ids=torch.tensor([[true_position]], device=device),
            cache_position=torch.tensor([physical_position], device=device),
        )
        if phase:
            cache.finish_decode_step()
        logits = out.logits[:, -1, :]
        del out
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {
        "case_id": case["id"], "config": name, "answer": case["answer"],
        "answer_hit": answer_hit(response, [case["answer"]]),
        "text": response, "generated_ids": generated,
        "prompt_tokens": prompt_len, "forced_tokens": len(case["forced_ids"]),
        "cold_tokens_at_answer": cold_at_answer,
        "hot_tokens_at_answer": cache.layer_tiers(0).hot_len if phase else None,
    }


def write_report(out_prefix: Path, payload: dict) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Qwen2 PhaseKV 冷层准确性消融", "",
        "固定 Prompt 全保留、4-bit；对所有配置逐 token 输入相同的长回复前缀，随后贪心补全编号。",
        "`no_cold` 保持生成段 FP16 热缓存；`keepall` 只量化冷层；`keep50` 还会筛掉一半冷层 token。",
        "命中表示回答包含预设六位编号。前缀由实验脚本固定输入，不是模型自然生成。",
        "本测试只覆盖同一轮长回复形成的冷层，不评估多轮会话缓存复用。", "",
        "| 配置 | 命中 | 相对 prompt4_no_cold 的 token 一致率 | 回答前冷层 token 数 |",
        "|---|---:|---:|---:|",
    ]
    cases = payload["cases"]
    for name, *_ in CONFIGS:
        rows = payload["rows"][name]
        hits = sum(row["answer_hit"] for row in rows)
        matches = [full_sequence_token_match(
            row["generated_ids"], payload["rows"]["prompt4_no_cold"][i]["generated_ids"])
            for i, row in enumerate(rows)]
        cold = "—" if name == "full" else str(rows[0]["cold_tokens_at_answer"])
        lines.append(f"| {name} | {hits}/{len(cases)} | {sum(matches) / len(matches):.3f} | {cold} |")
    lines += ["", "## 逐题回答", "", "| 题目 | 标准答案 | "
              + " | ".join(name for name, *_ in CONFIGS) + " |",
              "|---|---:|" + "---:|" * len(CONFIGS)]
    for i, case in enumerate(cases):
        cells = [payload["rows"][name][i]["text"].replace("|", "\\|").replace("\n", " ")
                 for name, *_ in CONFIGS]
        lines.append(f"| {case['id']} | {case['answer']} | " + " | ".join(cells) + " |")
    if "natural_long_reply" in payload:
        lines += ["", "## 自然长回复", "",
                  "使用内置 `long_followup`，每种配置独立贪心生成；此题没有标准答案，token 一致率只表示输出是否分叉。", "",
                  "| 配置 | 生成 token | 冷层 token | 相对 no_cold 首次分叉 token | token 一致率 |",
                  "|---|---:|---:|---:|---:|"]
        natural = payload["natural_long_reply"]
        reference = natural["prompt4_no_cold"]["generated_ids"]
        for name, *_ in CONFIGS:
            row = natural[name]
            ids = row["generated_ids"]
            first_div = next((i + 1 for i, (a, b) in enumerate(zip(ids, reference)) if a != b),
                             None)
            if first_div is None and len(ids) != len(reference):
                first_div = min(len(ids), len(reference)) + 1
            cold = "—" if row["cold_tokens"] is None else str(row["cold_tokens"])
            lines.append(f"| {name} | {len(ids)} | {cold} | {first_div or '—'} | "
                         f"{full_sequence_token_match(ids, reference):.3f} |")
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 4-bit 全保留下的冷层准确性消融")
    parser.add_argument("--model", default=qwen2.MODEL_ID)
    parser.add_argument("--out", default="results/qwen2_phasekv_cold_ablation")
    parser.add_argument("--filler-repeats", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--natural-max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.filler_repeats <= 0 or args.max_new_tokens <= 0 or args.natural_max_new_tokens <= 0:
        raise ValueError("所有 token 数与 filler-repeats 必须为正")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("此实验需要支持 BF16 的 CUDA GPU")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    install_qwen2_phasekv_attention(model)
    cases = [make_case(tokenizer, case_id, code, args.filler_repeats)
             for case_id, code in CASES]
    rows = {name: [] for name, *_ in CONFIGS}
    for case in cases:
        for name, cold_bits, ratio, recent_window in CONFIGS:
            gc.collect()
            torch.cuda.empty_cache()
            row = run_case(
                model, tokenizer, case, name=name, cold_bits=cold_bits,
                decode_keep_ratio=ratio, recent_window=recent_window,
                max_new_tokens=args.max_new_tokens)
            rows[name].append(row)
            print(f"[{case['id']}] {name}: {row['answer_hit']} "
                  f"cold={row['cold_tokens_at_answer']} answer={row['text']!r}", flush=True)
    payload = {
        "meta": {"model": args.model, "prompt_bits": 4, "prefill_keep_ratio": 1.0,
                 "recent_window": 32, "lazy_g": qwen2.LAZY_G,
                 "filler_repeats": args.filler_repeats,
                 "max_new_tokens": args.max_new_tokens,
                 "gpu": torch.cuda.get_device_name(0)},
        "cases": [{"id": c["id"], "answer": c["answer"],
                   "prompt_tokens": len(c["prompt_ids"]),
                   "forced_tokens": len(c["forced_ids"]),
                   "forced_text": c["forced_text"]} for c in cases],
        "rows": rows,
    }
    natural_case = next(c for c in normalize_cases(builtin_cases())
                        if c["id"] == "long_followup")
    natural_rows = {}
    for name, cold_bits, ratio, recent_window in CONFIGS:
        gc.collect()
        torch.cuda.empty_cache()
        phase_kwargs = {} if name == "full" else {
            "prompt_bits": 4, "cold_bits": cold_bits or 4,
            "prefill_keep_ratio": 1.0, "decode_keep_ratio": ratio,
            "recent_window": (args.natural_max_new_tokens + 1
                              if recent_window is None else recent_window),
        }
        row = run_natural_case(
            model, tokenizer, natural_case,
            method="full" if name == "full" else "phasekv",
            max_new_tokens=args.natural_max_new_tokens,
            sample=False, seed=1234, phase_kwargs=phase_kwargs)
        natural_rows[name] = {
            "generated_ids": row["generated_ids"], "text": row["text"],
            "cold_tokens": row["cold_tokens"], "stop_reason": row["stop_reason"],
        }
        print(f"[long_followup] {name}: tokens={row['generated_tokens']} "
              f"cold={row['cold_tokens']}", flush=True)
    payload["natural_long_reply"] = natural_rows
    payload["meta"]["natural_max_new_tokens"] = args.natural_max_new_tokens
    write_report(Path(args.out), payload)
    print(f"报告已保存：{args.out}.json、{args.out}.md", flush=True)


if __name__ == "__main__":
    main()
