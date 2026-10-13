"""LongBench 任务评分流水线 (CH-10)。

补上 eval.run_benchmark 缺失的一环: 精度只有 perplexity + token_match, 没有 LongBench
任务分数 (允许改写语义的 qa_f1)。这是 M1 主结果里 "iso-memory 下 PhaseKV 精度占优"
论证所需的证据 (与 CH-7/CH-3 判读一致: 0.5B 的 token_match/perplexity 看不出 PhaseKV
优势, 需 7B + LongBench 任务分数)。

设计 (严格不改 CH-3 的 eval/tests/docs/qa):
  - 复用 eval.runtime.load_model_and_tokenizer / generate_with_compression。
  - 复用 eval.compressors.build_compressor (并 import baselines / src.phasekv 注入)。
  - 复用 eval.metrics 计算逻辑显存节省。
  - 官方口径自带: LongBench 的 prompt 模板 / 每任务最大生成长度 / qa_f1 指标 / 中部截断
    均按 THUDM/LongBench 原始实现忠实复刻 (见下方常量与函数), 不简化。

显存约束 (真实工程约束, 如实记录): 需注意力分数的方法 (snapkv/scope/h2o/phasekv) 走
eager + output_attentions=True, 会保留全部层注意力矩阵, 7B 上长上下文 O(n^2) 易 OOM。
故用 --max-length 对 prompt 做 LongBench 式中部截断 (默认 1536, 适配 32G), 并在 meta
中标注截断预算, 保证 iso-context 公平对比。

在 release/kvcomp/ 下运行, 例:
  HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf-cache \
  python scripts/run_longbench.py --model Qwen/Qwen2.5-7B-Instruct \
    --tasks narrativeqa,multifieldqa_en,hotpotqa --n-samples 8 --max-length 1536 \
    --methods-json '[{"method":"full"},{"method":"kivi","bits":2,"group_size":32,"recent_window":64},
                     {"method":"snapkv","prefill_keep_ratio":0.3,"obs_window":16},
                     {"method":"scope","recent_window":16,"max_cache_tokens":256},
                     {"method":"phasekv","prefill_keep_ratio":0.3,"prompt_bits":2,"cold_bits":4,"recent_window":32}]' \
    --out results/m1_longbench_7b
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import baselines  # noqa: E402,F401  注入 snapkv/kivi/scope(/pyramidkv)

try:
    import src.phasekv  # noqa: E402,F401  架构侧就绪则注入 phasekv
    _PHASEKV = True
except Exception as exc:  # 仅插件发现失败, 非算法回退
    _PHASEKV = False
    print(f"[run_longbench] 未注入 src.phasekv: {exc.__class__.__name__}: {exc}", file=sys.stderr)

from eval import metrics, runtime  # noqa: E402
from eval.compressors import available, build_compressor  # noqa: E402


# --------------------------------------------------------------------------- #
# LongBench 官方常量 (忠实复刻 THUDM/LongBench: config/dataset2prompt.json,
# config/dataset2maxlen.json, metrics.py)。
# --------------------------------------------------------------------------- #
DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a "
        "question. Answer the question asconcisely as you can, using a single phrase if "
        "possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the "
        "question based on the story asconcisely as you can, using a single phrase if "
        "possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as "
        "concisely as you can, using a single phrase or sentence if possible. If the "
        "question cannot be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer "
        "the question based on the above article as concisely as you can, using a single "
        "phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no "
        "question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the "
        "following question based on the above text, only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"),
}

DATASET2MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
}

# 本流水线聚焦 qa_f1 任务 (英文 QA, 允许改写)。
QA_F1_TASKS = set(DATASET2PROMPT.keys())


# --------------------------------------------------------------------------- #
# LongBench 官方 qa_f1 指标 (忠实复刻 metrics.py)。
# --------------------------------------------------------------------------- #
def normalize_answer(s: str) -> str:
    """SQuAD 式归一化: 小写, 去标点, 去冠词 a/an/the, 规整空白。"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1(prediction_tokens, ground_truth_tokens) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    return _f1(pred_tokens, gt_tokens)


def best_qa_f1(prediction: str, answers) -> float:
    """对多个标准答案取最大 F1 (LongBench scorer 口径)。"""
    return max((qa_f1_score(prediction, gt) for gt in answers), default=0.0)


# --------------------------------------------------------------------------- #
# 数据加载与 prompt 构建
# --------------------------------------------------------------------------- #
# LongBench 本地 JSONL 目录 (由 scripts/download_data.py 从 hf-mirror 解压得到)。
# 不走 THUDM/LongBench 的 HF 加载脚本: 它把 data.zip URL 硬编码为 huggingface.co,
# 在仅 hf-mirror 可达的机器上必然失败 (见 download_data.py 注释)。
LONGBENCH_DIR = os.environ.get("LONGBENCH_DIR", "/root/autodl-tmp/hf-cache/longbench")


def load_task(task: str, n_samples: int):
    path = os.path.join(LONGBENCH_DIR, "data", f"{task}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LongBench 数据缺失: {path}。请先运行 scripts/download_data.py 下载并解压。")
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            ex = json.loads(line)
            out.append({"context": ex.get("context", ""), "input": ex.get("input", ""),
                        "answers": list(ex.get("answers", []))})
    return out


def _middle_truncate(tokenizer, prompt: str, max_length: int) -> str:
    """LongBench 式中部截断: 超长时保留头尾各 half。"""
    ids = tokenizer(prompt, truncation=False, return_tensors=None)["input_ids"]
    if len(ids) <= max_length:
        return prompt
    half = max_length // 2
    return (tokenizer.decode(ids[:half], skip_special_tokens=True)
            + tokenizer.decode(ids[-half:], skip_special_tokens=True))


def build_input_ids(tokenizer, task: str, sample: dict, max_length: int):
    """构建 prompt -> 中部截断 -> 套 chat 模板 -> tokenize, 返回 input_ids[1, L]。"""
    import torch

    raw = DATASET2PROMPT[task].format(context=sample["context"], input=sample["input"])
    raw = _middle_truncate(tokenizer, raw, max_length)
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}], tokenize=False, add_generation_prompt=True)
    else:
        text = raw
    return tokenizer(text, return_tensors="pt").input_ids


# --------------------------------------------------------------------------- #
# 评测主流程
# --------------------------------------------------------------------------- #
def _kv_dims(cfg):
    num_layers = getattr(cfg, "num_hidden_layers")
    num_heads = getattr(cfg, "num_attention_heads")
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_heads)
    return num_layers, num_kv_heads, head_dim


def _needs_attention(methods) -> bool:
    for spec in methods:
        params = {k: v for k, v in spec.items() if k != "method"}
        if build_compressor(spec["method"], **params).requires_attention:
            return True
    return False


def run(model_name, methods, tasks, *, n_samples=8, max_length=1536,
        dtype="float16", device="cuda", seed=1234):
    attn_impl = "eager" if _needs_attention(methods) else "sdpa"
    model, tokenizer = runtime.load_model_and_tokenizer(
        model_name, dtype=dtype, device=device, attn_implementation=attn_impl)
    num_layers, num_kv_heads, head_dim = _kv_dims(model.config)

    # 预加载并构建所有样本 (一次 tokenize, 各方法共享同一 input_ids 保证 iso-context)。
    samples = []
    for task in tasks:
        for s in load_task(task, n_samples):
            input_ids = build_input_ids(tokenizer, task, s, max_length)
            samples.append({"task": task, "answers": s["answers"],
                            "input_ids": input_ids, "prompt_len": input_ids.shape[1],
                            "max_new": DATASET2MAXLEN[task]})
    print(f"[run_longbench] 样本总数={len(samples)} 任务={tasks} "
          f"截断预算={max_length} attn={attn_impl}", flush=True)

    # 各方法逐样本生成; full 作为 token_match 参考 (若在 methods 中)。
    full_ref_ids = {}  # sample_idx -> generated_ids
    rows = []
    method_names = [m["method"] for m in methods]
    has_full = "full" in method_names

    ordered = sorted(methods, key=lambda m: 0 if m["method"] == "full" else 1)
    for spec in ordered:
        params = {k: v for k, v in spec.items() if k != "method"}
        name = spec["method"]
        compressor = build_compressor(name, **params)
        per_task_scores = {t: [] for t in tasks}
        savings, peaks, ttfts, tpots, tpss, matches = [], [], [], [], [], []

        for idx, smp in enumerate(samples):
            gen = runtime.generate_with_compression(
                model, tokenizer, smp["input_ids"], compressor,
                max_new_tokens=smp["max_new"], greedy=True, seed=seed)
            # LongBench: trec/triviaqa/samsum/lsht 取首行; 本子集均非该类, 保留原样。
            pred = gen["text"].strip()
            score = best_qa_f1(pred, smp["answers"])
            per_task_scores[smp["task"]].append(score)

            total_tokens = smp["prompt_len"] + gen["latency"].num_generated
            full_ref_bytes = metrics.kv_cache_dtype_bytes(
                num_layers, num_kv_heads, head_dim, total_tokens, dtype_bytes=2)
            savings.append(metrics.memory_saving_ratio(gen["final_cache_bytes"], full_ref_bytes) * 100.0)
            peaks.append(gen["peak_gpu_bytes"] / 1e6)
            ttfts.append(gen["latency"].ttft_s * 1000.0)
            tpots.append(gen["latency"].tpot_s * 1000.0)
            tpss.append(gen["latency"].decode_throughput)

            if name == "full":
                full_ref_ids[idx] = gen["generated_ids"]
            elif has_full and idx in full_ref_ids:
                matches.append(metrics.token_match_rate(gen["generated_ids"], full_ref_ids[idx]))

        task_means = {t: (round(100.0 * sum(v) / len(v), 2) if v else None)
                      for t, v in per_task_scores.items()}
        valid_task_means = [m for m in task_means.values() if m is not None]
        rows.append({
            "method": compressor.name,
            "longbench_score": round(sum(valid_task_means) / len(valid_task_means), 2) if valid_task_means else None,
            "longbench_per_task": task_means,
            "kv_saving_pct": round(sum(savings) / len(savings), 4) if savings else None,
            "compression_ratio": round(100.0 / max(1e-9, 100.0 - (sum(savings) / len(savings))), 4) if savings else None,
            "peak_gpu_mb": round(max(peaks), 1) if peaks else None,
            "ttft_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
            "tpot_ms": round(sum(tpots) / len(tpots), 2) if tpots else None,
            "decode_tps": round(sum(tpss) / len(tpss), 2) if tpss else None,
            "token_match_vs_full": round(sum(matches) / len(matches), 4) if matches else None,
            "perplexity": None,
            "num_samples": len(samples),
            "params": params,
        })
        print(f"[run_longbench] {compressor.name}: LB={rows[-1]['longbench_score']} "
              f"省{rows[-1]['kv_saving_pct']}% 峰值{rows[-1]['peak_gpu_mb']}MB", flush=True)

    meta = {
        "model": model_name,
        "experiment": "longbench_qa_f1",
        "dtype": dtype,
        "attn_implementation": attn_impl,
        "tasks": list(tasks),
        "n_samples_per_task": n_samples,
        "max_length_truncate": max_length,
        "kv_dims": {"num_layers": num_layers, "num_kv_heads": num_kv_heads, "head_dim": head_dim},
        "device": device,
        "greedy": True,
        "score_metric": "qa_f1 (LongBench 官方口径, x100, 越高越好)",
        "note": ("真实模型真实测量, 无模拟数据。prompt 经 LongBench 官方模板 + chat 模板, 超 "
                 "--max-length 走中部截断 (头尾各 half); 截断为适配 32G 显存下 eager+"
                 "output_attentions 的 O(n^2) 注意力开销, 已在 meta 标注, iso-context 公平。"),
    }
    return {"rows": rows, "meta": meta}


def _save(result, out_prefix):
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    cols = [("method", "方法"), ("longbench_score", "LongBench(qa_f1)"),
            ("kv_saving_pct", "KV显存节省%"), ("peak_gpu_mb", "峰值显存MB"),
            ("ttft_ms", "TTFT(ms)"), ("tpot_ms", "TPOT(ms)"),
            ("token_match_vs_full", "对full一致率")]
    lines = [f"### LongBench 任务分数 · {result['meta']['model']}", "",
             "| " + " | ".join(l for _, l in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for r in result["rows"]:
        lines.append("| " + " | ".join(
            "-" if r.get(k) is None else (f"{r[k]:.0f}" if k == "peak_gpu_mb" else str(r[k]))
            for k, _ in cols) + " |")
    md = "\n".join(lines) + "\n"
    with open(f"{out_prefix}.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(md)
    print("报告已保存:", f"{out_prefix}.json / .md", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LongBench 任务评分流水线 (qa_f1)")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", default="narrativeqa,multifieldqa_en,hotpotqa")
    p.add_argument("--n-samples", type=int, default=8, help="每任务样本数")
    p.add_argument("--max-length", type=int, default=1536, help="prompt 中部截断 token 预算")
    p.add_argument("--methods-json", default='[{"method":"full"}]')
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/m1_longbench")
    p.add_argument("--list-methods", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"[run_longbench] phasekv 注入={_PHASEKV}; 可用方法={available()}", file=sys.stderr)
    if args.list_methods:
        return 0
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in QA_F1_TASKS]
    if unknown:
        raise SystemExit(f"本流水线仅支持 qa_f1 任务 {sorted(QA_F1_TASKS)}; 未支持: {unknown}")
    methods = json.loads(args.methods_json)
    result = run(args.model, methods, tasks, n_samples=args.n_samples,
                 max_length=args.max_length, dtype=args.dtype, device=args.device)
    _save(result, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
