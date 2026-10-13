"""QA 显著性验证 harness (CH-3 / 质量验证)。

目的: 关闭 RISK_LIST R-13 —— "PhaseKV 精度反超无损 full" 的头条主张此前只建立在
n=32/任务 (160 样) 的均值上, 无置信区间、无配对显著性检验、无第二 seed 复核, 且绝对分
在 n=12→n=32 间漂移 ~47%, 中段方法在噪声内并列。本 harness 在更大 N 上独立复现并给出:

  1. 逐样本 qa_f1 (现有 scripts/run_longbench.py 只落每任务均值, 缺逐样本, 无法做区间/配对)
  2. 每方法总分 (LongBench 官方 mean-of-task-means 口径) 的分层 bootstrap 95% 置信区间
  3. 每个方法 vs full 的配对比较: 分层 bootstrap 的均值差 CI + 双侧 bootstrap p 值 +
     逐样本 win/tie/loss 与二项符号检验 p 值 + (若 scipy 可用) Wilcoxon 符号秩 p 值
  4. 对 R-13 头条主张 (phasekv>full) 给出统计学裁决

边界与红线 (QA 自持, 不越界):
  - 不修改任何他人文件。import 复用 scripts/run_longbench.py 的官方 qa_f1 打分 / 数据加载 /
    prompt 构建 (导入即触发 baselines + src.phasekv 注入), 复用 CH-3 自己的 eval.runtime。
  - 方法学与 run_longbench 完全一致 (同 iso-context 共享 input_ids / greedy / 同一中部截断 /
    同一 DATASET2MAXLEN), 故本结果是对 M1 主结果的独立复现 + 统计扩展, 数字可直接对照。
  - 全部真实模型真实测量, 无模拟数据; 不报 kernel 加速比; 无 dense 回退。
  - 产物仅落 results/_qa_significance_*(QA 命名空间), 增量保存 (每方法跑完即落盘, 防中断丢失)。

在 release/kvcomp/ 下运行, 例:
  HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  python scripts/qa_significance.py \
    --model /root/autodl-tmp/modelscope-cache/Qwen/Qwen2___5-7B-Instruct \
    --tasks narrativeqa,multifieldqa_en,hotpotqa,2wikimqa,qasper \
    --n-samples 100 --max-length 1536 --seed 1234 \
    --out results/_qa_significance_7b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 复用 run_longbench 的官方口径与数据加载 (导入即注入 baselines / src.phasekv)。
import scripts.run_longbench as rlb  # noqa: E402
from eval import metrics, runtime  # noqa: E402
from eval.compressors import available, build_compressor  # noqa: E402

import numpy as np  # noqa: E402

try:
    from scipy import stats as _scipy_stats  # noqa: E402
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# 与 M1 主结果 (results/m1_longbench_7b) 完全一致的方法配置, 保证可直接对照。
# 顺序: full 必须最先 (作为配对参考与 token_match 基准), phasekv 次之 (头条)。
DEFAULT_METHODS = [
    {"method": "full"},
    {"method": "phasekv", "prefill_keep_ratio": 0.3, "prompt_bits": 2,
     "cold_bits": 4, "recent_window": 32},
    {"method": "kivi", "bits": 2, "group_size": 32, "recent_window": 64},
    {"method": "snapkv", "prefill_keep_ratio": 0.3, "obs_window": 16},
    {"method": "pyramidkv", "prefill_keep_ratio": 0.3, "obs_window": 16, "beta": 20},
    {"method": "scope", "recent_window": 16, "max_cache_tokens": 256},
]


# --------------------------------------------------------------------------- #
# 统计工具 (分层 bootstrap 尊重 LongBench mean-of-task-means 与按任务聚类的结构)
# --------------------------------------------------------------------------- #
def _mean_of_task_means(scores_by_task):
    """LongBench 官方口径: 先每任务取均值, 再对任务求平均 (x100)。"""
    task_means = [100.0 * float(np.mean(v)) for v in scores_by_task.values() if len(v)]
    return float(np.mean(task_means)) if task_means else float("nan")


def _stratified_bootstrap_mean(scores_by_task, n_boot, rng):
    """对 mean-of-task-means 做分层 bootstrap: 每任务内有放回重采样后重算任务均值再平均。"""
    tasks = list(scores_by_task.keys())
    arrs = {t: np.asarray(scores_by_task[t], dtype=float) for t in tasks}
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        tms = []
        for t in tasks:
            a = arrs[t]
            if len(a) == 0:
                continue
            idx = rng.integers(0, len(a), size=len(a))
            tms.append(100.0 * a[idx].mean())
        boots[b] = float(np.mean(tms)) if tms else float("nan")
    return boots


def _paired_stratified_bootstrap(diff_by_task, n_boot, rng):
    """对配对差值 (逐样本 method-full) 做分层 bootstrap, 返回 mean-of-task-means 差的分布。"""
    tasks = list(diff_by_task.keys())
    arrs = {t: np.asarray(diff_by_task[t], dtype=float) for t in tasks}
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        tms = []
        for t in tasks:
            a = arrs[t]
            if len(a) == 0:
                continue
            idx = rng.integers(0, len(a), size=len(a))
            tms.append(100.0 * a[idx].mean())
        boots[b] = float(np.mean(tms)) if tms else float("nan")
    return boots


def _ci(boots, alpha=0.05):
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return round(lo, 3), round(hi, 3)


def _boot_two_sided_p(boots):
    """双侧 bootstrap p 值: 差分布落在 0 另一侧的比例 x2 (截断到 [0,1])。"""
    n = len(boots)
    p_le = float(np.mean(boots <= 0.0))
    p_ge = float(np.mean(boots >= 0.0))
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def _sign_test_p(wins, losses):
    """二项符号检验 (双侧), 忽略 tie。n=wins+losses, k=min(wins,losses), p=0.5。"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def _wilcoxon_p(diffs):
    if not _HAS_SCIPY:
        return None
    nz = [d for d in diffs if d != 0.0]
    if len(nz) < 1:
        return None
    try:
        res = _scipy_stats.wilcoxon(nz, alternative="two-sided",
                                    zero_method="wilcox", mode="auto")
        return float(res.pvalue)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(model_name, methods, tasks, *, n_samples, max_length, dtype, device,
        seed, n_boot, out_prefix):
    attn_impl = "eager" if rlb._needs_attention(methods) else "sdpa"
    print(f"[qa_sig] 加载模型 {model_name} attn={attn_impl} dtype={dtype}", flush=True)
    model, tokenizer = runtime.load_model_and_tokenizer(
        model_name, dtype=dtype, device=device, attn_implementation=attn_impl)
    num_layers, num_kv_heads, head_dim = rlb._kv_dims(model.config)

    # 一次 tokenize, 全方法共享 (iso-context)。逐样本记录任务归属以便分层统计。
    samples = []
    for task in tasks:
        for s in rlb.load_task(task, n_samples):
            input_ids = rlb.build_input_ids(tokenizer, task, s, max_length)
            samples.append({"task": task, "answers": s["answers"],
                            "input_ids": input_ids, "prompt_len": input_ids.shape[1],
                            "max_new": rlb.DATASET2MAXLEN[task]})
    print(f"[qa_sig] 样本总数={len(samples)} 任务={tasks} 截断={max_length} "
          f"seed={seed} n_boot={n_boot}", flush=True)

    ordered = sorted(methods, key=lambda m: 0 if m["method"] == "full" else 1)
    per_method = {}          # name -> {"per_sample":[{task,score,pred?}], "agg":{...}}
    full_scores_by_task = None
    full_ref_ids = {}

    t_start = time.perf_counter()
    for spec in ordered:
        params = {k: v for k, v in spec.items() if k != "method"}
        name = spec["method"]
        compressor = build_compressor(name, **params)
        per_sample = []
        scores_by_task = {t: [] for t in tasks}
        savings, peaks, matches = [], [], []
        m_t0 = time.perf_counter()

        for idx, smp in enumerate(samples):
            gen = runtime.generate_with_compression(
                model, tokenizer, smp["input_ids"], compressor,
                max_new_tokens=smp["max_new"], greedy=True, seed=seed)
            pred = gen["text"].strip()
            score = rlb.best_qa_f1(pred, smp["answers"])
            scores_by_task[smp["task"]].append(score)
            per_sample.append({"i": idx, "task": smp["task"], "qa_f1": round(score, 6)})

            total_tokens = smp["prompt_len"] + gen["latency"].num_generated
            full_ref_bytes = metrics.kv_cache_dtype_bytes(
                num_layers, num_kv_heads, head_dim, total_tokens, dtype_bytes=2)
            savings.append(metrics.memory_saving_ratio(
                gen["final_cache_bytes"], full_ref_bytes) * 100.0)
            peaks.append(gen["peak_gpu_bytes"] / 1e6)
            if name == "full":
                full_ref_ids[idx] = gen["generated_ids"]
            elif idx in full_ref_ids:
                matches.append(metrics.token_match_rate(
                    gen["generated_ids"], full_ref_ids[idx]))

        rng = np.random.default_rng(20260701)
        boots = _stratified_bootstrap_mean(scores_by_task, n_boot, rng)
        ci_lo, ci_hi = _ci(boots)
        task_means = {t: (round(100.0 * float(np.mean(v)), 3) if v else None)
                      for t, v in scores_by_task.items()}
        agg = {
            "longbench_score": round(_mean_of_task_means(scores_by_task), 3),
            "longbench_score_ci95": [ci_lo, ci_hi],
            "mean_over_samples": round(100.0 * float(np.mean(
                [s["qa_f1"] for s in per_sample])), 3),
            "longbench_per_task": task_means,
            "kv_saving_pct": round(float(np.mean(savings)), 4) if savings else None,
            "peak_gpu_mb": round(float(np.max(peaks)), 1) if peaks else None,
            "token_match_vs_full": round(float(np.mean(matches)), 4) if matches else None,
            "num_samples": len(per_sample),
            "params": params,
        }
        per_method[name] = {"per_sample": per_sample, "agg": agg}
        if name == "full":
            full_scores_by_task = scores_by_task

        dt = time.perf_counter() - m_t0
        print(f"[qa_sig] {name}: LB={agg['longbench_score']} "
              f"CI95=[{ci_lo},{ci_hi}] 省{agg['kv_saving_pct']}% "
              f"峰值{agg['peak_gpu_mb']}MB 耗时{dt:.0f}s", flush=True)

        _save(_assemble(model_name, methods, tasks, n_samples, max_length, dtype,
                        device, seed, n_boot, attn_impl, num_layers, num_kv_heads,
                        head_dim, per_method, full_scores_by_task), out_prefix)

    total_dt = time.perf_counter() - t_start
    print(f"[qa_sig] 全部完成, 总耗时 {total_dt/60:.1f} min", flush=True)
    return _assemble(model_name, methods, tasks, n_samples, max_length, dtype,
                     device, seed, n_boot, attn_impl, num_layers, num_kv_heads,
                     head_dim, per_method, full_scores_by_task)


def _paired_stats(per_method, full_scores_by_task, tasks, n_boot):
    """对每个非 full 方法算 vs full 的配对统计 (逐样本对齐)。"""
    if "full" not in per_method:
        return {}
    full_ps = {s["i"]: s["qa_f1"] for s in per_method["full"]["per_sample"]}
    full_task = {s["i"]: s["task"] for s in per_method["full"]["per_sample"]}
    out = {}
    rng = np.random.default_rng(20260702)
    for name, obj in per_method.items():
        if name == "full":
            continue
        diffs, wins, losses, ties = [], 0, 0, 0
        diff_by_task = {t: [] for t in tasks}
        for s in obj["per_sample"]:
            i = s["i"]
            if i not in full_ps:
                continue
            d = s["qa_f1"] - full_ps[i]
            diffs.append(d)
            diff_by_task[full_task[i]].append(d)
            if d > 1e-9:
                wins += 1
            elif d < -1e-9:
                losses += 1
            else:
                ties += 1
        boots = _paired_stratified_bootstrap(diff_by_task, n_boot, rng)
        lo, hi = _ci(boots)
        out[name] = {
            "mean_diff_vs_full": round(100.0 * float(np.mean(diffs)), 3) if diffs else None,
            "mean_diff_ci95": [lo, hi],
            "bootstrap_two_sided_p": round(_boot_two_sided_p(boots), 4),
            "wilcoxon_p": (round(p, 6) if (p := _wilcoxon_p(diffs)) is not None else None),
            "win_tie_loss": [wins, ties, losses],
            "sign_test_p": round(_sign_test_p(wins, losses), 4),
            "significant_at_0.05": bool(lo > 0.0 or hi < 0.0),
        }
    return out


def _assemble(model_name, methods, tasks, n_samples, max_length, dtype, device,
              seed, n_boot, attn_impl, num_layers, num_kv_heads, head_dim,
              per_method, full_scores_by_task):
    paired = _paired_stats(per_method, full_scores_by_task, tasks, n_boot) \
        if full_scores_by_task is not None else {}
    rows = [{"method": n, **obj["agg"]} for n, obj in per_method.items()]
    return {
        "rows": rows,
        "paired_vs_full": paired,
        "per_sample": {n: obj["per_sample"] for n, obj in per_method.items()},
        "meta": {
            "model": model_name,
            "experiment": "qa_f1_significance (R-13 关闭)",
            "dtype": dtype,
            "attn_implementation": attn_impl,
            "tasks": list(tasks),
            "n_samples_per_task": n_samples,
            "max_length_truncate": max_length,
            "kv_dims": {"num_layers": num_layers, "num_kv_heads": num_kv_heads,
                        "head_dim": head_dim},
            "device": device,
            "greedy": True,
            "seed": seed,
            "n_bootstrap": n_boot,
            "has_scipy_wilcoxon": _HAS_SCIPY,
            "score_metric": "qa_f1 (LongBench 官方口径, x100); 总分=mean-of-task-means",
            "ci_method": "分层 bootstrap (每任务内有放回重采样, 重算任务均值再平均)",
            "paired_method": "逐样本对齐配对差 (method-full); 分层 bootstrap 差 CI + "
                             "双侧 bootstrap p + 二项符号检验 + Wilcoxon(若 scipy)",
            "note": "真实模型真实测量, 无模拟。方法学与 scripts/run_longbench.py 一致 "
                    "(iso-context 共享 input_ids / greedy / 中部截断 / DATASET2MAXLEN), "
                    "是对 results/m1_longbench_7b 的独立复现+统计扩展。",
        },
    }


def _fmt_p(p):
    if p is None:
        return "-"
    return f"{p:.3g}"


def _save(result, out_prefix):
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    rows = result["rows"]
    paired = result.get("paired_vs_full", {})
    meta = result["meta"]
    lines = [f"### QA 显著性验证 (R-13) · {meta['model']}", "",
             f"- 口径: {meta['score_metric']}; CI: {meta['ci_method']}",
             f"- 配置: n={meta['n_samples_per_task']}/任务, seed={meta['seed']}, "
             f"bootstrap={meta['n_bootstrap']}, attn={meta['attn_implementation']}, "
             f"截断={meta['max_length_truncate']}, Wilcoxon={'有' if meta['has_scipy_wilcoxon'] else '无scipy'}",
             "", "#### 每方法总分 (含 95% 置信区间)", "",
             "| 方法 | LongBench(qa_f1) | 95%CI | KV省% | 峰值MB | 对full一致 | N |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        ci = r.get("longbench_score_ci95") or [None, None]
        lines.append(
            f"| {r['method']} | {r.get('longbench_score')} | "
            f"[{ci[0]}, {ci[1]}] | {r.get('kv_saving_pct')} | "
            f"{r.get('peak_gpu_mb'):.0f} | {r.get('token_match_vs_full')} | "
            f"{r.get('num_samples')} |")
    lines += ["", "#### vs full 配对显著性 (逐样本对齐)", "",
              "| 方法 | 均值差 vs full | 差95%CI | bootP | Wilcoxon p | 符号检验 p | W/T/L | 0.05显著 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, pv in paired.items():
        ci = pv.get("mean_diff_ci95") or [None, None]
        wtl = pv.get("win_tie_loss") or [None, None, None]
        lines.append(
            f"| {name} | {pv.get('mean_diff_vs_full')} | [{ci[0]}, {ci[1]}] | "
            f"{_fmt_p(pv.get('bootstrap_two_sided_p'))} | {_fmt_p(pv.get('wilcoxon_p'))} | "
            f"{_fmt_p(pv.get('sign_test_p'))} | {wtl[0]}/{wtl[1]}/{wtl[2]} | "
            f"{'是' if pv.get('significant_at_0.05') else '否'} |")
    md = "\n".join(lines) + "\n"
    with open(f"{out_prefix}.md", "w", encoding="utf-8") as f:
        f.write(md)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="QA 显著性验证 (R-13): 逐样本 qa_f1 + bootstrap CI + 配对检验")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", default="narrativeqa,multifieldqa_en,hotpotqa,2wikimqa,qasper")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--methods-json", default="", help="留空则用 DEFAULT_METHODS (与 M1 主结果同配置)")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", default="results/_qa_significance_7b")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    print(f"[qa_sig] phasekv 注入={rlb._PHASEKV}; 可用方法={available()}", file=sys.stderr)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in rlb.QA_F1_TASKS]
    if unknown:
        raise SystemExit(f"仅支持 qa_f1 任务 {sorted(rlb.QA_F1_TASKS)}; 未支持: {unknown}")
    methods = json.loads(args.methods_json) if args.methods_json else DEFAULT_METHODS
    result = run(args.model, methods, tasks, n_samples=args.n_samples,
                 max_length=args.max_length, dtype=args.dtype, device=args.device,
                 seed=args.seed, n_boot=args.n_boot, out_prefix=args.out)
    _save(result, args.out)
    print(f"报告已保存: {args.out}.json / .md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
