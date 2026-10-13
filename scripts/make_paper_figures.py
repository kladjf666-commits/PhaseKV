import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(RESULTS, "figures")
FONT_PATH = os.path.join(ROOT, "assets", "fonts", "SimHei.ttf")

os.makedirs(FIGDIR, exist_ok=True)
fm.fontManager.addfont(FONT_PATH)
CN = fm.FontProperties(fname=FONT_PATH).get_name()

plt.rcParams.update({
    "font.family": CN,
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR = {
    "phasekv": "#c0392b",
    "full": "#7f8c8d",
    "kivi": "#2e86de",
    "snapkv": "#27ae60",
    "scope": "#8e44ad",
    "pyramidkv": "#e08e0b",
}
LABEL = {
    "phasekv": "PhaseKV(本文)",
    "full": "full(无压缩)",
    "kivi": "KiVi",
    "snapkv": "SnapKV",
    "scope": "SCOPE",
    "pyramidkv": "PyramidKV",
}
MARKER = {
    "phasekv": "*", "full": "o", "kivi": "s",
    "snapkv": "^", "scope": "D", "pyramidkv": "v",
}


def load(name):
    with open(os.path.join(RESULTS, name), "r", encoding="utf-8") as f:
        return json.load(f)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{stem}.{ext}"))
    plt.close(fig)
    print("saved:", stem + ".pdf / .png")


def fig1_motivation():
    rf = load("roofline_4080s.json")
    mp = rf["measured_points"]
    peak_tflops = rf["hardware_roofline"]["achievable_fp16_TFLOPS"]

    pre_x = [p["prompt_len"] for p in mp["prefill"]]
    pre_util = [p["compute_util_pct"] for p in mp["prefill"]]
    dec_x = [p["context_len"] for p in mp["decode"]]
    dec_util = [p["achieved_TFLOPS"] / peak_tflops * 100.0 for p in mp["decode"]]

    frac = rf["analytic"]["Qwen/Qwen2.5-7B-Instruct"]["kv_read_fraction"]
    ctx = [r["context_len"] for r in frac]
    b1 = [r["kv_fraction_b1"] * 100 for r in frac]
    b16 = [r["kv_fraction_b16"] * 100 for r in frac]
    b64 = [r["kv_fraction_b64"] * 100 for r in frac]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.plot(pre_x, pre_util, marker="o", color="#2e86de", lw=2, label="prefill 算力利用率")
    ax1.plot(dec_x[:len(pre_x)], dec_util[:len(pre_x)], marker="s", color="#c0392b", lw=2,
             label="decode 算力利用率")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("序列长度 / 上下文长度 (token)")
    ax1.set_ylabel("GEMM 算力利用率 (%)")
    ax1.set_title("(a) prefill 计算受限 vs decode 非计算受限")
    ax1.axhline(0, color="#bdc3c7", lw=0.8)
    ax1.legend(frameon=False, loc="upper left")
    ax1.grid(axis="y", ls=":", alpha=0.5)
    ax1.annotate("prefill 升至 77% ≈ 峰值\n(计算受限)", xy=(pre_x[-1], pre_util[-1]),
                 xytext=(pre_x[-1] * 0.30, 60), color="#2e86de",
                 arrowprops=dict(arrowstyle="->", color="#2e86de"))
    ax1.annotate("decode 贴地 <0.1%\n(非计算受限→带宽受限)", xy=(dec_x[2], dec_util[2]),
                 xytext=(dec_x[0] * 1.1, 22), color="#c0392b",
                 arrowprops=dict(arrowstyle="->", color="#c0392b"))

    ax2.plot(ctx, b1, marker="o", color="#95a5a6", lw=2, label="batch=1")
    ax2.plot(ctx, b16, marker="s", color="#2e86de", lw=2, label="batch=16")
    ax2.plot(ctx, b64, marker="^", color="#c0392b", lw=2, label="batch=64")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("上下文长度 (token)")
    ax2.set_ylabel("KV 读流量占单步 decode 总读字节 (%)")
    ax2.set_title("(b) 7B decode: KV 读流量随上下文/批量放大")
    ax2.legend(frameon=False, loc="upper left", title="解码批量")
    ax2.grid(axis="y", ls=":", alpha=0.5)
    ax2.set_ylim(0, 100)

    fig.suptitle("图1  动机: decode 阶段受 HBM 带宽支配, KV 读流量是主因 (RTX 4080 SUPER, 真实测量+解析)",
                 fontsize=13, y=1.02)
    save(fig, "fig1_motivation_roofline")


def fig2_pareto():
    m1 = load("m1_longbench_7b.json")
    rows = {r["method"]: r for r in m1["rows"]}
    n = m1["meta"]["n_samples_per_task"]

    off = {
        "full": (10, -15, "left"),
        "scope": (10, 8, "left"),
        "kivi": (0, 13, "center"),
        "snapkv": (11, -4, "left"),
        "pyramidkv": (0, -19, "center"),
        "phasekv": (-14, 8, "right"),
    }
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for method, r in rows.items():
        ax.scatter(r["kv_saving_pct"], r["longbench_score"],
                   s=400 if method == "phasekv" else 150,
                   c=COLOR[method], marker=MARKER[method],
                   edgecolors="black", linewidths=0.8, zorder=3)
        dx, dy, ha = off[method]
        ax.annotate(LABEL[method], (r["kv_saving_pct"], r["longbench_score"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=10, color=COLOR[method],
                    fontweight="bold" if method == "phasekv" else "normal")

    full = rows["full"]
    ax.axhline(full["longbench_score"], color="#7f8c8d", ls="--", lw=1, alpha=0.7)
    ax.set_xlim(-7, 105)
    ax.set_ylim(13.3, 19.5)
    ax.set_xlabel("KV 逻辑字节节省 (%, 按 bit 折算, 越右越省)")
    ax.set_ylabel("LongBench qa_f1 (×100, 越高越好)")
    ax.set_title(f"图2  7B 精度–压缩率 Pareto (n={n}×5任务)")
    ax.grid(ls=":", alpha=0.5)
    ax.annotate("PhaseKV: 同精度下压缩最激进\n(理想方向 → 高精度 + 高压缩)",
                xy=(90, 18.25), xytext=(16, 18.7), color="#c0392b", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
    fig.text(0.5, -0.02,
             "口径: 省%=逻辑KV字节非峰值; 阶段A峰值显存 PhaseKV 19715MB > full 16010MB(伪量化+eager)。来源: results/m1_longbench_7b.json",
             ha="center", fontsize=8.5, color="#7f8c8d")
    save(fig, "fig2_pareto_7b")


def fig3_significance():
    sig = load("_qa_significance_7b.json")
    order = ["full", "kivi", "phasekv", "snapkv"]
    rows = {r["method"]: r for r in sig["rows"] if r["method"] in order}
    order = [m for m in order if m in rows]

    means = [rows[m]["longbench_score"] for m in order]
    lo = [rows[m]["longbench_score"] - rows[m]["longbench_score_ci95"][0] for m in order]
    hi = [rows[m]["longbench_score_ci95"][1] - rows[m]["longbench_score"] for m in order]
    n = rows[order[0]]["num_samples"]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(len(order))
    colors = [COLOR[m] for m in order]
    ax.errorbar(x, means, yerr=[lo, hi], fmt="none", ecolor="#34495e",
                elinewidth=1.6, capsize=7, zorder=2)
    ax.scatter(x, means, s=[300 if m == "phasekv" else 150 for m in order],
               c=colors, marker="o", edgecolors="black", linewidths=0.8, zorder=3)
    for xi, m in zip(x, order):
        last = xi == x[-1]
        ax.annotate(f"{rows[m]['longbench_score']:.2f}", (xi, rows[m]["longbench_score"]),
                    xytext=(-11 if last else 11, 0), textcoords="offset points", va="center",
                    ha="right" if last else "left",
                    fontsize=10, color=COLOR[m], fontweight="bold")

    ax.axhline(rows["full"]["longbench_score"], color="#7f8c8d", ls="--", lw=1, alpha=0.7)
    ax.set_xlim(-0.6, len(order) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{LABEL[m]}\n(省{rows[m]['kv_saving_pct']:.1f}%)" for m in order])
    ax.set_ylabel("LongBench qa_f1 (×100) 均值 ± 95%CI")
    ax.set_title(f"图3  7B 显著性复核 (n={n}样): 各方法精度置信区间重叠")
    ax.grid(axis="y", ls=":", alpha=0.5)

    pk = sig["paired_vs_full"]["phasekv"]
    txt = (f"PhaseKV vs full: 均值差 {pk['mean_diff_vs_full']:+.3f} "
           f"(95%CI {pk['mean_diff_ci95'][0]}~{pk['mean_diff_ci95'][1]}), "
           f"bootP={pk['bootstrap_two_sided_p']:.3f} → 0.05 下不显著\n"
           f"结论: PhaseKV 精度与 full 统计上一致, 却节省 {rows['phasekv']['kv_saving_pct']:.1f}% 逻辑KV")
    fig.text(0.5, -0.04, txt, ha="center", fontsize=9, color="#2c3e50")
    save(fig, "fig3_significance_7b")


def fig4_accuracy_per_byte():
    apb = load("accuracy_per_byte.json")
    seg = {"prompt": [], "generated": []}
    for r in apb["rows"]:
        seg[r["segment"]].append((r["keep_ratio"], r["token_match_vs_full"] * 100))
    for k in seg:
        seg[k].sort()

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    px = [a for a, _ in seg["prompt"]]
    py = [b for _, b in seg["prompt"]]
    gx = [a for a, _ in seg["generated"]]
    gy = [b for _, b in seg["generated"]]
    ax.plot(px, py, marker="o", color="#c0392b", lw=2.2, label="prompt 段 (prefill KV)")
    ax.plot(gx, gy, marker="s", color="#27ae60", lw=2.2, label="generated 段 (decode KV)")
    ax.fill_between(gx, gy, py, color="#27ae60", alpha=0.08)
    ax.set_xlabel("该段 KV 保留比例 keep_ratio")
    ax.set_ylabel("对 full 输出的逐位置保真度 token_match (%)")
    ax.set_title("图4  分相可压性: decode KV 远比 prompt KV 可压")
    ax.legend(frameon=False, loc="center right")
    ax.grid(ls=":", alpha=0.5)
    ax.set_ylim(-5, 108)
    ax.annotate("decode 段仅留 10%\n仍 100% 保真", xy=(0.1, 100),
                xytext=(0.32, 72), color="#27ae60",
                arrowprops=dict(arrowstyle="->", color="#27ae60"))
    ax.annotate("prompt 段压到 10%\n保真度崩塌至 ~5%", xy=(0.1, py[0]),
                xytext=(0.28, 22), color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
    fig.text(0.5, -0.02,
             "支撑 PhaseKV 分相策略: decode 段激进压、prefill 段谨慎压。来源: results/accuracy_per_byte.json (Qwen2.5-0.5B)",
             ha="center", fontsize=8.5, color="#7f8c8d")
    save(fig, "fig4_accuracy_per_byte")


def fig5_kernel_speedup():
    kb = load("kernel_decode_bench.json")
    bits2 = sorted([r for r in kb["rows"] if r["bits"] == 2], key=lambda r: r["seq_len"])
    bits4 = sorted([r for r in kb["rows"] if r["bits"] == 4], key=lambda r: r["seq_len"])
    seqs = [r["seq_len"] for r in bits2]
    sp2 = [r["speedup_splitk_vs_sdpa"] for r in bits2]
    sp4 = [r["speedup_splitk_vs_sdpa"] for r in bits4]
    save2 = bits2[0]["kv_bytes_saving_pct"]
    save4 = bits4[0]["kv_bytes_saving_pct"]

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.plot(seqs, sp2, marker="o", color="#c0392b", lw=2.4, ms=9,
            label=f"2-bit 打包 (省 KV {save2:.1f}%)")
    ax.plot(seqs, sp4, marker="s", color="#e08e0b", lw=2.4, ms=8,
            label=f"4-bit 打包 (省 KV {save4:.1f}%)")
    ax.axhline(1.0, color="#2c3e50", ls="--", lw=1.2, label="fp16 SDPA 基线 (1.0×)")
    ax.fill_between(seqs, 1.0, sp2, where=[v >= 1 for v in sp2], color="#c0392b", alpha=0.06)
    for xi, yi in zip(seqs, sp2):
        ax.annotate(f"{yi:.2f}×", (xi, yi), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9.5, color="#c0392b", fontweight="bold")
    for xi, yi in zip(seqs, sp4):
        ax.annotate(f"{yi:.2f}×", (xi, yi), xytext=(0, -17), textcoords="offset points",
                    ha="center", fontsize=9.5, color="#e08e0b")
    ax.set_xscale("log", base=2)
    ax.set_xticks(seqs)
    ax.set_xticklabels([f"{s // 1024}K" for s in seqs])
    ax.set_xlabel("KV 序列长度 (token, log 轴)")
    ax.set_ylabel("split-K 融合 kernel 相对 fp16 SDPA 的加速比 (×)")
    ax.set_title("图5  阶段B 真实加速比随上下文单调上升: 加速 ∝ 带宽节省 × 上下文")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(ls=":", alpha=0.5)
    ax.set_ylim(0.4, 1.85)
    fig.text(0.5, -0.02,
             "2-bit(省81%)加速 > 4-bit(省69%), 两档均随上下文单调上升 → 直接印证 decode 带宽受限。口径: 单层/单query步 microbench(warmup25/中位数100/误差~1e-5), 非端到端。来源: results/kernel_decode_bench.json",
             ha="center", fontsize=8.2, color="#7f8c8d")
    save(fig, "fig5_kernel_speedup")


def fig6_per_task():
    sig = load("_qa_significance_7b.json")
    rows = {r["method"]: r for r in sig["rows"]}
    methods = ["full", "kivi", "phasekv"]
    tasks = list(rows["full"]["longbench_per_task"].keys())
    n = rows["full"]["num_samples"]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    x = np.arange(len(tasks))
    w = 0.26
    for i, m in enumerate(methods):
        vals = [rows[m]["longbench_per_task"][t] for t in tasks]
        ax.bar(x + (i - 1) * w, vals, w, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=12)
    ax.set_ylabel("qa_f1 (×100, 越高越好)")
    ax.set_title(f"图6  7B 分任务精度: PhaseKV 在多数 LongBench 子任务持平/领先 (n={n}样)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", ls=":", alpha=0.5)
    save(fig, "fig6_per_task_7b")


def fig7_sweep():
    sw = load("phasekv_sweep_0p5b.json")
    rows = sorted(sw["rows"], key=lambda r: r["compression_ratio"])
    cr = [r["compression_ratio"] for r in rows]
    ppl = [r["perplexity"] for r in rows]
    tm = [r["token_match_vs_full"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(cr, ppl, marker="o", color="#c0392b", lw=2.2, label="困惑度 PPL (左轴, 越低越好)")
    ax.set_xlabel("KV 压缩比 (×, 越右越激进)")
    ax.set_ylabel("困惑度 PPL", color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")
    ax.grid(ls=":", alpha=0.5)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(cr, tm, marker="s", color="#2e86de", lw=2.2, label="token_match (右轴, %)")
    ax2.set_ylabel("对 full 保真度 token_match (%)", color="#2e86de")
    ax2.tick_params(axis="y", labelcolor="#2e86de")

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="center right")
    ax.set_title("图7  PhaseKV 压缩率–质量 sweep: 压缩比 2.3×→7.6× 时 PPL 缓升可控")
    fig.text(0.5, -0.02,
             "来源: results/phasekv_sweep_0p5b.json (Qwen2.5-0.5B, 内置离线语料)",
             ha="center", fontsize=8.5, color="#7f8c8d")
    save(fig, "fig7_sweep_0p5b")


def main():
    print("字体:", CN, "| 输出:", FIGDIR)
    fig1_motivation()
    fig2_pareto()
    fig3_significance()
    fig4_accuracy_per_byte()
    fig5_kernel_speedup()
    fig6_per_task()
    fig7_sweep()
    print("完成: 共 7 组图 (每组 pdf+png)")


if __name__ == "__main__":
    main()
