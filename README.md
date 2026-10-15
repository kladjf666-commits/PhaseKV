# PhaseKV：访问放大感知的相位异构 KV Cache 压缩

PhaseKV 将 KV Cache 视为由访问模式决定表示形式的异构存储层级。prompt KV 在 prefill 阶段写入一次，并在后续生成过程中重复读取；generated KV 则随 decode 逐 token 追加，且新近 token 的重要度尚未稳定。基于这一差异，PhaseKV 使用只读低比特 prompt 层、FP16 decode 热层和懒批量化冷层三种表示，并通过 roofline 预算分配器与单遍混合精度融合注意力 kernel 统一管理。

代码库包含 PhaseKV 方法实现、统一 KV Cache 压缩评估框架、对比基线、实验脚本和完整技术文档。Qwen2.5-7B + LongBench 的实测结果显示，PhaseKV 在与 full 和主要基线统计不可区分的任务精度下，实现 93.31% 的逻辑 KV 字节节省（14.94× 压缩）；真实 2-bit decode kernel 在上下文长度 4096–32768 时相对 FP16 SDPA 达到 1.20–1.63× 加速。逻辑 KV 节省不等同于阶段 A 的峰值显存下降，现有速度数据为 kernel 级测量。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 设计依据](#2-设计依据)
- [3. 方法 PhaseKV](#3-方法-phasekv)
- [4. 目录结构](#4-目录结构)
- [5. 环境与安装](#5-环境与安装)
- [6. 快速开始](#6-快速开始)
- [7. 复现主要实验](#7-复现主要实验)
- [8. 主要结果](#8-主要结果)
- [9. 测量口径](#9-测量口径)
- [10. 当前结果范围](#10-当前结果范围)
- [11. 文档索引](#11-文档索引)
- [12. 相关工作对比](#12-相关工作对比)

---

## 1. 项目简介

PhaseKV 面向 GPU 上长上下文大模型推理的 KV Cache 容量与带宽问题。现有方法通常沿层、head、新近度或 query 维度组织异构压缩；已有 phase-aware 方法也主要在 prefill 与 decode 之间切换选择策略或预算。PhaseKV 从访问放大和阶段瓶颈出发，同时调整压缩机制、位宽、存储布局与执行时机：prefill 一次形成只读 prompt 层，decode 维护热窗口并懒批处理冷层，最终由融合 kernel 在压缩域内完成注意力计算。

本代码库同时提供方法无关的 KV Cache 压缩评估框架。压缩方法实现统一的 `KVCompressor` 接口并注册后，即可通过同一流水线评估内存、速度和精度指标。完整设计说明与相关工作对比见 [`docs/idea/PhaseKV-技术原理与系统设计.md`](docs/idea/PhaseKV-技术原理与系统设计.md)。

---

## 2. 设计依据

PhaseKV 的设计依据包括阶段执行特征、KV 读取次数和重要度信号开销。

### 2.1 Prefill 与 Decode 的执行特征
prefill 长 prompt 下注意力复杂度为 O(n²)，主要受计算能力限制，适合承担一次性的全局压缩；decode 每步只计算 1 个 query，主要开销是从 HBM 读取累积的 KV，任何逐步压缩开销都会增加关键路径上的带宽压力。
→ 实测与解析结果（RTX 4080S）：0.5B 模型的 prefill 算力利用率随 prompt 从 15.6%(512) 升至 77.0%(4096)；7B 单步 decode 的权重读取量为 15.23GB，按实测 copy 带宽计算的理想下界约 24.3ms。后者是解析结果，当前未包含 7B decode 的真实 roofline 落点。

### 2.2 Prompt-KV 与 Generated-KV 的读取次数
一个 prompt token 的 KV 在随后**每个** decode step 都被读一次（读放大 = T），且只写一次、写在带宽富余的 prefill；generated-KV 边写边用、收益与成本同阶。
→ 推论：prompt-KV 应采用比 generated-KV 更低的位宽和保留比例。该预算差异由带宽和读放大推导，而不是由任务精度假设推导。

### 2.3 重要度信号的获取开销
prefill 可以复用注意力计算产生的重要度信息；但在 FlashAttention/SDPA 路径中，显式输出完整注意力分数仍有额外开销。decode 若 materialize scores 会增加存储与带宽压力，因此冷层使用 value-norm 等无需完整注意力矩阵的信号。
→ 推论：prefill 用「注意力分数选择」，decode 用「无需分数的廉价信号」（value-norm）。

---

## 3. 方法 PhaseKV

### 3.1 三层异构缓存（`src/phasekv/hetero_cache.py`）

按生命周期/访问模式分三层，每层不同表示与布局（单层物理布局 `[b, Hkv, seq, hd]`，时间升序）：

```
[ 0 .. K0 )                  Prompt 层（frozen 只读）：per-head 注意力 heavy-hitter 选择 + KIVI 低比特（prompt_bits）量化，成型后不再改写
[ K0 .. K0+cold_len )        Decode 冷层（cooled）：滑出热窗口的老生成 token，每 G 步懒批一次 value-norm 选择 + 低比特(cold_bits)量化
[ K0+cold_len .. seq )       Decode 热层（hot）：最近 recent_window 个生成 token，FP16 全精度，追加 O(1)
```

与 KIVI 的区别：KIVI 只有「新近 FP16 残差 + 旧量化」两段、且 prompt 与 generated 不分；PhaseKV 把 prompt 单列为只读、低位宽压缩层，decode 内部再分热/冷，且压缩**时机**是阶段感知的（prefill 一次 / decode 懒批）。

### 3.2 Roofline 预算分配器（`src/phasekv/planner.py`）

把阶段执行特征和 KV 读取次数写入可计算的受约束离散优化：在「稳态逻辑显存 ≤ 目标」的可行域内，主目标「用满预算取逻辑字节最大（精度单调代理）」，次目标「decode 每步读字节最少（把预算优先给读放大低的 decode 层）」。输出是 Pareto 最优的相位异构超参配置 `(prompt 选择比/位宽, decode 热窗口, 冷层位宽/选择比, 懒批 G)`。分配器只产出**超参**，真实精度/加速比仍由 eval 实测。

### 3.3 两阶段实现与验证

| 阶段 | 定位 | 报告什么 | 不报告什么 |
| --- | --- | --- | --- |
| **阶段 A**（`compressor.py` + tiers + `quant.py`） | 操作标准 transformers `Cache`，用 fake-quant（量化-反量化）让模型据低比特栅格做真实前向 | 真实精度 + 真实逻辑显存（按三层位宽解析折算）+ 真实峰值显存 | **不报 decode 加速比**（没有真 kernel；伪量化不降峰值显存） |
| **阶段 B**（`kernels/packing.py` + `kernels/attention_triton.py`） | 真实低比特打包驻留 HBM + 单遍混精融合 decode 注意力 kernel（片上反量化，不反量化回 HBM，不做 dense 回退） | 真实 kernel 级加速比；Qwen2-1.5B 端到端小样例另见 `scripts/bench_qwen2_phasekv.py` | kernel 微基准不代表模型端到端加速比 |

---

## 4. 目录结构

```
release/kvcomp/
├── README.md                  本文件：项目总览 + 快速开始
├── requirements.txt           依赖清单（torch/triton 由环境提供，见 §5）
├── src/phasekv/               PhaseKV 方法实现
│   ├── __init__.py            导入即 register("phasekv", PhaseKVCompressor)
│   ├── compressor.py          PhaseKVCompressor(KVCompressor)，接 eval 接口（阶段 A）
│   ├── planner.py             roofline 预算分配器（3.2）
│   ├── quant.py               KIVI 式 per-channel K / per-token V 伪量化 + 逻辑字节折算
│   ├── hetero_cache.py        三层异构缓存状态 + 底层算子（GQA 聚合/选择/eviction）
│   ├── prompt_tier.py         prefill 注意力选择 + 低比特打包（3.1 Prompt 层）
│   ├── decode_tier.py         decode FP16 热环 + 懒批冷压（3.1 Decode 层）
│   └── kernels/               【阶段 B】真实加速
│       ├── packing.py         2/4/8-bit sub-byte 打包原语 + PackedTensor
│       ├── attention_triton.py 单遍混精融合 decode 注意力 Triton kernel（含 split-K / 三层融合）
│       └── reference.py       kernel 数值基准（正确性自检）
├── baselines/                 【对比基线】忠实复刻，均 register 注入
│   ├── snapkv.py  kivi.py  scope.py  pyramidkv.py
│   └── _common.py
├── eval/                      【评估框架】方法无关的评估流水线
│   ├── interface.py           KVCompressor 接口契约 + CompressionConfig
│   ├── compressors/           参考实现（full/recent/streaming/kvquant/h2o/hetero）+ 注册表
│   ├── runtime.py             模型加载 + 带压缩生成循环 + 真实计时/显存（真实位置 RoPE）
│   ├── metrics.py  datasets.py  report.py  runner.py  run_benchmark.py (CLI)
│   └── accuracy_per_byte.py   prompt/generated accuracy-per-byte 拆分
├── tests/                     【回归测试】39 用例（含 5 端到端）
├── scripts/                   【实验脚本】下载 / 跑分 / 画图 / 测速
│   ├── download_models.py  download_data.py     模型 / LongBench 数据下载
│   ├── run_eval.py  run_longbench.py            评估 / LongBench 主结果
│   ├── roofline_measure.py  bench_decode_kernel.py  图1 roofline / 阶段B kernel 测速
│   ├── qa_significance.py                         统计显著性分析
│   └── smoke_eval.sh
├── results/                   实验产物（*.json/*.md/*.csv）
├── docs/
│   ├── idea/                  PhaseKV 技术原理、相关工作与实验结果
│   └── delivery/              使用与复现指南
```

---

## 5. 环境与安装

已验证的复现环境如下，完整 Python 包快照见 `results/_qa_pip_freeze.txt`：

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 SUPER（32760 MiB）/ 驱动 595.71.05 |
| OS | Ubuntu（Linux x86_64） |
| Python | 3.10.8（miniconda） |
| torch | 2.1.2+cu121（CUDA 12.1，**环境提供，勿改动**） |
| triton | 2.1.0（阶段 B kernel） |
| transformers | 4.46.3（新式 RoPE，支持 KV 驱逐，**勿降级**） |

> **关键版本约束**：`transformers` 必须不低于 4.46。PhaseKV 使用驱逐后独立于物理缓存长度的真实位置索引，旧式 RoPE 接口无法满足这一要求。

安装 Python 依赖（在既有 CUDA 环境中，torch/triton 已随环境提供、不重装）：

```bash
cd release/kvcomp
pip install -r requirements.txt
```

统一环境变量（离线本地、模型缓存约定）：

```bash
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1                 # 已下载模型后离线运行
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
```

> `eval/__init__.py` 已 `setdefault` HF_HOME=`/root/autodl-tmp/hf-cache` 与 HF_ENDPOINT=`https://hf-mirror.com`，显式环境变量仍然优先。

---

## 6. 快速开始

所有命令均在 `release/kvcomp/` 目录下执行。

### 6.1 验证方法注册

```bash
python -c "import eval.compressors as c, baselines, src.phasekv; print(sorted(c.available()))"
# 期望包含: full, h2o, hetero, kivi, kvquant, phasekv, pyramidkv, recent, scope, segment_ablation, snapkv, streaming
```

### 6.2 冒烟评估（0.5B 真实端到端）

```bash
# 非注意力方法（sdpa）
python -m eval.run_benchmark --model Qwen/Qwen2.5-0.5B-Instruct --preset full \
    --prompt-tokens 512 --max-new-tokens 32 --ppl-source builtin --out results/smoke_full
```

### 6.3 作为库使用：接入你自己的压缩方法

```python
from eval.interface import KVCompressor, CompressionConfig
from eval.compressors import register

class MyCompressor(KVCompressor):
    requires_attention = False        # prefill/decode 需要注意力分数则 True（走 eager）
    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        ...   # 压缩 prompt KV，返回 cache
        return cache
    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        ...   # 维护/压缩增长的 KV，返回 cache
        return cache
    def cache_memory_bytes(self, cache):
        ...   # 量化类需覆盖，按 bit 折算逻辑显存

register("mymethod", MyCompressor)
```

然后：`python -m eval.run_benchmark --model <m> --methods-json '[{"method":"mymethod",...}]'`

### 6.4 直接构造 PhaseKV 压缩器

```python
import src.phasekv                              # 导入即注册
from eval.compressors import build_compressor

comp = build_compressor("phasekv",
    prefill_keep_ratio=0.3,                     # prompt 层保留比例
    extra={"prompt_bits": 2, "cold_bits": 4, "recent_window": 32})
# 或开启 roofline planner 自动配超参:
comp = build_compressor("phasekv", extra={"use_planner": True, "target_memory_ratio": 0.3})
```

---

## 7. 复现主要实验

完整的复现命令和参数说明见 **[`docs/delivery/USAGE.md`](docs/delivery/USAGE.md)**。主要命令如下：

```bash
# M1 主结果：7B LongBench qa_f1 Pareto（5 任务 × n=32 = 160 样，1536 中部截断）
HF_HUB_OFFLINE=1 python scripts/run_longbench.py \
  --model /root/autodl-tmp/modelscope-cache/Qwen/Qwen2___5-7B-Instruct \
  --tasks narrativeqa,multifieldqa_en,hotpotqa,2wikimqa,qasper \
  --n-samples 32 --max-length 1536 \
  --methods-json '[{"method":"full"},{"method":"phasekv","prefill_keep_ratio":0.3,"prompt_bits":2,"cold_bits":4,"recent_window":32},{"method":"kivi","bits":2,"group_size":32,"recent_window":64},{"method":"snapkv","prefill_keep_ratio":0.3,"obs_window":16},{"method":"pyramidkv","prefill_keep_ratio":0.3,"obs_window":16,"beta":20},{"method":"scope","recent_window":16,"max_cache_tokens":256}]' \
  --out results/m1_longbench_7b

# 阶段 B kernel 加速微基准（真实 CUDA events 计时）
python scripts/bench_decode_kernel.py

# 当前真实 PhaseKVCache vs 普通 DynamicCache（内置六题、贪心解码、GPU 实测）
python scripts/bench_qwen2_phasekv.py --prompt-bits 2 --prefill-keep-ratio 0.5 \
  --out results/qwen2_phasekv_comparison

# 专测长回复的随机重复；每种方法使用相同的固定种子
python scripts/bench_qwen2_phasekv.py --case-id long_followup --mode sample \
  --seeds 5 --max-new-tokens 256 --prompt-bits 2 --prefill-keep-ratio 0.5 \
  --out results/qwen2_phasekv_long_sample

# 压缩强度消融：Prompt 全保留、8-bit
python scripts/bench_qwen2_phasekv.py --prompt-bits 8 --cold-bits 8 \
  --prefill-keep-ratio 1 --decode-keep-ratio 1 \
  --out results/qwen2_phasekv_8bit_comparison

# 隔离 Prompt 位宽与 token 保留率的影响（真实 PhaseKVCache，六题）
python scripts/ablate_qwen2_prompt.py \
  --out results/qwen2_phasekv_prompt_ablation

# Prompt 全保留且 4-bit：对比不冷却、冷层量化、冷层 token 筛选
python scripts/ablate_qwen2_cold.py \
  --out results/qwen2_phasekv_cold_ablation

# 图1 roofline 标定
python scripts/roofline_measure.py

# 回归测试（39 用例，含 5 端到端；需 CUDA+模型）
python -m pytest tests/ -q
# 纯逻辑子集（无需 GPU）
python -m pytest tests/test_metrics.py tests/test_interface.py tests/test_compressors.py -q
```

`bench_qwen2_phasekv.py` 比较的是 `qwen2.py` 使用的真实打包缓存；上述 `run_longbench.py`
仍使用阶段 A 的 fake quant。新脚本每题先同配置预热，再记录答案命中、重复 4-gram、
token 上限、setup/TPOT、峰值显存和常驻 KV 字节，保存同名 `.json` 与 `.md`。
可用 `--cases-jsonl` 指向自定义题目，每行格式为
`{"id":"case1","prompt":"问题","answers":["标准答案"]}`；也可用 `messages` 数组代替 `prompt`。
`ablate_qwen2_prompt.py` 只比较贪心回答与 KV 大小，不报告未经逐配置预热的速度。
`qwen2.py` 的交互默认配置为 Prompt 4-bit、100% 保留；首轮 prefill 后复用同一份
PhaseKVCache，后续只输入新用户消息的 chat 模板 token，并将它们按 Prompt 位宽打包。
`/clear` 才清空会话缓存。`ablate_qwen2_cold.py` 固定同一轮长回复前缀，隔离测试
冷层量化和 50% token 筛选；结果见 `results/qwen2_phasekv_cold_ablation.md`。

实际 Qwen2 解码路径的 `fused_phasekv_decode` 默认自动选择并行参数。规则位于
`src/phasekv/kernels/launch_config.py`：按每个 prompt、追加 prompt、cold 段的实际保留
token 数、query head 数、head_dim 和 GPU SM 数，独立选择拆分数量、tile 大小与 warp 数。
自动 tile 控制在约 1024 个元素内，以限制反量化时的寄存器压力；拆分边界按 tile 对齐，
并限制最大拆分数量以控制合并开销。设备信息和规划结果会缓存，不在每个 token 上试跑调参。
这是有界启发式规则，并非所有 GPU 上的最优配置保证。
例如 RTX 3070 Ti Laptop（46 SM）、Hq=12、HD=128 时，128/512/8192/28672 token
分别使用 16/16/64/128 个 split，BLOCK_N=8、num_warps=4。
可以分别覆盖任意参数；复现旧配置请显式传入
`num_splits=8, block_n=64, num_warps=4`。
`python scripts/bench_phasekv_launch.py` 可对照旧配置与自动配置，记录内核时间、
含 split 合并的时间及编译器寄存器/溢出信息，默认保存到 `results/phasekv_auto_launch.json`。
该微基准用 CUDA graph 计时，数据驻留 GPU，不代表完整模型的 token 生成速度。
2026-10-15 本机实测：8K/28K 的单层 packed 内核由 1.69/5.71 ms 降至
0.18/0.67 ms，编译器报告的 `n_spills` 从 596 降至 0。
完整 Qwen2-1.5B-Instruct 在相同初始 KV 快照（另有 368 个历史 token）上，
8 步预热后测 24 步贪心解码，中位速度由 13.5/5.4 提升至 44.5/27.0 token/s。
这是旧 PhaseKV 参数与自动参数的对照，不含 prefill 和编译耗时；原始数据见
`results/phasekv_auto_launch.json` 与 `results/phasekv_auto_decode.json`。

当前内置五道有标准答案的题中，100% 保留时 2/4/8-bit 分别命中 3/5、5/5、5/5；
固定 8-bit 时，100%/75%/50%/25% 保留分别命中 5/5、3/5、0/5、0/5。
这是用于定位问题的小样本结果，完整逐题输出见 `results/qwen2_phasekv_prompt_ablation.json`。

---

## 8. 主要结果

> 本节数字均来自 `results/` 下的真实测量文件。使用结果时需同时保留对应的数据集、样本量、显存定义和计时范围，具体见 [§9](#9-测量口径)。

### 8.1 M1 精度 Pareto —— Qwen2.5-7B + LongBench

来源 `results/m1_longbench_7b.{json,md}`（n=32/任务=160 样，1536 中部截断，eager）：

| 方法 | qa_f1 | KV 逻辑节省% | 压缩比× | 峰值显存MB | 对 full 一致率 |
| --- | --- | --- | --- | --- | --- |
| full | 15.62 | 0.06 | 1.00 | 16010 | — |
| kivi | 17.27 | 77.07 | 4.36 | 16010 | 0.62 |
| snapkv | 15.31 | 69.14 | 3.24 | 19716 | 0.75 |
| scope | 15.62 | 0.06 | 1.00 | 19729 | 1.00 |
| pyramidkv | 14.18 | 69.12 | 3.24 | 19715 | 0.72 |
| **phasekv** | **18.43** | **93.31** | **14.94** | 19715 | 0.62 |

**结果分析**：在 n=32 观测下 PhaseKV 任务分居首（18.43）；但更严格的 **每任务 100 条、总计 500 条的显著性复核**（R-13，`results/_qa_significance_7b`）显示 **各方法 qa_f1 差异均不显著**（PhaseKV vs full +0.256，bootP=0.776，95%CI 跨 0；总计 500 条样本时均值最高者为 KiVi 17.18）。因此当前数据支持的结论是：**在（逻辑 KV 显存, qa_f1）平面上，PhaseKV 以 93.3% 逻辑 KV 节省（14.9× 压缩）取得与 full 及主要基线统计不可区分的任务精度**，不支持“精度反超”的结论。
**结果范围**：`93.31%` 是按目标位宽折算的逻辑 KV 字节节省，不是峰值显存下降。阶段 A 的 phasekv 峰值显存为 19715MB，高于 full 的 16010MB；hotpotqa 多跳任务仍是相对弱点；0.5B 与 0.5B-LongBench 上 PhaseKV 不占优。每任务 100 条、总计 500 条的分层 bootstrap 完整结果见 `results/_qa_significance_7b.md`。当前评估使用前 n 条确定性样本、greedy 生成和固定 bootstrap 随机数生成器，因此修改 `--seed` 不会形成独立实验重复。

### 8.2 阶段 B 融合 kernel 加速 —— 真实测量

来源 `results/kernel_decode_bench.{json,md}`（RTX 4080S，CUDA events 中位数 of 100，单 batch/单 query 步/单层，per-token 量化）：

| 上下文 S | 2bit vs SDPA | 4bit vs SDPA | KV 读省%(2bit/4bit) | 数值误差 |
| --- | --- | --- | --- | --- |
| 4096 | 1.20× | 0.65× | 81.25% / 68.75% | ~6e-5 |
| 8192 | 1.42× | 1.09× | 81.25% / 68.75% | ~3e-5 |
| 16384 | 1.57× | 1.16× | 81.25% / 68.75% | ~3e-5 |
| 32768 | **1.63×** | 1.19× | 81.25% / 68.75% | ~1.5e-5 |

**结果分析**：2-bit 的 KV 读取节省为 81%，高于 4-bit 的 69%，且加速比随上下文长度增长，结果与 decode 带宽受限的分析一致。
**测量范围**：这是 **kernel 级 microbench**，非模型端到端 TTFT/TPOT；S=4096/4bit 的 0.65× 来自短上下文下的切分与合并开销。三层融合已验证正确性（maxerr ~1e-4），单 launch 优化与模型端到端 TTFT/TPOT 尚未测量。

### 8.3 图1 Roofline（`results/roofline_4080s.md`）

实测 RTX 4080S：HBM copy **625.5 GB/s**、纯读 **686.1 GB/s**、fp16 GEMM **110.2 TFLOPS**、经验 ridge **176.18 FLOP/Byte**。7B KV 读占 decode 带宽（ctx=131072）：batch1=33% / batch16=88.8% / batch64=96.9% ⇒ 长上下文/批量下 KV 读流量成为 decode 带宽主因。

---

## 9. 测量口径

1. 所有实验数字来自真实模型或真实 kernel 测量，不使用模拟数据代替测量结果。
2. 速度与加速比来自阶段 B 的真实 kernel microbenchmark。阶段 A 的 TTFT/TPOT 不作为加速结论，roofline 解析值也不替代实测结果。
3. 逻辑 KV 节省按目标位宽和量化元数据折算，不等同于阶段 A 的峰值显存下降。阶段 A 的物理张量仍为 FP16，真实低比特驻留由阶段 B 实现。
4. 压缩与融合路径不包含 dense 回退。配置或执行不满足约束时直接报错。
5. n=32 结果用于展示主实验表；精度关系采用 每任务 100 条、总计 500 条的分层 bootstrap 复核。PhaseKV 与 full 的差异不显著，因此结论是统计精度相当条件下的高压缩率，而不是精度提升。

---

## 10. 当前结果范围

| 项目 | 当前结果 |
| --- | --- |
| 7B 任务精度与逻辑 KV 压缩 | Qwen2.5-7B + LongBench 已完成；PhaseKV 达到 93.31% 逻辑 KV 节省，总计 500 条样本时与 full 精度差异不显著 |
| 低比特 kernel | 2/4-bit 打包、数值正确性与 CUDA Events microbenchmark 已完成；2-bit 在 S=4096–32768 达到 1.20–1.63× |
| Roofline 与读放大 | RTX 4080 SUPER 实测 roofline、KV 带宽占比与 prompt/generated 分段扫描已完成 |
| 消融 | prompt 选择、prompt 位宽、分层位宽和热窗口消融已完成 |
| 未覆盖测量 | 模型端到端 TTFT/TPOT、吞吐与能耗；Llama-3.1-8B 泛化；重要度信号的真实单步延迟 |

---

## 11. 文档索引

- [`docs/idea/PhaseKV-技术原理与系统设计.md`](docs/idea/PhaseKV-技术原理与系统设计.md)：方法动机、设计推导、实现细节、相关工作、实验结果与适用范围。
- [`docs/delivery/USAGE.md`](docs/delivery/USAGE.md)：环境配置、模型与数据准备、完整复现命令、自定义方法接入和常见问题。
- `results/`：实验 JSON、Markdown、CSV 和运行日志。

---

## 12. 相关工作对比

| 维度 | SCOPE | KIVI | MiniKV | Oaken | **PhaseKV** |
| --- | --- | --- | --- | --- | --- |
| 异构主轴 | phase | 精度×新近度 | 全局 | 全局 | **phase×生命周期×精度** |
| 压缩机制 | 驱逐 | 量化 | 选择+量化 | 量化 | **选择+量化按相位协同** |
| 设计依据 | 精度 | 精度 | 精度+系统 | 硬件（ASIC） | **硬件（读放大/roofline）** |
| 压缩时机 | 固定策略 | 固定策略 | 固定策略 | 固定策略 | **prefill 一次/decode 懒批** |
| prompt 只读专层 | 否 | 否 | 否 | 否 | **是** |
| 预算分配 | 手动 | 固定 | 固定 | 固定 | **roofline 成本模型** |
| 融合 kernel | 无 | 有 | 有 | ASIC | **三层混合精度单遍 kernel** |
| 平台 | GPU | GPU | GPU | ASIC | **GPU** |

完整分类、逐项机制差异和证据映射见 [`docs/idea/PhaseKV-技术原理与系统设计.md`](docs/idea/PhaseKV-技术原理与系统设计.md) 第 6 节。
