# PhaseKV：使用与复现指南

---

## 1. 适用范围

本仓库是一个 **方法无关的 KV Cache 压缩评估框架 + PhaseKV 方法实现 + 忠实竞品基线** 的一体化研究代码库。你可以用它来：

1. **复现 PhaseKV 的实验结果**（7B 精度 Pareto、阶段 B kernel 加速、roofline 和统计显著性）。
2. **把你自己的 KV 压缩方法**接入统一评估流水线，与 full / KIVI / SnapKV / SCOPE / PyramidKV / PhaseKV 同表对比。
3. **在给定目标显存下用 roofline planner 自动推导 PhaseKV 超参**。

---

## 2. 环境搭建

### 2.1 已验证的复现环境


| 项            | 值                                                      |
| ------------ | ------------------------------------------------------ |
| GPU          | NVIDIA GeForce RTX 4080 SUPER（32760 MiB）/ 驱动 595.71.05 |
| OS           | Ubuntu（Linux 5.15 x86_64）                              |
| Python       | 3.10.8（miniconda）                                      |
| torch        | 2.1.2+cu121                                            |
| triton       | 2.1.0                                                  |
| transformers | 4.46.3（新式 RoPE，**必须 ≥ 4.46**）                          |


> **关键版本**：`transformers` 必须不低于 4.46。PhaseKV 使用驱逐后独立于物理缓存长度的真实位置索引，旧式 RoPE 接口无法满足这一要求。

### 2.2 安装依赖

`torch` / `triton` 与 CUDA 强绑定，由既有环境提供，**不要用本仓库的 requirements 重装**：

```bash
cd release/kvcomp
python -c "import torch,triton;print('torch',torch.__version__,'triton',triton.__version__)"  
pip install -r requirements.txt   # transformers/datasets/matplotlib 等（不含 torch/triton）
```



---

## 3. 模型与数据准备


| 资源                    | 本地路径                                                                | 用途                      |
| --------------------- | ------------------------------------------------------------------- | ----------------------- |
| Qwen2.5-7B-Instruct   | `/root/autodl-tmp/modelscope-cache/Qwen/Qwen2___5-7B-Instruct`      | 主力                      |
| Qwen2.5-0.5B-Instruct | `/root/autodl-tmp/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct` | 冒烟 / 端到端自检              |
| LongBench jsonl       | `/root/autodl-tmp/hf-cache/longbench/data/`                         | qa_f1 任务（各任务 150~200 样） |


```bash
# 模型下载（魔塔源；positional 传模型 ID，--help 看全部参数）
python scripts/download_models.py Qwen/Qwen2.5-7B-Instruct --source modelscope
python scripts/download_models.py Qwen/Qwen2.5-0.5B-Instruct --source hf

# LongBench 数据（默认拉 qa_f1 任务集；--no-wikitext 跳过 WikiText）
python scripts/download_data.py --no-wikitext
```

---

## 4. 复现各实验

### 4.1 M1 主结果 · 7B LongBench 精度 Pareto（`results/m1_longbench_7b`）

5 任务 × n=32 = 160 样，1536 中部截断，六方法同 iso-context：

```bash
HF_HUB_OFFLINE=1 python scripts/run_longbench.py \
  --model /root/autodl-tmp/modelscope-cache/Qwen/Qwen2___5-7B-Instruct \
  --tasks narrativeqa,multifieldqa_en,hotpotqa,2wikimqa,qasper \
  --n-samples 32 --max-length 1536 \
  --methods-json '[{"method":"full"},{"method":"phasekv","prefill_keep_ratio":0.3,"prompt_bits":2,"cold_bits":4,"recent_window":32},{"method":"kivi","bits":2,"group_size":32,"recent_window":64},{"method":"snapkv","prefill_keep_ratio":0.3,"obs_window":16},{"method":"pyramidkv","prefill_keep_ratio":0.3,"obs_window":16,"beta":20},{"method":"scope","recent_window":16,"max_cache_tokens":256}]' \
  --out results/m1_longbench_7b
```

- 参数：`--tasks` 逗号分隔任务名；`--n-samples` 每任务样本数；`--max-length` prompt 中部截断预算；`--methods-json` 方法数组（同名方法可加 `"name"` 字段打 label）；`--list-methods` 列出可用方法后退出。
- 峰值显存：full/kivi ~16GB；需 `output_attentions` 的方法 ~19.7GB（32G 无 OOM）。

### 4.2 R-13 统计显著性 · 逐样本 qa_f1 + bootstrap CI + 配对检验（`results/_qa_significance_7b`）

在 M1 基础上额外做每方法 95% 分层 bootstrap CI + vs full 配对（差 CI / bootstrap p / 二项符号检验 / 胜平负）：

```bash
python scripts/qa_significance.py \
  --model /root/autodl-tmp/modelscope-cache/Qwen/Qwen2___5-7B-Instruct \
  --n-samples 100 --seed 1234 --n-boot 10000 \
  --out results/_qa_significance_7b
# 该命令每任务使用 100 条样本，5 个任务总计 500 条；执行 10000 次分层 bootstrap
```



### 4.3 图1 · Roofline 标定（`results/roofline_4080s`）

```bash
python scripts/roofline_measure.py --measured-model Qwen/Qwen2.5-0.5B-Instruct \
  --analytic-models Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-7B-Instruct \
  --out results/roofline_4080s
# 7B decode 真实落点：模型就绪后加 --measured-model 指向 7B 复跑
```

产出：实测 HBM 带宽 / fp16 GEMM 峰值 / ridge point + prefill 算力利用率 + 7B/0.5B 各上下文 KV 读占比（解析）。

### 4.4 阶段 B · decode kernel 加速微基准（`results/kernel_decode_bench`）

```bash
python scripts/bench_decode_kernel.py \
  --num-kv-heads 4 --num-q-heads 28 --head-dim 128 --group-size 32 \
  --seqs 4096 8192 16384 32768 --bits 2 4 --warmup 25 --iters 100 \
  --out results/kernel_decode_bench
```

产出：packed 低比特 KV split-K flash-decode kernel vs fp16 SDPA 的真实 CUDA-events 中位数加速比（2bit 1.20→1.63×）。**kernel 级 microbench，非端到端 TTFT/TPOT。**

### 4.5 Prompt/Generated KV 分段分析（`results/accuracy_per_byte`）

分别只压缩 prompt-KV 与 generated-KV，扫描保留比例并比较逻辑字节和精度。实现位于 `eval/accuracy_per_byte.py`，当前结果见 `results/accuracy_per_byte.{json,md}`，使用 Qwen2.5-0.5B-Instruct。该结果用于分析缓存字节分布，不替代 7B LongBench 的任务精度结论。

### 4.6 评估（0.5B）

```bash
bash scripts/smoke_eval.sh               # 纯逻辑单测 + full(sdpa) + attention(eager) + 端到端
# 或单独跑：
python -m eval.run_benchmark --model Qwen/Qwen2.5-0.5B-Instruct --preset full \
    --prompt-tokens 512 --max-new-tokens 32 --ppl-source builtin --out results/smoke_full
```

### 4.7 回归测试

```bash
python -m pytest tests/ -q              # 39 用例（含 5 端到端，需 CUDA+模型）
python -m pytest tests/test_metrics.py tests/test_interface.py tests/test_compressors.py -q  # 纯逻辑，无需 GPU
```

---

## 5. 作为库使用

### 5.1 验证方法注册

```bash
python -c "import eval.compressors as c, baselines, src.phasekv; print(sorted(c.available()))"
# 期望含: full, h2o, hetero, kivi, kvquant, phasekv, pyramidkv, recent, scope, segment_ablation, snapkv, streaming
```

### 5.2 接入你自己的压缩方法

实现 `KVCompressor` 接口三个生命周期方法并 `register` 即可入表：

```python
from eval.interface import KVCompressor, CompressionConfig
from eval.compressors import register

class MyCompressor(KVCompressor):
    requires_attention = False        # prefill/decode 需要注意力分数则 True（runtime 走 eager）
    physically_evicts = True          # 是否物理改变缓存张量形状（纯量化=False）

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        ...                           # 压缩 prompt KV
        return cache
    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        ...                           # 维护/压缩增长的 KV
        return cache
    def cache_memory_bytes(self, cache):
        ...                           # 量化类须覆盖：按 bit 折算逻辑显存

register("mymethod", MyCompressor)
```

然后 `python -m eval.run_benchmark --model <m> --methods-json '[{"method":"mymethod",...}]'`。

### 5.3 直接构造 PhaseKV / 启用 planner

```python
import src.phasekv                              # 导入即 register("phasekv", ...)
from eval.compressors import build_compressor

# 手调档
comp = build_compressor("phasekv", prefill_keep_ratio=0.3,
    extra={"prompt_bits": 2, "cold_bits": 4, "recent_window": 32})

# roofline planner 自动配超参（在目标显存比下推导相位异构配置）
comp = build_compressor("phasekv",
    extra={"use_planner": True, "target_memory_ratio": 0.3, "decode_len_hint": 256})
```

PhaseKV 关键超参：`prompt_bits`（prompt 只读层位宽）、`cold_bits`（decode 冷层位宽）、`prefill_keep_ratio`（prompt 选择保留比）、`recent_window`（decode 热层 FP16 窗口）、`lazy_g`（冷层懒批步长）、`group_size`（量化分组）。

---



---



