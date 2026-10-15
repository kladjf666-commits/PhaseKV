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

## 6. 真实 PhaseKVCache 从 prefill 到 decode 的调用链

本节对应 `src/phasekv/phasekv_cache.py`、`src/phasekv/qwen2_attention.py` 和 `qwen2.py` 的真实打包路径。第 5.3 节的 `PhaseKVCompressor` 是 fake-quant 评估入口；这里直接构造 `PhaseKVCache`，不调用 `prompt_tier.select_and_pack()` 或 `decode_tier.maintain()`。

### 6.1 各层缓存与三种长度

`PhaseKVCache.layers[i]` 是第 i 个模型层的 `LayerTiers`，其中：

| 字段 | 内容 | 更新时机 |
| --- | --- | --- |
| `prompt` | 首轮选择后、按 `prompt_bits` 打包的 prompt | `from_dense_prefill()` 创建，之后冻结 |
| `prompt_extensions` | 后续轮次新增输入的打包段 | 每次 `finish_prompt_chunk()` 追加一段 |
| `cold` | 生成 token 的冷段列表 | decode 达到懒批冷却阈值时追加 |
| `hot_k / hot_v` | FP16 连续缓冲区 `[B,Hkv,capacity,hd]` | 每次写入当前位置；冷却后搬移尾部 |
| `hot_len` | 热区实际有效的 token 数 | 写入时增加，冷却或块结束时减少 |

这里的热区是连续缓冲区，冷却时把尾部搬回起点，当前实现不是环形队列。正常 decode 时容量为 `recent_window + lazy_g`；新增 prompt 块可以临时扩容。

三个数字不能混用：

- `cache.true_seq_length`：已经前向并提交的真实 token 数，包含被驱逐的历史位置，用于下一次 `position_ids` 的起点。
- `cache.get_seq_length()`：第 0 层当前实际保留的 token 数，用于下一次物理 `cache_position` 的起点；不是热区容量，也不是原始上下文长度。
- `layer.hot_k.shape[2]`：热区已分配的容量；只有 `hot_views()` 返回的前 `hot_len` 个位置可以参与注意力。

真实长度只在 `finish_decode_step()` 或 `finish_prompt_chunk()` 推进。模型前向过程中，各层依次写入，物理长度可能暂时不同，因此应在前向开始前读取位置起点。

### 6.2 首轮 prefill

```text
加载 Qwen2 模型并 eval()
└─ install_qwen2_phasekv_attention(model)             安装一次

model(prompt_ids, past_key_values=DynamicCache(), use_cache=True, ...)
└─ 每层 PhaseKVQwen2Attention.forward()
   └─ 输入不是 PhaseKVCache → original.forward()    原始 prefill 注意力
      └─ DynamicCache.update()                     保存完整 K/V

build_phasekv_cache(prefill, model, prompt_len, ...)
├─ select_prompt_indices(prefill.attentions, ...)   keep_ratio<1 时选择
│  └─ 观察窗口投票 → GQA 聚合 → pooling → 逐 head 选择
└─ PhaseKVCache.from_dense_prefill(...)
   ├─ 校验原始 K/V 和 keep_indices
   ├─ _gather_kv()                                 保留各 head 的位置
   ├─ _pack_kv() → quantize_pack()                  K 沿 token 轴，V 沿 head_dim 轴
   └─ 每层建立 LayerTiers(prompt=..., hot_len=0)     cold 和扩展 prompt 为空
```

`from_dense_prefill()` 不计算注意力重要度，它接收调用方提供的下标；`keep_indices=None` 时保留全部 prompt。输入 K 已由原模型按真实位置施加 RoPE，压缩时不重新计算位置编码。真实打包支持 2、4、8 bit；不要把 fake-quant 路径允许的 3/16 bit 配置直接传进来。

需要 prompt 选择时，首次前向须取得每层注意力分数；本仓库示例使用 eager attention。全部保留时可以关闭 `output_attentions`。构建压缩缓存后，释放不再需要的 prefill 输出和 dense cache 引用，才能释放原稠密 KV 占用。

### 6.3 单 token decode

```text
从 prefill 最后位置的 logits 采样第一个生成 token y0
│  此时只得到 token ID；y0 的 K/V 尚未进入缓存
│
├─ 读取 true_seq_length 和 get_seq_length()，构造两种位置
├─ model(y0, past_key_values=cache, ...)
│  └─ 每层 PhaseKVQwen2Attention.forward()
│     ├─ q_proj / k_proj / v_proj
│     ├─ apply_rotary_pos_emb()                    使用真实绝对位置
│     ├─ Q/K/V 转为 FP16
│     ├─ cache.append_hot(layer_idx, K, V)         先写入当前 token
│     ├─ cache.layer_tiers(layer_idx)
│     ├─ fused_phasekv_decode(Q, tiers, Hq)        当前 Q 能看见自己的 K/V
│     │  ├─ 读取 prompt、prompt_extensions、cold 的 packed 段
│     │  ├─ hot_views() 读取有效热区
│     │  └─ 合并各段的 softmax partial
│     └─ o_proj
├─ cache.finish_decode_step()                     模型所有层返回后，只调用一次
│  ├─ 检查每层都写入了一个 token
│  ├─ 每层 _cool_aged_tokens()
│  │  ├─ aged = hot_len - recent_window
│  │  ├─ aged < lazy_g → 保持热区，暂不压缩
│  │  └─ aged >= lazy_g
│  │     ├─ 只取最旧 aged 个生成 token
│  │     ├─ 按 value L2 范数逐 head 选择，并按时间排序
│  │     ├─ _pack_kv(cold_bits) → 追加新 cold 段
│  │     └─ 最近 recent_window 个 token 搬回热区起点
│  ├─ true_seq_length += 1
│  └─ 清空本步已写入层记录
└─ 从这次输出 logits 采样 y1，继续下一步
```

冷却在当前 token 的注意力完成之后执行。因此，当前 query 仍能看到这批即将被冷却、甚至被驱逐的 token。冷层选中数为 `max(1, round(decode_keep_ratio * aged))`；比例为 1 时不计算 value-norm，直接打包整批。首轮 prompt 和旧 cold 段都不重新量化。

例如 `recent_window=32, lazy_g=16`：前 47 个生成 token 都还在热区；第 48 个 token 写入后先计算注意力，再在 `finish_decode_step()` 处理最旧的 16 个，留下最近 32 个 FP16 token。若 `decode_keep_ratio=0.5`，这批 16 个中各 head 保留 8 个进入冷段。

下面是可放入现有 Qwen2 运行环境的最小贪心生成循环。前提是模型已加载到 CUDA、处于 FP16/BF16，输入为 batch=1、无 padding，首轮使用能返回 attention 的实现；这里显式选择 `keep_ratio=0.5`。实际对话的采样、EOS 和模板处理见 `qwen2.generate_reply()`。

```python
import torch
from transformers.cache_utils import DynamicCache
from src.phasekv import install_qwen2_phasekv_attention
from qwen2 import build_phasekv_cache

@torch.inference_mode()
def generate_with_phasekv(model, input_ids, max_new_tokens=16):
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须为正")
    model.eval()
    install_qwen2_phasekv_attention(model)
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    positions = torch.arange(prompt_len, device=device)
    prefill = model(
        input_ids=input_ids, past_key_values=DynamicCache(),
        use_cache=True, output_attentions=True,
        position_ids=positions.unsqueeze(0), cache_position=positions,
    )
    cache = build_phasekv_cache(
        prefill, model, prompt_len, prefill_keep_ratio=0.5,
        prompt_bits=2, cold_bits=4,
        recent_window=32, lazy_g=16, decode_keep_ratio=1.0,
    )
    token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del prefill
    generated = []
    for _ in range(max_new_tokens):
        generated.append(token)
        output = model(
            input_ids=token, past_key_values=cache,
            use_cache=True, output_attentions=False,
            position_ids=torch.tensor([[cache.true_seq_length]], device=device),
            cache_position=torch.tensor([cache.get_seq_length()], device=device),
        )
        cache.finish_decode_step()
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del output
    # 每个返回的 token 都已前向写入，缓存可接着用于下一轮输入。
    return torch.cat(generated, dim=1), cache
```

只需最终生成文本时，最后一个采样 token 可以不再前向；但若还要复用缓存进行下一轮对话，就需要把它补写入。上面为便于连续使用，统一把所有返回 token 都写入缓存，最后一次得到的下一 token 预测不再使用。

### 6.4 后续轮次追加 prompt

```text
qwen2.append_prompt_chunk(model, cache, new_token_ids)
├─ 记录真实位置起点、物理长度起点
├─ cache.begin_prompt_chunk(length)
│  └─ 记录每层原 hot_len，必要时扩容
├─ model(整块新增 token, past_key_values=cache, ...)
│  └─ 每层 PhaseKVQwen2Attention.forward()
│     ├─ 整块投影并按真实位置施加 RoPE
│     └─ 对块内 token 依次执行
│        ├─ cache.append_prompt_token(layer_idx, K_i, V_i)
│        └─ fused_phasekv_decode(Q_i, cache.layer_tiers(layer_idx), Hq)
└─ cache.finish_prompt_chunk()
   ├─ 检查所有层已写完该块
   ├─ 块开始前剩余的生成热 token → 原样保留为 FP16，不转入 cold
   ├─ 本次新增输入 → 全保留、prompt_bits 打包到 prompt_extensions
   ├─ 恢复正常热区容量，复制原生成 KV，hot_len 恢复为块开始前的长度
   └─ true_seq_length += length，结束块状态
```

新增块没有首轮的 attention 选择。新增 prompt 不推进生成 token 的热窗口，不触发冷却，也不清空已积累的待冷却 token；后续 decode 继续在 `hot_len - recent_window >= lazy_g` 时按 `decode_keep_ratio` 选择并打包。`length=1` 仍然可以作为 prompt 块；阶段由 `begin_prompt_chunk()` 显式决定，不由 token 数自动判定。

虽然新增输入一次送入模型，但适配器在每层内部逐 token 写入、逐 query 计算注意力，保证不会读到该块未来的 K/V。已经冻结的不同类型段按用途保存；当前单 query 注意力只读取可见历史，各段使用一致的 K/V 配对并合并 softmax，所以不要求列表拼接顺序等于多轮时间顺序。位置关系已编码在 K 的 RoPE 中。

### 6.5 调用职责与错误定位

| 调用位置 | 应负责的函数 |
| --- | --- |
| 模型加载后 | `install_qwen2_phasekv_attention()` |
| 首次 prefill 返回后 | `build_phasekv_cache()` → `from_dense_prefill()` |
| attention 每层内部 | `append_hot()` 或 `append_prompt_token()` → `layer_tiers()` → `fused_phasekv_decode()` |
| 单 token 模型前向返回后 | `finish_decode_step()`，全模型一次 |
| 后续输入块前后 | `begin_prompt_chunk()` / `finish_prompt_chunk()`，可用 `qwen2.append_prompt_chunk()` 封装 |
| 步间统计 | `physical_nbytes()`；计入热区预留空间，不计临时工作区和模型权重 |

不要在外部生成循环重复调用 `append_hot()`：适配器已经逐层写入。也不要在每个 attention 层末尾调用 `finish_decode_step()`：此时后面的模型层可能尚未写入。`update()` 的报错表示接到了普通 dense attention 接口，需要检查适配器是否安装。

参数错误会给出具体字段或形状；状态错误会说明进行中的阶段和需要完成的步骤。热区满只报告实际有效长度与容量，不能仅凭容量推断某次前向遗漏了结束调用。这里没有中途失败的事务回滚；若模型前向或打包本身异常中断，不应直接重跑整个前向到同一份部分写入的缓存上。

`reorder_cache()` 只实现存储的 batch 重排，不表示当前 batch=1 的 Qwen2 attention 已支持 beam search。`fused_phasekv_decode()` 当前逐段启动计算再合并 partial，多冷段会产生多次 kernel launch；函数名不意味着整份多段缓存只启动一个 kernel。

## 7. 启动本地 Qwen2 对话

在项目根目录使用项目虚拟环境运行：

```bash
.venv/bin/python qwen2.py --check-cuda
.venv/bin/python qwen2.py --local-files-only --max-new-tokens 512
```

`--check-cuda` 会尝试执行 GPU 张量操作并检查 BF16 支持，通过后输出 GPU、PyTorch 和 CUDA runtime 信息。`--local-files-only` 使用本地缓存中的 `Qwen/Qwen2-1.5B-Instruct`，不访问网络；也可通过 `--model /path/to/model` 指定本地模型目录。

出现“你：”后输入消息即可连续对话，`/clear` 清空历史与缓存，`/exit` 退出。`--max-new-tokens` 是每轮最多生成的 token 数。每个输出 token（包括达到上限的最后一个）都会通过 `decode_token()` 完成模型前向和 `finish_decode_step()`，随后写入模板结束符，下一轮可以继续复用同一份缓存。新增输入和模板结束符均不清空原有生成热窗口。

若当前进程报告无法使用 CUDA，先在本机终端检查 `nvidia-smi`。本机正常、隔离环境失败时，应给运行进程提供 GPU 设备访问；不能仅凭隔离环境里的 `torch.cuda.is_available() == False` 判断驱动损坏。本机终端可用时，直接在该终端运行上面的启动命令。
