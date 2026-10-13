# PhaseKV：技术原理与系统设计

本文说明 PhaseKV 的设计依据、实现机制、相关工作和实验结果。所有定量结果均来自 `results/` 下的真实测量文件；解析模型只用于配置规划，不替代实测结果；速度数据仅采用真实低比特 kernel 的 CUDA Events 测量结果。

---

## 目录

- [1. 方法概述](#1-方法概述)
- [2. 问题与现有方法](#2-问题与现有方法)
- [3. 硬件与访问特征](#3-硬件与访问特征)
- [4. PhaseKV 系统设计](#4-phasekv-系统设计)
- [5. 实现细节](#5-实现细节)
- [6. 相关工作对比](#6-相关工作对比)
- [7. 实验结果](#7-实验结果)
- [8. 组件消融](#8-组件消融)
- [9. 测量范围](#9-测量范围)
- [10. 结论](#10-结论)

---

## 1. 方法概述

PhaseKV 按访问方式将 KV Cache 分为 prompt 层、decode 热层和 decode 冷层。prompt 层在 prefill 结束后完成 token 选择和低比特量化，之后不再改写；新生成的 token 先进入 FP16 热层，滑出窗口后再分批转入低比特冷层。Roofline 预算分配器用于确定各层的保留比例、位宽和窗口大小，融合注意力 kernel 直接读取三种表示。

---

## 2. 问题与现有方法

### 2.1 KV Cache 是 decode 阶段的带宽瓶颈

自回归解码时，每生成一个 token，模型都要重新读取到目前为止累积的**整份 KV Cache**做注意力。KV Cache 的体积随「上下文长度 × batch × 层数 × KV head 数 × head_dim × 2(K/V) × 精度字节」线性增长；在长上下文/大 batch 下，它从 HBM 读入的流量会**压过权重读取**，成为单步 decode 延迟的主导项。

`results/roofline_4080s.{json,md}` 分别记录实测硬件峰值、0.5B 模型实测落点和基于真实模型配置的解析结果。对 Qwen2.5-7B（28 层 / 4 KV head / head_dim=128）的解析计算显示，在上下文 ctx=131072 时，KV 读取占 decode 总读取字节的比例为 **batch1=33.0% / batch16=88.8% / batch64=96.9%**。该结论用于说明长上下文和批量场景下的 KV 读取压力，不等同于 7B 模型端到端带宽落点实测。

### 2.2 现有方法的缓存划分方式

“不同的 KV 使用不同表示”已经被多类工作探索，但各方法选择的异构主轴并不相同：

| 工作 | 缓存划分维度 | 设计依据 |
| --- | --- | --- |
| PyramidKV / PolyKV | **层**（层间金字塔/路由预算） | 精度 |
| KIVI | **精度 × 新近度**（旧的量化、新的 FP16 残差） | 精度 |
| DuoAttention | **head**（retrieval head vs streaming head） | 精度（需训练识别 head） |
| Quest | **query**（按 page min/max 只载 top-K page） | 带宽（省带宽不省容量） |
| **SCOPE / FastKV** | **phase**（prefill 保留 / decode 驱逐） | **精度**（heavy-hitter 在 decode 偏移） |
| MiniKV | 全局（选择 + 统一 INT2 + 融合 kernel） | 精度 + 系统 |
| Oaken | 全局（在线-离线混合量化 + ASIC） | 硬件（ASIC 路线） |

仅在 prefill 和 decode 之间切换策略并不足以形成完整的方法差异。SCOPE 等 phase 类工作已经采用阶段化策略，但仍存在两个边界：

1. **它们只切换「选择策略/预算」，用的是单一机制（驱逐），不触及压缩「机制本身」和「存储布局」**；
2. 它们主要根据精度变化分配预算，没有直接使用访存流量确定缓存布局。

### 2.3 PhaseKV 的设计范围

> 在本文纳入比较的工作中，PhaseKV 的主要区别是从访问放大和阶段瓶颈出发，将压缩机制、存储布局与执行时机按 phase 协同设计，并结合 roofline 预算模型和单遍混合精度融合 kernel。

PhaseKV 使用硬件成本确定缓存处理方式。下一节分别说明 roofline、读取次数和信号获取成本。

---

## 3. 硬件与访问特征

本节分析 prefill 与 decode 的执行特征、不同 KV 区段的读取次数，以及重要度信号的获取开销。解析关系实现在 `src/phasekv/planner.py`，实验数据位于 `results/`。

### 3.1 Prefill 与 Decode 的执行特征

**现象**：同一个模型，prefill（处理长 prompt）和 decode（逐 token 生成）跑在硬件的**两个不同 regime** 上。

**底层机制（算术强度分析，`planner.py: prefill_attention_intensity / decode_attention_intensity`）**：

- **prefill 自注意力**：QK^T 与 AV 各约 `2·B·H·P²·d` FLOPs，而写入 HBM 的 KV 约 `2·B·H_kv·P·d·2字节`。算术强度 ∝ P（随 prompt 长度增长），很快越过 roofline 的 ridge point ⇒ **计算受限**。
- **单步 decode 自注意力**：1 个 query 对 S 个 key，FLOPs 约 `2·2·B·H·S·d`，读 KV 约 `2·B·H_kv·S·d·2字节`。算术强度 ≈ `H/H_kv`（GQA 组大小量级），**与 S 无关**，通常远低于 ridge ⇒ **带宽受限**。

**硬件成本量化 + 实测证据**（`results/roofline_4080s.md`，RTX 4080S 真实微基准）：

- 实测硬件峰值：HBM copy **625.5 GB/s**、纯读 **686.1 GB/s**、fp16 GEMM **110.2 TFLOPS**、经验 ridge **176.18 FLOP/Byte**。
- **prefill 计算受限（0.5B 模型实测落点）**：算力利用率随 prompt 长度 **15.6%(P=512) → 36.6%(1024) → 64.9%(2048) → 77.0%(4096)**，单调逼近实测 GEMM 峰值。
- **decode 带宽压力（7B 解析结果）**：7B 单步 decode 的权重读取量为 15.23GB，按实测 copy 带宽 625.5GB/s 计算，理想带宽下界约为 **24.3ms**。随着上下文和 batch 增长，解析 KV 读取占比从 33.0% 增至 96.9%。当前未包含 7B 模型的真实 decode roofline 落点，因此该项作为硬件分析依据，不表述为端到端实测。

**对系统设计的影响**：

- prefill 阶段的全局选择与打包是一次性操作，其开销不会重复进入每个 decode step，因此更容易随生成长度摊销。当前数据未单独测量该操作对 TTFT 的增量。
- decode 阶段存在较高带宽压力：逐步压缩会增加关键路径上的读写量，因此冷层处理采用分批执行。

> PhaseKV 根据上述执行特征分别设计 prefill 和 decode 的缓存处理方式。SCOPE 主要根据精度变化调整阶段策略，PhaseKV 则将 roofline 和访存成本纳入设计。

### 3.2 Prompt-KV 与 Generated-KV 的读取次数

**现象**：同样是「压掉 1 字节 KV」，压 prompt 段和压生成段，在整段解码中省下的 HBM 流量**不一样多**。

**底层机制（读放大分析，`planner.py` docstring / `roofline_report`）**：

- 一个 **prompt token** 的 KV 在 prefill 时写入一次，随后在**每一个** decode step 都被读取一次 ⇒ **读放大 = T**（T 为解码长度）；而且它只写一次，写在带宽富余的 prefill。
- 第 i 个 **generated token** 的 KV 仅被其后的 (T−i) 步读取 ⇒ **平均读放大 ≈ T/2**；而且它是边写边用，写入发生在带宽吃紧的 decode。

**量化推论**：把 1 字节从 prompt 层压掉，在整段解码中省下的流量，约是把 1 字节从 decode 层压掉的 **2 倍**。因此**应给 prompt-KV 比 generated-KV 更激进的压缩预算（更低 bit + 更狠选择）**。

这里根据带宽和读取次数分配 prompt 与 generated KV 的预算，不假设 prompt token 的任务重要度更低。

**实测证据（`results/accuracy_per_byte.md`，Qwen2.5-0.5B，图 2 accuracy-per-byte 拆分）**：分别「只压 prompt-KV」和「只压 generated-KV」，扫描保留比例：

| 只压 prompt-KV | 总显存节省% | 对 full 一致率 |  | 只压 generated-KV | 总显存节省% | 对 full 一致率 |
| --- | --- | --- | --- | --- | --- | --- |
| prompt@0.1 | **78.40** | 0.05 |  | generated@0.1 | **11.68** | **1.00** |
| prompt@0.25 | 65.35 | 0.03 |  | generated@0.25 | 9.78 | 1.00 |
| prompt@0.5 | 43.61 | 0.01 |  | generated@0.5 | 6.52 | 1.00 |

**结果分析**：① **可压的字节主要在 prompt 层**（只压 prompt 就能省 78.40% 总显存，只压 generated 顶多省 11.68%）——因为在长上下文任务里 prompt 远长于生成段；② 结合读放大=T，**压 prompt 是「省得多 × 读放大高」的双重收益**。（0.5B token 一致率口径较严格，prompt 压缩的一致率约为 0.05；任务精度以 7B LongBench 结果为准，见 §7。）

### 3.3 重要度信号的获取开销

**现象**：要做「选择性压缩」就得先知道哪些 token 重要。但「拿到重要度信号」这件事本身，在 prefill 和 decode 的成本天差地别。

**底层机制**：

- **prefill**：可以复用 prefill 注意力计算产生的重要度信息执行 SnapKV 式“观察窗口投票 + per-head 选择”。在 eager attention 下可直接获取分数；在 FlashAttention/SDPA 路径中，显式输出完整注意力分数仍会产生额外存储和执行开销。
- **decode**：要拿注意力分数做选择，得显式 `materialize` 出 scores，这会**破坏 FlashAttention/SDPA 的融合**（它们从不把完整 attention matrix 写回 HBM），并额外占用本已吃紧的带宽——**昂贵**。

**实测证据（`results/smoke_attention.md`）**：需要注意力分数的方法（h2o/hetero）峰值显存比无信号方法（full/streaming）高 **21.6%**（1436/1435MB vs 1182/1180MB）。该结果只能说明 materialize scores 存在额外显存成本，当前数据不包含真实单步 decode 延迟，因此不据此报告延迟差。

**信号选择**：prefill 使用注意力分数，decode 使用无需完整注意力矩阵的信号。PhaseKV 的 decode 冷层因此改用 **value-norm**（Value 向量的 L2 范数，`hetero_cache.value_norm_signal`）作为重要度代理——直接对 `value_cache` 的 head_dim 轴取范数，不 materialize 任何注意力分数、不破坏 Flash/SDPA。

> 因此，prefill 和 decode 使用不同的重要度信号。SCOPE 在两个阶段均采用驱逐机制，没有单独处理信号获取成本。

---

## 4. PhaseKV 系统设计

PhaseKV 的各项设计由上一节的硬件分析推导。下表给出设计决策、依据与代码位置之间的对应关系。

| 设计内容 | 依据 | 代码位置 |
| --- | --- | --- |
| 按生命周期分「prompt / decode 热 / decode 冷」三层 | 阶段执行特征与读取次数 | `hetero_cache.py` |
| prompt 层：prefill 一次成型、选择+量化+**冻结** | 一次性成本可摊销，读取次数为 T | `prompt_tier.py` |
| decode 热层：最近窗口 FP16、追加 O(1) | decode 带宽压力和新 token 重要度未知 | `decode_tier.py` |
| decode 冷层：value-norm 信号 + **懒批**量化 | 避免完整注意力分数并摊薄处理开销 | `decode_tier.py` |
| prompt bit < cold bit（分层位宽） | prompt 读取次数高于 generated | `compressor.py` 默认 pb2/cb4 |
| roofline 预算分配器（字典序目标） | 阶段执行特征与读取次数的成本模型 | `planner.py` |
| 单遍混精融合 kernel（片上反量化） | decode 带宽压力，直接读取 packed 字节 | `kernels/attention_triton.py` |

### 4.1 三层缓存结构

KIVI 使用“旧 token 量化 + 最近 token FP16 残差”的两段结构，不单独区分 prompt 和 generated。PhaseKV 按**生命周期/访问模式**切三层（单层物理布局 `[b, H_kv, seq, hd]`，时间升序）：

```
[ 0 .. K0 )                 Prompt 层（frozen 只读）：per-head 注意力选择 + KIVI prompt_bits 量化，冻结不再改写
[ K0 .. K0+cold_len )       Decode 冷层（cooled）：滑出热窗口的老生成 token，懒批 value-norm 选择 + cold_bits 量化
[ K0+cold_len .. seq )      Decode 热层（hot）：最近 recent_window 个生成 token，FP16 全精度，追加 O(1)
```

- **Prompt 层**：它是读放大=T 的对象，适合采用更低位宽和保留比例；它在 prefill 一次成型，冻结后无 per-step 重量化开销。
- **Decode 热层和冷层**：新生成的 token 重要度未知，因此最近 `recent_window` 个先留 FP16（热层）；等它们滑出窗口、重要度可判后再冷却（冷层）。冷却时使用 value-norm，不生成完整注意力分数。

三层状态由 `HeteroTierState` 维护（`prompt_len_phys=K0` / `cold_len` / 三层位宽 / `group_size` / `recent_window`），是单条样本生命周期内的三层边界。

### 4.2 Prompt 层：选择 + 打包 + 冻结（`prompt_tier.select_and_pack`）

prefill 结束后一次成型，三步：

1. **选择**：用 prompt 末尾 `obs_window` 个 query 对各 key 的注意力**投票**（`observation_votes`），GQA 聚合到 kv head，再 `pool_scores` 做 1D 池化让被选 token 形成**连续簇**（保留局部上下文）。每个 kv head 各自保留 `sink（最前 sink_tokens 个）+ top-k 选中 + 观察窗口（最近 w 个）`，共 `K0 = max(sink, round(prompt_len × prefill_keep_ratio))` 个。
2. **打包**：对保留的 K/V 做 KIVI 式 per-channel(K)/per-token(V) 的 `prompt_bits` 位伪量化（`_pack_prompt`）。
3. **冻结**：成型后**不再改写**——避免每步重量化的累积误差与开销。

选择过程固定保留 attention sink（最前几个 token）和观察窗口，只对 `[sink, prefix_len)` 执行 top-k。StreamingLLM 和 SnapKV 的实验表明这两部分通常具有较高重要度，因此不参与 top-k 淘汰。

### 4.3 Decode 层：热环 FP16 + 懒批冷压（`decode_tier.maintain`）

- **热层**：最近 `recent_window` 个生成 token 保持 FP16，由模型前向自然追加，本模块不动它。
- **冷层懒批**：当滑出热窗口的 token 累积到 `≥ lazy_g` 才触发一次批量冷却——用 `value_norm_signal`逐 head 选保留 `decode_keep_ratio` 比例，驱逐其余，对保留者做 `cold_bits` 位量化后**冻结**；**永不触碰 prompt 层 `[0, K0)`**。

**懒批处理**：decode 每步都在和注意力读取抢带宽，如果每步都做一次选择+量化，压缩开销会叠加到每一步的关键路径上。懒批把冷压成本**摊薄到每 G 步一次**，且冷层一旦成型即冻结、后续不再重量化（避免反复 fake-quant 误差累积）。

### 4.4 分层位宽：prompt_bits < cold_bits

默认配置为 `prompt_bits=2`、`cold_bits=4`，因为 prompt 读放大=T 而 cold≈T/2。PhaseKV 与 MiniKV 统一 INT2 配置的区别是不是「统一低比特」，而是「按相位分层位宽 + 只读打包」。§8 的 `abl_uniform4` 消融用于比较：把 pb2/cb4 换成统一 pb4/cb4，qa_f1 从 18.43 掉到 14.59。

### 4.5 Roofline 预算分配器（`planner.plan`）

planner 使用一个**受约束的离散优化**（网格搜索 + 成本模型），输入 `(模型维度, prompt_len, decode_len, target_memory_ratio, group_size)`，输出**超参配置** `(prefill_keep_ratio / prompt_bits / recent_window / decode_keep_ratio / cold_bits / lazy_g)`。目标是**两层字典序**，受「稳态逻辑显存 ≤ 目标」约束：

1. **主目标「目标显存下精度最高」**：在可行域内**用满预算**，取逻辑字节（= 保留 token × 位宽）最大的配置。逻辑字节是精度的单调代理（保留越多 token、位宽越高，量化/驱逐误差越小）。这样可以避免在目标预算放宽后仍返回最低字节配置。
2. **次目标「decode 每步读字节最少」**：同等预算下最小化整段解码 KV 读流量（= Σ token 逻辑字节 × 该 token 读放大）。因 prompt 读放大=T、generated≈T/2，该次目标等价于「**把预算优先分给读放大低的 decode 层**」。

代码里字典序评分是 `feasible ? (0, -total, read_proxy, -ck, -cb) : (1, total, read_proxy, -ck, -cb)`：可行组（标志 0）恒排在不可行组（标志 1）前；可行组内 `-total` 最小即 total 最大（用满预算），再比 `read_proxy`（读流量），末位偏好读放大低的冷层高 keep/高 bit。

Planner 只输出超参数配置。精度、显存和加速比仍由评估流水线实测，解析成本模型的输出不作为实验结果使用。

### 4.6 KIVI 式量化：per-channel K / per-token V（`quant.py`）

沿用 KIVI 的关键观察：**Key cache 沿 channel 维度存在系统性离群值**，宜按 channel 量化；**Value cache 无显著 channel 离群**，宜按 token 量化。

- **K per-channel**：沿 token 轴分组，每个 (head, channel) 组内取 min/max ⇒ 同一 channel 的所有 token 共享一套量化参数，抵消 channel 离群。
- **V per-token**：沿 head_dim 轴分组，每个 (head, token) 组内取 min/max。
- `group_size`（默认 32）控制粒度；每组存 1 scale + 1 min（`SCALE_ZERO_BYTES=4`）。逻辑字节按「数据位 + 分组开销」精确折算（`quantized_tier_bytes`）。

### 4.7 阶段 A 与阶段 B

阶段 A 与阶段 B 验证的对象不同，需要分别报告：

| | 阶段 A | 阶段 B |
| --- | --- | --- |
| 存储 | **fake-quant**：量化后立即反量化回同 dtype，物理张量仍 fp16 | 真实低比特 sub-byte 打包驻留 HBM（`PackedTensor`） |
| 模型如何用 | 据低比特栅格做**真实前向**，得到量化对精度的**真实**影响 | 单遍混精融合 decode 注意力 kernel，只读 packed 字节、片上反量化 |
| 报告什么 | 真实精度 + 真实逻辑显存（按三层位宽折算）+ 真实峰值显存 | 真实 **kernel 级**加速比（HBM 读流量按位宽下降） |
| 结果边界 | 不报告 decode 加速比；伪量化不降低物理峰值显存 | 当前结果为 kernel 级测量，不等同于模型端到端 TTFT/TPOT |

**阶段 A 使用 fake-quant**：阶段 A 用于测量低比特量化对模型精度的影响。fake-quant 让模型在低比特栅格上执行真实前向；逻辑显存按目标位宽解析折算。物理显存与速度收益由阶段 B 的真实打包和 kernel 测量，阶段 A 结果不用于报告这两项收益。主结果表中 phasekv 的峰值显存 `19715MB > full 16010MB`（伪量化不降峰值），而「省 93.31%」明确标注为「逻辑 KV 字节节省」。

### 4.8 单遍混精融合注意力 kernel（阶段 B 系统贡献，`kernels/`）

一次注意力同时读三层：`[低比特打包 prompt] + [FP16 热窗口] + [低比特冷层]`，**在寄存器/共享内存内反量化**（compressed-domain execution），**不把整份 KV 反量化回 HBM、不做 dense 回退**。三件套：

- `packing.py`：2/4/8-bit 无损 sub-byte 打包原语 + 分组量化产 `PackedTensor`；与 `quant` 数值等价（自检保证），`PackedTensor.nbytes` 与 `quantized_tier_bytes` 口径一致 ⇒「阶段 A 逻辑显存 == 阶段 B 物理显存」。
- `attention_triton.py`：单遍混精融合 decode 注意力 Triton kernel（含 split-K flash-decode 与三层融合版）。
- `reference.py`：纯 torch 语义 golden，作为 kernel 数值基准（自检 maxerr ~1e-4）。

**压缩域执行与错误处理**：decode 带宽受限。若先将 packed KV 反量化并写回 HBM，再重新读取，会增加一次 HBM 写入和读取，抵消部分带宽节省。片上反量化才能把「省下的读流量」转化为加速。不做 dense 回退也使 PhaseKV 与采用“误差认证 + dense 回退”的分层融合 kernel 工作（arXiv 2605.20868）形成明确区别。PhaseKV 通过 token 选择、精度分配和热窗口控制数值质量。

---

## 5. 实现细节

PhaseKV 基于标准 transformers Cache 实现，并处理逐 head 选择、位置编码、字节统计和参数校验。

### 5.1 逐 head 选择，但保持物理张量矩形

不同 head 在 prompt/冷层可以保留**不同的 token 下标**（因为不同 head 关注的 token 不同），但同层各 head 必须保留**相同的数量**，否则物理张量 `[b, H_kv, budget, hd]` 就不再是矩形、无法用一个稠密张量表示。实现上，`evict_per_head` 用逐头 `gather` 保留各自下标，`topk_per_head` 保证每 head 取相同的 `select_k` 个（`hetero_cache.py`）。

### 5.2 驱逐后仍用「真实绝对位置」驱动 RoPE

驱逐后物理缓存长度 < 真实已见 token 数。若用物理下标去算 RoPE 的相对距离就会错位。约定（对齐 `eval/runtime.py`）：**保留下标按时间升序排序**（`evict_per_head` 内 `torch.sort`），后续新 query 由 runtime 以**真实绝对位置**施加 RoPE，物理 `cache_position` 只驱动写入/因果掩码。这依赖 `transformers >= 4.46` 提供的新式 RoPE 接口；旧式 4.44.2 无法在 KV 驱逐后将真实绝对位置与物理缓存长度解耦。

### 5.3 逻辑字节折算：精确到「数据位 + 分组开销」

`cache_memory_bytes` 直接返回 `HeteroTierState.logical_bytes`，它把三层分别按位宽折算：prompt(prompt_bits) + cold(cold_bits) + hot(fp16)。量化层字节 = `batch·layers·heads·tokens·head_dim·bits/8`（K、V 各算）+ 分组元数据 `(K组数+V组数)·4B`（`quantized_tier_bytes`）。**这保证「逻辑显存」是可复现的解析量，而不是估算**，也保证与阶段 B 物理打包字节口径一致。

### 5.4 超参数校验与错误处理

`compressor._validate()` 强制 `prompt_bits/cold_bits ∈ {2,3,4,8,16}`、`prefill_keep_ratio ∈ (0,1]` 及各正整数约束。非法配置直接抛错，不静默改写参数，也不回退到全精度路径。

### 5.5 与三个基线的实现级差异（忠实实现，非简化）

- **vs SnapKV**（只压 prefill、decode 全 FP16、无量化）：PhaseKV 给 prompt 层叠加较低位宽量化，且额外管理 decode 冷层。
- **vs KIVI**（统一量化、不分 prompt/generated、无选择）：PhaseKV 把 prompt 单列为只读、低位宽的选择与量化层，decode 内部再分热/冷。
- **vs SCOPE**（只驱逐不量化、只压生成段、prompt 全保留）：PhaseKV 根据读取次数对 prompt 使用更低位宽和保留比例，且叠加量化与阶段感知的压缩**时机**（prefill 一次 / decode 懒批）。

---

## 6. 相关工作对比

相关工作可以按照异构发生的维度以及设计依据是精度还是硬件成本进行分类。PhaseKV 的差异不在于简单地组合已有选择和量化方法，而在于以相位、生命周期和精度为联合主轴，并由读放大和 roofline 模型共同确定机制、布局、位宽与执行时机。

### 6.1 相位维度：SCOPE 与 FastKV

**SCOPE** 在 prefill 阶段保留完整缓存，在 decode 阶段使用滑动窗口和 heavy-hitter 驱逐。它与 PhaseKV 都区分 prefill 和 decode，但差异体现在四个方面：SCOPE 切换选择策略和预算，PhaseKV 同时切换选择、量化和存储布局；SCOPE 以 heavy-hitter 漂移解释精度，PhaseKV 以读放大和带宽瓶颈解释硬件成本；SCOPE 没有 prefill 一次成型与 decode 懒批之间的时机差异；SCOPE 也没有独立的只读 prompt 层。代码库中的 `baselines/scope.py` 使用相同评估流水线进行直接比较。

**FastKV** 将 prefill 的 token-selective propagation 与 decode KV 保留解耦，并分别设置预算。其异构仍停留在预算层面，没有进一步区分相位对应的压缩机制和物理布局，也没有以读放大驱动跨层预算分配。

### 6.2 层、Head 与 Query 维度

**PyramidKV** 通过层间金字塔预算分配实现层级异构，低层保留较多 token、高层保留较少 token，并在层内使用 SnapKV 式选择。**PolyKV** 进一步为不同层路由不同策略并分配非均匀预算。二者的主轴均为模型层，与 PhaseKV 的相位和生命周期主轴正交。

**DuoAttention** 将注意力头分为 retrieval head 和 streaming head，分别执行全量注意力和窗口注意力，异构发生在 head 维度，并需要训练过程识别 head 类型。**Quest** 根据当前 query 和 page 的 Key 范围估计重要度，只加载 top-K page，主要减少读取带宽而不减少缓存容量，异构发生在 query 维度。这些方法可以与相位异构组合，但不能替代 prompt、热层和冷层之间的表示差异。

### 6.3 精度与新近度维度：KIVI

**KIVI** 对 Key 采用 per-channel 量化，对 Value 采用 per-token 量化，并为最近 token 保留 FP16 residual。它形成“旧 token 量化 + 新近 token 全精度”的两段结构，但不区分 prompt 和 generated token。PhaseKV 继承其 K/V 非对称量化依据，同时把 prompt 单独组织为冻结的只读层，并在 decode 内继续区分热层与冷层。二者的主要区别是缓存划分依据：KIVI 依据精度和新近度，PhaseKV 进一步依据相位、生命周期和读放大分配位宽与保留比例。

### 6.4 选择与量化协同：MiniKV、SnapKV、H2O 与 StreamingLLM

**MiniKV** 将 heavy-hitter 选择、统一 INT2 量化和融合 selective FlashAttention kernel 结合，是与 PhaseKV 在算法和系统协同方面最接近的工作。MiniKV 对各相位使用统一 INT2；PhaseKV 使用相位分层位宽、独立只读 prompt 布局和 decode 热/冷分层。`abl_uniform4` 消融用于验证统一位宽与分层位宽之间的差别。

**SnapKV** 在 prefill 阶段利用观察窗口进行 per-head 选择，decode 过程中继续以全精度追加，缓存仍随生成长度增长。**H2O** 和 **StreamingLLM** 分别使用 heavy-hitter 与 attention sink 加滑动窗口处理 decode 缓存。这些方法覆盖单一阶段或使用单一机制，而 PhaseKV 同时管理 prompt 和 generated KV 的完整生命周期。

### 6.5 体系结构路线：Oaken 与分层融合 Kernel

**Oaken** 采用在线—离线混合 KV 量化，并通过定制加速器中的量化、反量化和数据搬运模块降低访存成本。它以 ASIC 为目标并采用统一量化；PhaseKV 面向通用 GPU，在软件层完成相位布局、预算规划和 Triton 融合 kernel 协同。

另一类分层融合 kernel 工作将 KV 放置在 GPU 与 CPU 的不同层级，运行时通过误差认证决定是否回退到 dense attention。PhaseKV 不采用 dense 回退：非法配置或 kernel 执行错误直接失败，数值质量由 token 选择、位宽分配和热窗口保护控制。这一路线与认证后回退的设计目标不同。

### 6.6 方法对比

| 维度 | SCOPE | FastKV | PyramidKV | KIVI | MiniKV | DuoAttention | Oaken | **PhaseKV** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 异构主轴 | phase | phase/层 | 层 | 精度×新近度 | 全局 | head | 全局 | **phase×生命周期×精度** |
| 压缩机制 | 驱逐 | 驱逐 | 驱逐 | 量化 | 选择+量化 | 驱逐 | 量化 | **选择+量化按相位协同** |
| 设计依据 | 精度 | 精度 | 精度 | 精度 | 精度+系统 | 精度 | 硬件（ASIC） | **硬件（读放大/roofline）** |
| 压缩时机感知 | 否 | 否 | 否 | 否 | 否 | 否 | 否 | **prefill 一次/decode 懒批** |
| 只读 prompt 专层 | 否 | 否 | 否 | 否 | 否 | 否 | 否 | **是** |
| 预算分配 | 手动 | 手动 | 标定 | 固定 | 固定 | 训练 | 固定 | **roofline 成本模型** |
| 融合 kernel | 无 | 无 | 无 | 有 | 有 | 有 | ASIC | **三层混合精度单遍 kernel** |
| 平台 | GPU | GPU | GPU | GPU | GPU | GPU | ASIC | **GPU** |

### 6.7 设计与实验数据对应关系

| 设计内容 | 实现与数据 | 当前结果 |
| --- | --- | --- |
| prefill 计算受限、decode 带宽受限 | `planner.py`；`results/roofline_4080s` | prefill 利用率随 P=512→4096 从 15.6% 增至 77.0%；长上下文和大 batch 下 KV 读占 decode 带宽 33.0%→96.9% |
| prompt-KV 具有更高读放大，应获得更激进预算 | `prompt_tier.py`、`planner.py`；`results/accuracy_per_byte` | 只压 prompt keep=0.1 节省 78.40%，只压 generated keep=0.1 节省 11.68%；精度结论以 7B LongBench 为准 |
| 重要度信号获取成本随相位变化 | `prompt_tier.py`、`decode_tier.py`；`results/smoke_attention` | materialize scores 的方法峰值显存高 21.6%；该项目前是显存代理证据，不作为单步延迟结论 |
| 机制、布局和位宽按相位协同 | 三层缓存与融合 kernel；7B 主结果、总计 500 条样本的显著性复核和消融 | 在统计不可区分的任务精度下达到 93.31% 逻辑 KV 节省；2-bit kernel 在长上下文达到 1.20–1.63× |

PhaseKV 与现有方法的主要区别是按 prompt、decode 热层和 decode 冷层组织缓存，并统一配置预算模型、数据布局与融合执行。

## 7. 实验结果

> 本节数字来自 `results/` 下的真实测量文件。引用结果时需同时保留数据集、样本量、显存定义和计时范围，具体见 §9。

### 7.1 主结果：Qwen2.5-7B + LongBench（`results/m1_longbench_7b.md`，n=32/任务=160 样，1536 中部截断，eager）

| 方法 | qa_f1 | KV 逻辑节省% | 压缩比× | 峰值显存MB | 对 full 一致率 |
| --- | --- | --- | --- | --- | --- |
| full | 15.62 | 0.06 | 1.00 | 16010 | — |
| kivi | 17.27 | 77.07 | 4.36 | 16010 | 0.62 |
| snapkv | 15.31 | 69.14 | 3.24 | 19716 | 0.75 |
| scope | 15.62 | 0.06 | 1.00 | 19729 | 1.00 |
| pyramidkv | 14.18 | 69.12 | 3.24 | 19715 | 0.72 |
| **phasekv** | **18.43** | **93.31** | **14.94** | 19715 | 0.62 |

在 n=32 的主实验中，PhaseKV 的逻辑 KV 节省为 93.31%（14.9×），qa_f1 为 18.43。

### 7.2 显著性复核：每任务 100 条、总计 500 条（`results/_qa_significance_7b.md`）

n=32 的结果不确定性较大，因此采用 **每任务 100 条、5 个任务总计 500 条样本**，执行 10000 次分层 bootstrap 复核：

| 方法 | qa_f1（总计 N=500） | 95%CI | vs full 均值差 | 差 95%CI | bootP | 0.05 显著 |
| --- | --- | --- | --- | --- | --- | --- |
| full | 16.185 | [13.646, 18.908] | — | — | — | — |
| **phasekv** | 16.44 | [13.895, 19.116] | **+0.256** | **[-1.544, 2.073]** | **0.776** | **否** |
| kivi | **17.179** | [14.566, 19.982] | +0.995 | [-0.688, 2.68] | 0.243 | 否 |
| snapkv | 15.733 | — | -0.451 | — | 0.382 | 否 |
| pyramidkv | 14.68 | — | -1.504 | [-3.003, -0.143] | 0.0294 | **是（负）** |
| scope | 16.185 | — | 0.0 | — | 1 | 否 |

**结果分析**：总计 500 条样本的复核中 **phasekv vs full 差 +0.256、bootP=0.776、差 95%CI 跨 0 ⇒ 统计不可区分**；均值最高者是 kivi（17.179）。因此**不声称「精度反超 full」**，而声称：

> PhaseKV 的逻辑 KV 节省为 93.3%（14.9×）。每任务 100 条、总计 500 条的分层 bootstrap 结果显示，其 qa_f1 与 full 的差异不显著。

显著性结果表明，当前数据支持“统计精度相当条件下的高压缩率与 kernel 收益”，不支持“精度反超 full”。

### 7.3 阶段 B 融合 kernel 加速：真实测量（`results/kernel_decode_bench.md`，RTX 4080S，CUDA events 中位数 of 100，单 batch/单 query 步/单层）

| 上下文 S | 2bit vs SDPA | 4bit vs SDPA | KV 读省%(2bit/4bit) | 数值误差 |
| --- | --- | --- | --- | --- |
| 4096 | 1.20× | 0.65× | 81.25% / 68.75% | ~6e-5 |
| 8192 | 1.42× | 1.09× | 81.25% / 68.75% | ~3e-5 |
| 16384 | 1.57× | 1.16× | 81.25% / 68.75% | ~3e-5 |
| 32768 | **1.63×** | 1.19× | 81.25% / 68.75% | ~1.5e-5 |

**结果分析**：2-bit 的带宽节省为 81%，高于 4-bit 的 69%，并且加速比随上下文长度增长，与 decode 带宽受限的分析一致。该结果是 kernel 级 microbenchmark，不是模型端到端 TTFT/TPOT；S=4096/4-bit 的 0.65× 来自短上下文下的切分与合并开销。

### 7.4 图 1 Roofline / 图 2 accuracy-per-byte

见 §3.1 与 §3.2 的实测数字（`results/roofline_4080s.md` / `results/accuracy_per_byte.md`），分别说明阶段执行特征和 prompt/generated KV 的读取差异。

### 7.5 结果边界：0.5B 上 PhaseKV 不占优

0.5B 结果用于限定方法的适用范围：

- **0.5B 跨方法（困惑度口径，`results/m1_pareto_0p5b.md`）**：KIVI（省 69.5% / ppl 1.76 / 一致 1.00）明显优于 PhaseKV（省 86.84% / ppl 2.34 / 一致 0.15）。PhaseKV 的压缩比为 7.6×，但精度低于 KIVI。
- **0.5B LongBench（`results/_validate_longbench_0p5b.md`）**：PhaseKV 的 qa_f1 为 13.46，PyramidKV 为 22.0。

**结果分析**：PhaseKV 在不同模型规模和上下文长度下的结果不同。在 7B LongBench（§7.1/§7.2）上，phasekv 与 pyramidkv 的排名相对 0.5B **对调且拉大**（n=32 下 phasekv 18.43 最高、pyramidkv 14.18 最低）。故**主结果以 7B 为准，0.5B 冒烟不作为精度结论依据**。因此主结果采用 7B LongBench，0.5B 冒烟结果不作为精度优势的依据。

---

## 8. 组件消融

本节比较各组件配置。消融矩阵（`results/ablation_phasekv_7b.md`，Qwen2.5-7B + LongBench，同 §7.1 口径）逐一关掉 PhaseKV 的一个分层组件，看 qa_f1 与节省如何变化：

| 变体 | 相对完整配置改了什么 | qa_f1 | KV 逻辑节省% | 对 full 一致率 |
| --- | --- | --- | --- | --- |
| **phasekv_full** | 完整：keep0.3 / pb2 / cb4 / rw32 | **18.43** | 93.31 | 0.62 |
| abl_no_select | keep1.0（**关 prompt 选择**） | 17.49 | 80.40 | 0.63 |
| abl_no_promptq | pb16（**关 prompt 量化**） | 14.45 | 69.28 | 0.76 |
| abl_uniform4 | pb4/cb4（**关分层位宽**） | 14.59 | 89.67 | 0.77 |
| abl_min_hot | rw1（**关 hot 保护**） | 18.48 | 93.66 | 0.62 |

**各配置结果**：

- **Prompt 选择**（`abl_no_select`）：保留全部 prompt token 后，qa_f1 从 18.43 变为 17.49，逻辑节省从 93.31% 变为 80.40%。
- **Prompt 位宽**（`abl_no_promptq`）：把 prompt 从 2-bit 调整为 16-bit 后，qa_f1 从 18.43 变为 14.45，逻辑节省从 93.31% 变为 69.28%。在当前 n=32 的实验设置下，2-bit prompt 配置取得了更高任务分和压缩率；该观察仍需更大样本的独立消融复核。
- **分层位宽**（`abl_uniform4`）：将 `pb2/cb4` 换成统一的 `pb4/cb4` 后，qa_f1 从 18.43 变为 14.59，逻辑节省从 93.31% 变为 89.67%。
- **hot 窗口在本任务近中性**（`abl_min_hot`）：把 rw32 降到 rw1，qa_f1（18.48 vs 18.43）与节省（93.66% vs 93.31%）都在噪声内。在 LongBench QA 的短输出条件下，热窗口大小影响不显著；其作用仍需在长输出任务中测量。

**消融结果总结**：在当前 n=32 设置下，prompt 选择、prompt 量化和相位分层位宽均改变了任务分与逻辑节省，变化方向与 §3 和 §4 的设计分析一致。热窗口消融在短输出 QA 任务上没有显示出明显差异。

---

## 9. 测量范围

引用实验结果时需要同时保留以下测量口径：

1. **所有数字来自真实模型真实测量，无模拟数据。**
2. **速度/加速比只来自阶段 B 真实 kernel microbenchmark**；阶段 A 的 TTFT/TPOT 和 planner 解析量不作为加速结果。
3. **逻辑 KV 节省 ≠ 峰值显存下降**：`93.31%` 是逻辑 KV 字节节省（按 bit 折算），阶段 A 伪量化物理张量仍 fp16，实测 phasekv 峰值 `19715MB > full 16010MB`。真实显存/带宽/能耗下降需阶段 B 打包 + kernel + NVML 测量。
4. **不声称「精度反超 full」**：每任务 100 条、总计 500 条的分层 bootstrap 显示 phasekv vs full 不显著（+0.256，bootP=0.776）；当前结论为统计精度相当条件下的逻辑 KV 压缩结果。
5. **已知结果边界**：hotpotqa 多跳仅 13.87（full 16.54）；0.5B 上 PhaseKV 不占优；一致率 0.62 属阶段 A 激进改写预期。
6. **错误处理**：不使用 dense 回退，配置或执行错误直接抛出。
7. **当前未覆盖的测量**：重要度信号的真实单步延迟、阶段 B 模型端到端 TTFT/TPOT/吞吐与能耗、Llama-3.1-8B 泛化。

---

## 10. 结论

PhaseKV 的方法与结果总结如下：

1. **问题**：KV Cache 是 decode 带宽瓶颈（实测 7B ctx=131072 时 KV 占 decode 带宽 33%~97%）；现有工作按层、head、新近度、query 或 phase 划分缓存，并主要根据精度配置预算。
2. **硬件分析**：阶段执行特征、KV 读取次数和信号获取开销均有解析或实测依据。
3. **方法**：每个设计决策（三层缓存 / prompt 只读低位宽压缩 / decode 热环+懒批冷压 / 相位分层位宽 / roofline planner / 单遍融合 kernel）均与前述硬件分析对应。
4. **实现**：`src/phasekv/` 完成逐 head 选择且保持矩形张量、真实绝对位置 RoPE、精确逻辑字节折算和 fail-fast 配置校验。
5. **实验结果**：7B LongBench 的逻辑 KV 节省为 93.3%（14.9×）；每任务 100 条、总计 500 条的分层 bootstrap 显示其任务精度与 full 差异不显著；阶段 B 2-bit kernel 的加速比为 1.20–1.63×。
6. **边界**：逻辑节省不等同于峰值显存下降；现有统计结果不支持精度反超；0.5B 不占优；端到端加速和能耗尚未测量。

PhaseKV 从访存流量的硬件视角出发，将 KV Cache 压缩的机制、布局和执行时机按相位协同设计。当前实测结果验证了逻辑压缩率、任务精度、kernel 正确性和 kernel 级加速；未完成的端到端与泛化测量保持为明确的结果边界。
