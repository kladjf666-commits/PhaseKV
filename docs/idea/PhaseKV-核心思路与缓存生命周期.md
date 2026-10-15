# PhaseKV 核心思路与缓存生命周期

PhaseKV 根据 KV 的来源、后续读取次数和更新时机，分别管理 Prompt KV、生成热 KV 和生成冷 KV：Prompt 在处理完成后一次性选择、量化并冻结；生成 token 先保留 FP16，变旧且积累到一批后再选择、量化。

本文对应当前真实打包实现 `PhaseKVCache` 和 Qwen2 对话入口，重点解释“何时处理、选择什么、压缩什么、存在哪里”。多轮对话按修正后的语义执行：**新增 prompt 块结束时，保留原有生成热窗口，不强制转冷。**

## 1. 先区分 token、KV 和模型层

模型接收或生成的是 token ID；缓存保存的是每个模型层为这些 token 计算出的 Key 和 Value 向量。PhaseKV 压缩的是这些 K/V 向量，不是文本、token ID 或模型权重。

对某个模型层，未压缩 K/V 的常见形状为：

```text
[B, Hkv, S, D]
 B：batch 大小
 Hkv：KV head 数
 S：缓存中的 token 位置数
 D：每个 head 的向量维度
```

“一个 token 被选中”具体指：在某个模型层、某个 KV head 中保留它对应的 K 和 V。不同层、不同 head 可以保留不同位置，不能简单理解为对整段文本做一次统一删词。

## 2. 为什么要区分 Prompt 与生成 token

Prompt 在首次 prefill 后已经完整可用，并会被后续每次 decode 读取。生成 KV 则随着解码逐步出现，越晚产生，剩余被读取的次数通常越少。

在一个固定长度的生成过程里，可以粗略理解为：Prompt KV 被后续 decode 反复读取约 T 次，生成 KV 的平均读取次数约为 T/2。这是规划预算的近似分析，不是所有多轮场景的精确公式。

因此两者适合不同的处理时机：

| KV 来源 | 可用信息与访问特点 | 处理方式 |
| --- | --- | --- |
| 首轮 Prompt | 整段已完成，可利用 prefill 注意力；后续反复读取 | 一次性选择、量化、冻结 |
| 最新生成 token | 刚产生，近期内容需要保护 | 先保持 FP16 |
| 较旧生成 token | 已滑出保护窗口，数量逐步累积 | 达到批次阈值后选择、量化、冻结 |

设计关注的不只是缓存容量，还包括每步读取、选择和量化的成本。较长 prompt 的 prefill 有更多并行计算，而单 token decode 更容易受到数据读取影响；实际瓶颈仍随模型、batch、上下文和 kernel 实现变化。

Prompt 值得更积极压缩，是读取成本上的动机，**不代表 Prompt 在语义上更不重要**。`prompt_bits < cold_bits` 也不是强制条件；当前日常对话配置使用 4/4 bit。

## 3. Prompt 和生成 KV 分开存，但只有一个 PhaseKVCache

一条当前对话复用一个 `PhaseKVCache`，其中每个 Transformer 层有一个 `LayerTiers`。不会为 Prompt 和生成部分分别创建两个 PhaseKVCache。

```text
一个 PhaseKVCache
│
├─ layers[0]：第 0 个模型层的 LayerTiers
│  ├─ prompt              首轮 Prompt 的压缩 KV 段
│  ├─ prompt_extensions   后续输入块的压缩 KV 段列表
│  ├─ cold                已冷却生成 KV 的压缩段列表
│  ├─ hot_k / hot_v       可写的 FP16 缓冲区
│  └─ hot_len             缓冲区内实际有效的 token 数
│
├─ layers[1]：第 1 个模型层的 LayerTiers
│  └─ 同样的字段
│
└─ ……其余模型层
```

这里有两个独立的划分维度：`layers[i]` 按模型层划分；各层内部再按 KV 生命周期划分。

`prompt_extensions` 与 `cold` 都是“压缩段列表”，不是“模型层列表”。增加一个段不会增加模型层，也不会创建一个新 PhaseKVCache。

Hot 正常保存尚未冷却的生成 KV；处理后续 prompt 时，也会临时借用这块缓冲区。两类 KV 的边界由 `_prompt_chunk_start` 记录，处理结束后再次分开存放。

## 4. 首轮 Prefill 做什么

### 4.1 先正常计算完整 Prompt

首轮输入使用原始 Qwen2 attention 和 `DynamicCache`。每层完成投影、RoPE 与因果注意力，得到完整 Prompt K/V。

当前方案在 prefill 完成之后压缩缓存，不是在首次 prefill 计算前就删除输入 token。模型 BF16 前向得到的 K/V 会在进入当前真实打包路径时转换为 FP16。

### 4.2 按预算挑选 Prompt token

若 `prefill_keep_ratio < 1.0`，调用 `select_prompt_indices()`；若等于 1.0，当前对话入口跳过选择，保留全部 Prompt。

选择逻辑分为以下步骤：

1. **观察窗口投票。** 取 Prompt 末尾 `obs_window` 个 query，对每个 key 位置的注意力权重求和。得到“末尾这些 query 总共关注该位置多少”的分数。
2. **GQA 聚合。** 若多个 query head 共享一个 KV head，将对应 query head 的分数相加，得到每个 KV head 的位置分数。
3. **保留边界 token。** 在总预算内优先保留开头的 sink，再保留尽可能多的末尾观察窗口 token。
4. **对中间候选位置做 pooling。** 当前入口使用窗口大小 7 的 max pooling，让高分位置附近的局部上下文也有机会得到较高分数。它不是硬性选取一个连续片段。
5. **逐 head 取 top-k。** 为中间候选分配剩余预算，选择分数最高的位置。
6. **合并并按原位置升序排序。** 输出每层的 `[B,Hkv,K0]` 下标。

预算可写为：

```python
sink = min(sink_tokens, prompt_len)
budget = min(prompt_len, max(1, sink, round(prompt_len * prefill_keep_ratio)))
sink_keep = min(sink, budget)
recent_keep = min(obs_window, budget - sink_keep, prompt_len - sink_keep)
select_k = budget - sink_keep - recent_keep
```

例如 Prompt 长 1000，保留比例 0.5，sink 为 4，观察窗口为 32：

```text
总保留 500 个位置
= 最前 4 个
+ 中间候选里选出的 464 个
+ 最后 32 个
```

短 Prompt 或预算很小时，不保证完整保留整个观察窗口，不能把它额外叠加到总预算之外。

不同 head 的位置可以不同，但数量必须相同，以维持矩形张量。当前 `PhaseKVCache.from_dense_prefill()` 还要求所有模型层保留相同数量的首轮 Prompt token。

注意力分数是重要度代理，不是未来所有问题的语义保证。取得显式注意力矩阵也并非总是免费：当前 eager 路径可返回分数，FlashAttention/SDPA 路径显式收集它们会涉及额外开销。

### 4.3 选择后量化，建立冻结段

`from_dense_prefill()` 接收选择好的下标，使用 gather 取出对应 K/V，再按 `prompt_bits` 真实量化打包：

```text
完整 Prompt KV
→ 按各层各 head 下标选择
→ 对保留 K/V 量化、打包
→ 保存为 layer.prompt
```

该函数不负责计算注意力重要度，也不修改传入的原始 dense cache。构建完成后，调用方需释放不再需要的 dense cache 和 prefill 输出引用，才能释放原稠密缓存占用。

“冻结”表示后续正常解码不再选择、改写或重复量化这个段，并不表示模型不再读取它。

## 5. 压缩具体分成哪两件事

### 5.1 Token 选择与驱逐

选择减少缓存保留的位置数。例如 1000 个位置保留 500 个，另外 500 个位置的 K/V 不再进入后续 attention。

被驱逐的 KV 不能靠反量化找回来。原始 token ID 或聊天文本可以仍在程序历史中，但当前 attention 不会自动从文本恢复已经丢弃的 KV。

### 5.2 量化与字节打包

量化降低每个保留数值的精度；打包把这些低比特码紧密存入字节。两者可以在完全不删除 token 的情况下使用。

对于一组数值，量化过程可近似表示为：

```text
qmax = 2^bits - 1
scale = max(group_max - group_min, epsilon) / qmax
code = clamp(round((x - group_min) / scale), 0, qmax)
重建值 ≈ code × scale + group_min
```

实际实现还会对分组尾部作处理，并将 scale/min 保存为 FP16；重建值通常不等于原始值。

当前 K/V 的分组方向不同：

| 对象 | 原始形状 | 分组轴 | 含义 |
| --- | --- | --- | --- |
| K | `[B,Hkv,S,D]` | token 轴 S，即 `axis=-2` | 固定一个 channel，跨一组 token 共享 scale/min |
| V | `[B,Hkv,S,D]` | head_dim 轴 D，即 `axis=-1` | 固定一个 token，跨一组 channel 共享 scale/min |

`group_size=32` 表示沿各自量化轴分组，并不是“每 32 个生成 token 冷却一次”。冷却频率由 `lazy_g` 决定。

每段 K 或 V 最终表示为 `PackedTensor`：

```text
packed       uint8 数据，存放紧密打包后的量化码
scale / mins 每个量化组的参数
bits         位宽
orig_shape   打包前的形状
axis         量化轴
group_size   该段的实际分组大小
```

2-bit 每字节容纳 4 个码，4-bit 每字节容纳 2 个码，8-bit 每字节容纳 1 个码。真实路径支持 2、4、8 bit。

4-bit 相对 FP16 的数据位数降至四分之一，但完整存储还包括量化元数据、字节补齐和 FP16 热区，所以整体缓存压缩比不能直接写成 4 倍。

## 6. Decode 做什么

### 6.1 先写入当前 token，再计算注意力

首轮 prefill 的末尾 logits 可以采样第一个生成 token。采样只得到 ID，此时该生成 token 的 K/V 还没有进入缓存。

将这个 token 再送入模型时，每个 attention 层执行：

```text
计算 Q/K/V
→ 对 Q/K 使用真实绝对位置的 RoPE
→ 转为当前 kernel 使用的 FP16
→ append_hot() 写入当前 token 的 K/V
→ fused_phasekv_decode() 读取各存储段计算注意力
→ 输出投影
```

写入之后再计算注意力，使当前 query 可以看到自己以及已保留的历史 K/V。

当前热区是预分配的连续缓冲区，不是环形队列。每步追加写到 `hot_len` 所指的位置，只有前 `hot_len` 个位置有效，不能把整个 capacity 当作已写入历史。

### 6.2 整个模型前向结束后，统一维护缓存

所有模型层返回后，外层调用一次 `finish_decode_step()`：

```text
检查每层都已写入当前 token
→ 每层按条件冷却
→ 真实长度加 1
→ 清空本步已写入层记录
```

不能在每个 attention 层结束时各调用一次，否则后面的模型层可能还没有写入。

### 6.3 什么时候冷却

```python
aged = hot_len - recent_window
if aged < lazy_g:
    return
```

`recent_window` 是受保护的最近生成 token 数；`aged` 是已经滑出保护窗口、仍临时保存在 FP16 热区的数量。达到 `aged >= lazy_g` 时就冷却，等号也会触发。

因此 Hot 可以大于 `recent_window`。尚未攒够一批的旧 token 仍留在里面，不代表窗口失效。

例如 `recent_window=32, lazy_g=16`：

```text
hot_len=47：暂不冷却
hot_len=48：先完成当前注意力，再处理最旧 16 个，留下最近 32 个
```

当前窗口按生成 token 流计数。新增 prompt 不推进这个生成窗口，轮次结束也不强制冷却。

### 6.4 如何挑选较旧生成 token

若 `decode_keep_ratio < 1.0`，当前实现仅在本次 aged 候选批中计算各 token 的 Value L2 范数：

```python
importance = torch.linalg.vector_norm(value.float(), dim=-1)
keep_count = max(1, round(decode_keep_ratio * aged))
```

每个 KV head 取分数最高的 `keep_count` 个位置，按原先顺序排序，再同时选出对应 K/V。它不重新筛选旧 Cold，也不触碰 Prompt。

使用 Value 范数是为了避免每个 decode 步显式输出完整注意力分数。它是便宜的代理信号，不能保证保留所有未来重要的代码、编号或事实。

若 `decode_keep_ratio=1.0`，直接保留整批，不需要计算范数和 top-k，但仍按 `cold_bits` 量化压缩。

### 6.5 冷却与驱逐后的去向

例如 `hot_len=48, aged=16, decode_keep_ratio=0.5`：

```text
原热区：[最旧 16 个][最近 32 个]
             │          │
             ├─ 8 个 → 量化打包，追加到 cold
             └─ 8 个 → 驱逐
                        └─ 32 个 → 继续留在 FP16 热区
```

因此：

```python
remaining = hot_len - aged  # 32，而不是 48 - 8 = 40
```

全部 aged token 都离开 Hot，其中一部分转冷，另一部分被驱逐。随后把最近 `remaining` 个 token 搬到缓冲区起点，并更新 `hot_len`。复制时使用 clone，避免源和目标内存区间重叠。

每次冷却追加一个 `PackedKVSegment` 到 `layer.cold`。已经成型的冷段不再次量化，也不因为后续冷却而重新选择。

## 7. 多轮对话与 prompt_extensions

假设第一轮输入为 P1，模型生成 G1，第二轮新增输入为 P2。处理 P2 前：

```text
prompt：P1 的压缩 KV
cold：G1 中已经冷却的部分
hot：G1 中剩余的部分
```

`begin_prompt_chunk(len(P2))` 记录各层原热长度 `start`，必要时扩容。逐 token 写入 P2 时：

```text
hot：[G1 剩余部分 | P2 已写入部分]
                 ↑ start
```

每写入 P2 的一个 token，立即计算它的 query 注意力。当前 kernel 只支持单 query，不能先写完整块再让每个 query 读取全部 Hot，否则块内早期 query 会看见未来 token。

模型完成 P2 的整块前向后，`finish_prompt_chunk()` 执行：

```text
[0:start]        → 原样复制到正常容量的 FP16 热区
[start:hot_len]  → 按 prompt_bits 打包到 prompt_extensions
hot_len         → 恢复为 start
cold            → 不变
```

结果是：

```text
prompt：P1
prompt_extensions：P2
cold：G1 原有冷段
hot：G1 原有热部分，等待后续生成 token 继续追加
```

这里不仅保留最近 `recent_window` 个生成 token，也保留已经滑出窗口但尚未攒够 `lazy_g` 的部分。不能在轮次边界悄悄重置冷却进度。

当前新增 prompt 块全部保留后量化，没有执行首轮的注意力筛选。它还可以包含聊天模板 token，例如当前对话入口通过该接口追加 assistant 结束符。因此，字段 `prompt_extensions` 更精确的含义是“经新增输入块接口写入并冻结的段”，并不只包含用户正文。

这些段按用途保存，不要求 `prompt + extensions + cold + hot` 的列表拼接顺序等于多轮时间顺序。当前每次单 query 读取的都是可见历史，K/V 成对保存，K 已包含原位置的 RoPE；注意力按各段的 softmax 统计量合并，不依赖列表顺序重新赋予 token 位置。

## 8. 模型如何同时读取三种表示

当前 `fused_phasekv_decode()` 读取：

```text
首轮 prompt packed 段
+ 后续 prompt_extensions packed 段
+ 所有 cold packed 段
+ hot_views() 返回的有效 FP16 热区
```

对 packed 段，kernel 读取量化码与元数据，在计算过程中反量化，不先把整份历史恢复为一个长期驻留的 FP16 缓存。

各段产生局部 softmax 统计量，再按统一归一化关系合并。不是对每段分别归一化后简单平均输出，否则不同段之间的注意力权重会错误。

当前实现仍会逐段启动计算，再合并结果。`fused_phasekv_decode` 这个名称不意味着所有段只需一次 kernel launch。冷段和扩展段过多时，调度与合并开销会影响速度；更小的 `lazy_g` 可能形成更多小冷段。

## 9. 真实位置与物理长度必须分开

假设原 Prompt 有 1000 个 token，筛选后保留 500 个：

```text
true_seq_length = 1000
get_seq_length() = 500
```

下一个输入 token 的位置应为：

```python
position_ids = [[1000]]   # 真实绝对位置，决定 RoPE
cache_position = [500]   # 当前集成中的物理位置
```

保留下来的旧 K 已经使用原位置施加 RoPE，不能在压缩后按 0、1、2……重新编码。新 token 也不能使用压缩后的物理长度作为真实位置。

`true_seq_length` 在 `finish_decode_step()` 或 `finish_prompt_chunk()` 后推进。逐层写入期间各层的物理长度可能暂时不同，因此在模型前向开始前构造位置。

压缩减少物理缓存，不代表扩大模型支持的真实上下文长度。多轮累计位置仍受模型上下文限制约束。

## 10. 完整调用链与关键状态

```text
加载模型、eval、安装 install_qwen2_phasekv_attention()
│
├─ 首轮输入
│  ├─ model(..., DynamicCache) → 原始 attention 完成 prefill
│  └─ build_phasekv_cache()
│     ├─ select_prompt_indices()：需要筛选时调用
│     └─ from_dense_prefill()：选择、打包、创建一个 PhaseKVCache
│
├─ 生成回答
│  ├─ 从 logits 采样 token ID
│  └─ decode_token()
│     ├─ model(..., PhaseKVCache)
│     │  └─ 每层：投影/RoPE → append_hot() → fused_phasekv_decode()
│     └─ finish_decode_step()：全模型一次，按需冷却
│
└─ 后续输入或模板结束符
   └─ append_prompt_chunk()
      ├─ begin_prompt_chunk()
      ├─ model() → 每层逐 token：append_prompt_token() → attention
      └─ finish_prompt_chunk()：冻结新增输入，保留生成热区
```

| 状态或接口 | 含义 |
| --- | --- |
| `_pending_layers` | 本 decode 步已经写入当前 token 的模型层编号集合；防止重复写入或提前提交 |
| `_prompt_chunk_len` | 本次新增输入块的目标 token 数，处理期间不递减；0 表示没有进行中的块 |
| `_prompt_chunk_start[i]` | 第 i 层开始新增输入前的热长度，用于区分生成 KV 与临时输入 KV |
| `hot_len` | 当前有效热区长度，通常包括近期生成 token 和未攒够批次的旧生成 token |
| `hot_k.shape[2]` | 热区分配容量，不是有效长度 |
| `reorder_cache()` | 按 batch 下标选择、复制或重排候选缓存；当前单条对话不需要调用 |

`_pending_layers` 只检查“是否写过 KV”，不能证明注意力已经执行；外层必须保证模型前向返回之后再提交。本实现也没有对部分前向失败提供事务回滚。

## 11. 参数与当前应用配置

下表来自撰写时的 `qwen2.py` 活跃配置。它已采用前面讨论的长回答起始配置；不是已测得的最优配置。

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `MAX_NEW_TOKENS` | 2048 | 每轮输出上限；EOS 可提前结束，不等于历史容量 |
| `PROMPT_BITS` | 4 | 首轮及扩展 Prompt 的量化位宽 |
| `COLD_BITS` | 4 | 生成冷层的量化位宽 |
| `PREFILL_KEEP_RATIO` | 1.0 | 首轮 Prompt 全保留，跳过注意力选择 |
| `DECODE_KEEP_RATIO` | 1.0 | 冷候选批全保留，但仍量化 |
| `SINK_TOKENS` | 4 | 启用首轮选择时优先保护的开头位置数 |
| `OBS_WINDOW` | 32 | 启用首轮选择时用于投票的末尾 query 窗口 |
| `RECENT_WINDOW` | 128 | 最近生成 token 的 FP16 保护窗口 |
| `LAZY_G` | 128 | 过期生成 token 积累到此数量后冷却 |
| `GROUP_SIZE` | 32 | 沿 K/V 各自量化轴的分组大小 |

该配置的主要取舍是先减少驱逐影响，再评估量化的质量与速度。由于首轮保留比例为 1.0，当前对话入口不执行 `SINK_TOKENS`、`OBS_WINDOW` 对应的选择逻辑。

从空生成热区开始，写入第 256 个生成 token 并完成其注意力后，才首次冷却前 128 个，留下最近 128 个。Prompt 块不计入这个生成数量。

对于 Qwen2-1.5B 的 28 层、2 个 KV head、head_dim=128，容量 256 的 FP16 热区约占 7 MiB；这是热区张量的计算量，不含模型权重、packed 段或工作区。

仓库中的 [2026 年 10 月 14 日对话报告](/home/kladjf/桌面/kvcomp-astra/results/qwen2_chat_validation_20261014.md)使用的是 `recent_window=32, lazy_g=16, decode_keep_ratio=0.5`。它验证了修正后的缓存生命周期，但不能当作当前 128/128、全保留配置已经完成长上下文质量或性能验证的证据。

## 12. 核心设计与当前实现的边界

| 项目 | 当前行为与边界 |
| --- | --- |
| 首轮 Prompt 选择 | 支持注意力投票、pooling、逐 head 选择；当前入口配置跳过筛选 |
| 后续 Prompt 选择 | 暂未实现，新增块全部保留后打包 |
| 生成 token 选择 | 支持每个过期批内的 Value 范数 top-k；当前配置全保留 |
| 热层更新 | 连续缓冲区追加，冷却时搬移尾部；不是环形队列 |
| 多轮边界 | 保留已有生成热区，不额外冷却 |
| 冻结段管理 | 持续追加；没有周期性重压缩、段合并或全局固定容量限制 |
| 注意力执行 | 单 query、逐段 packed 计算并合并；新增块内部仍逐 token 执行 |
| Qwen2 适配范围 | 当前为 CUDA、batch=1、无 padding 的显式生成循环 |
| 预算规划 | planner 可依据逻辑显存与读取成本输出超参数；当前对话入口直接使用配置常量 |
| 质量与速度 | 代理信号、较低位宽和较少字节不自动保证语义质量或端到端加速 |

仓库还保留另一条 fake-quant 评估路径：`PhaseKVCompressor` 配合 `prompt_tier.select_and_pack()`、`decode_tier.maintain()`，量化后仍把张量保存为浮点数。它与本文的真实 `PackedTensor` 存储路径不同，不能把其逻辑字节节省直接当作真实物理显存下降。

## 13. 对应代码与进一步阅读

- [phasekv_cache.py](/home/kladjf/桌面/kvcomp-astra/src/phasekv/phasekv_cache.py)：层级存储、真实打包入口、热区追加、懒批冷却、多轮输入块。
- [prompt_tier.py](/home/kladjf/桌面/kvcomp-astra/src/phasekv/prompt_tier.py)：首轮 Prompt 保留下标选择。
- [hetero_cache.py](/home/kladjf/桌面/kvcomp-astra/src/phasekv/hetero_cache.py)：观察窗口投票、GQA 聚合与 pooling。
- [packing.py](/home/kladjf/桌面/kvcomp-astra/src/phasekv/kernels/packing.py)：K/V 分组量化与真实低比特打包。
- [attention_triton.py](/home/kladjf/桌面/kvcomp-astra/src/phasekv/kernels/attention_triton.py)：读取 packed KV、计算和合并注意力。
- [qwen2_attention.py](/home/kladjf/桌面/kvcomp-astra/src/phasekv/qwen2_attention.py)：模型层内的阶段判断与因果写入顺序。
- [qwen2.py](/home/kladjf/桌面/kvcomp-astra/qwen2.py)：对话入口、采样、真实位置管理与跨轮缓存复用。
- [planner.py](/home/kladjf/桌面/kvcomp-astra/src/phasekv/planner.py)：预算与读取成本分析。
- [使用指南](/home/kladjf/桌面/kvcomp-astra/docs/delivery/USAGE.md)：调用示例与启动命令。
- [技术原理与系统设计](/home/kladjf/桌面/kvcomp-astra/docs/idea/PhaseKV-技术原理与系统设计.md)：更完整的设计依据、研究对比与实验结果。
