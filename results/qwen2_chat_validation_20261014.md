# Qwen2 PhaseKV 对话运行报告

运行日期：2026 年 10 月 14 日（Asia/Shanghai）。本报告整理自本次修复后的真实 GPU 执行记录；四轮对话原始数据见 [JSON](qwen2_chat_validation_20261014.json)。本次整理没有重新运行模型。

## 1. 运行结论

本地 Qwen2-1.5B-Instruct 已通过首轮输入、多轮缓存复用、长回答冷却和截断后续聊验证。22 项相关回归测试通过，其中包括真实 CUDA packed attention 的因果性检查。命令行启动、`/clear` 和 `/exit` 均通过实测。

CUDA 驱动和 PyTorch 原本可用。此前默认执行沙箱没有暴露 NVIDIA 设备，导致 `nvidia-smi` 和 CUDA 初始化失败；在允许访问 GPU 的本机执行环境中，二者均正常。本次未重装驱动或 PyTorch，也未修改全局隔离配置。

## 2. 环境与配置

| 项目 | 实际值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU，8 GiB |
| NVIDIA 驱动 | 535.288.01 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA runtime | 12.1 |
| Triton | 3.1.0 |
| Transformers | 4.46.3 |
| Python | 项目 `.venv`，Python 3.12.3 |
| 模型 | Qwen/Qwen2-1.5B-Instruct，本地权重 |
| 模型精度与 prefill | BF16，eager attention |
| 压缩域 attention | FP16 Q/K/V 输入，真实 packed KV，Triton kernel |
| Batch | 1，无 padding |
| Prompt / Cold 位宽 | 4 / 4 bit |
| Prompt / Decode 保留比例 | 1.0 / 0.5 |
| recent_window / lazy_g | 32 / 16 |
| group_size | 32 |
| 四轮验证随机种子 | 42 |
| 四轮生成上限 | 32、32、64、16 token |
| 采样配置 | temperature=0.7，top_k=20，top_p=0.8，repetition_penalty=1.1 |

本次首轮 prompt 全保留，因此真实模型对话验证不覆盖首轮 token 筛选。回归测试另行覆盖 prompt 下标选择及压缩后真实位置与物理位置分离的路径。

## 3. 四轮真实对话及缓存状态

下表的真实长度、物理长度、Cold 和 Hot 均按 token 计。后三者记录第 0 个模型层；KV 字节数统计所有模型层，并包含热区预留空间，不包含模型权重、临时工作区或 CUDA allocator 的额外保留显存。

| 轮次 | 验证内容 | 真实长度 | 物理长度 | Cold | Hot | 常驻 KV 字节 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 记住代号 | 41 | 41 | 0 | 3 | 1,768,704 |
| 2 | 跨轮次回忆 | 69 | 69 | 0 | 11 | 1,987,328 |
| 3 | 长回答与生成上限 | 171 | 155 | 16 | 43 | 2,566,144 |
| 4 | 冷却后继续对话 | 195 | 179 | 16 | 45 | 2,800,896 |

第三轮达到设置的 64-token 上限，输出停在“第二段”，这是人为设置的截断。此时 Cold 保留了 16 个生成 token，真实长度与物理长度相差 16，符合两批各 16 个旧生成 token、每批保留一半的配置。Hot 为 43，包含最近 32 个生成 token 及尚未攒够一批的 11 个旧生成 token，因此大于 32 是预期行为。

第四轮追加新 prompt 后继续生成“42”，Hot 从 43 增至 45，Cold 仍为 16；新增 prompt 及 assistant 模板结束符没有清空生成热区。

### 第 1 轮

输入：请记住我的代号是蓝鲸七号。只回复“记住了”。

实际输出：

```text
记住了。
```

### 第 2 轮

输入：我的代号是什么？只回答代号。

实际输出：

```text
代号是蓝鲸七号。
```

### 第 3 轮

输入：请详细解释大语言模型的工作原理，分三段分别说明训练、推理和局限，每段至少100字。

实际输出：

```text
第一段：大语言模型的训练主要是通过大量的文本数据来学习语言规则和模式。这些数据可以来自互联网上的各种文章、书籍、新闻等，也可以从人工标注的数据集中获得。在训练过程中，模型会根据输入的文本进行分析，学习到如何理解和生成文本。

第二段
```

### 第 4 轮

输入：17加25等于多少？只回答数字。

实际输出：

```text
42
```

## 4. 测试与断言

### 回归测试

执行命令（在可访问 GPU 的环境中运行）：

```bash
.venv/bin/python -m unittest \
  tests.test_phasekv_cache \
  tests.test_phasekv_prompt_indices \
  tests.test_qwen2_attention_adapter \
  tests.test_qwen2_chat \
  tests.test_qwen2_sampling \
  tests.test_bench_qwen2_phasekv -v
```

实际结果为 `Ran 22 tests in 1.135s`、`OK`。该时间是回归套件耗时，不是模型对话速度。

覆盖内容包括：prompt 打包与量化轴、BF16 prefill 转换、冷却阈值与逐 head 选择、旧段不重复量化、跨 prompt 块保持热窗口、状态错误、batch 重排、采样及聊天历史管理。新增 GPU 测试将 prompt 块前向的 logits 与逐 token 前向进行比较（atol=rtol=0.002），并检查块结束后的旧热 KV 与块开始前逐值相等；实际运行没有跳过该 GPU 测试。

### 真实模型检查

四轮验证逐轮检查：

- 复用同一个 PhaseKVCache，首轮 prompt 段对象不变。
- `true_seq_length == len(session.token_ids)`。
- 每轮结束后没有未完成的 decode 步或 prompt 块。
- 所有层的热长度小于 `recent_window + lazy_g`。
- 第二轮回答包含“蓝鲸七号”，第四轮包含“42”。

另外验证直接 EOS 分支：在真实模型前向基础上，测试人为将首个采样结果设为 EOS，确认空回答不会产生生成热 token，模板结束块仍正确提交。此项是受控边界测试，不是模型自然选择 EOS 的质量评估。

### 命令行检查

新入口的 `--check-cuda` 成功执行 CUDA 张量计算并报告 GPU。随后通过 `--local-files-only --max-new-tokens 16` 启动对话，问题“17加25等于多少？只回答数字。”得到“42”，`/clear` 输出“对话历史已清空”，`/exit` 正常退出。

最初用于触发冷却的问题是列出 100 个整数，但模型用省略号缩短回答，未达到冷却阈值，覆盖检查因此失败。随后改成长文问题，成功覆盖冷却及上限截断；这一中间失败属于测试输入未触发目标分支，不是 CUDA 或缓存执行错误。

## 5. 结果范围

这是功能及缓存生命周期验证，不是速度或质量基准。未测量 TTFT、TPOT、吞吐、峰值显存，也未与无压缩模型做同输入对照；不能据此宣称加速比例或总体精度相当。四轮问答通过只说明这些输入的运行及输出检查通过。

## 6. 启动方式

在项目根目录的本机终端运行：

```bash
.venv/bin/python qwen2.py --check-cuda
.venv/bin/python qwen2.py --local-files-only
```

输入 `/clear` 清空历史与缓存，输入 `/exit` 退出。实现和参数说明见 [使用指南](../docs/delivery/USAGE.md)。
