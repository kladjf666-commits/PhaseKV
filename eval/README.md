# eval/ —— KV Cache 压缩评估框架 (CH-3 / 质量验证)

方法无关的 KV Cache 压缩评估框架: 任何压缩方法实现 `KVCompressor` 接口并经 `register`
注入, 即可被统一装载、对比、验收。输出内存 / 速度 / 精度三类指标与 Markdown+CSV+JSON 报告。

## 目录
- `interface.py`  —— `KVCompressor` 接口契约 + `CompressionConfig` (集成规范, 团队按此实现)。
- `metrics.py`    —— 内存 / 延迟 / 困惑度 / 一致率 等纯函数指标。
- `compressors/`  —— 参考压缩器 (full/recent/streaming/kvquant/h2o/hetero) + 注册表工厂。
- `runtime.py`    —— 模型加载 + 带压缩的生成循环 + 流式困惑度 (真实计时/显存)。
- `datasets.py`   —— WikiText / 合成 Needle / LongBench + 内置离线语料。
- `report.py`     —— 报告 (Markdown/CSV/JSON)。
- `runner.py`     —— 编排 (一键基准)。
- `run_benchmark.py` —— CLI。

## 接入你的方法 (CH-2 / CH-5)
```python
from eval.interface import KVCompressor, CompressionConfig
from eval.compressors import register

class MyCompressor(KVCompressor):
    requires_attention = False        # 若 prefill/decode 需要注意力分数则 True (走 eager)
    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        ...   # 压缩 prompt KV, 返回 cache
    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        ...   # 维护/压缩增长的 KV, 返回 cache
    def cache_memory_bytes(self, cache):
        ...   # 量化类需覆盖, 按 bit 折算逻辑显存

register("mymethod", MyCompressor)
```
然后: `python -m eval.run_benchmark --model <m> --methods-json '[{"method":"mymethod",...}]'`

## 关键约定 (务必遵守)
- 驱逐后位置正确性由 runtime 保证 (每步传真实 position_ids + 物理 cache_position)。
  压缩器只需保证保留的 K/V 张量自身一致, 切片按时间升序。
- 需要 transformers **新式 RoPE** (>=4.46, position_ids 与 cache_position 解耦); 旧式
  (4.44.2 Qwen2) 与 KV 驱逐不兼容, 详见 docs/qa/RISK_LIST.md R-01。
- 显存口径: `kv_saving_pct` 为逻辑显存 (量化按 bit 折算); `peak_gpu_mb` 为实测峰值。
  伪量化只降逻辑显存, 实测下降需真实打包/kernel。

## 快速开始
见 `../scripts/smoke_eval.sh` 或 `docs/qa/EVAL_PLAN.md` 第 8 节。
