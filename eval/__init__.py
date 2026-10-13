"""KV Cache 压缩评估框架 (eval/) —— QA / CH-3 维护。

对外稳定入口:
  - interface.KVCompressor / CompressionConfig : 压缩方法集成契约。
  - compressors.build_compressor / register     : 方法注册与构造。
  - runner.run_benchmark / BenchmarkConfig       : 一键基准。
  - runtime.generate_with_compression / streaming_perplexity : 底层运行时。
  - metrics / report / datasets                  : 指标、报告、数据。
"""
from __future__ import annotations

import os as _os

# 项目固定约定: 模型统一缓存到 /root/autodl-tmp/hf-cache, 下载走 hf-mirror。
# 必须在 import runtime(→transformers/huggingface_hub) 之前 setdefault, 否则
# huggingface_hub 在 import 时即固化缓存路径, 之后再设无效 (会导致端到端用例
# 因模型在 ~/.cache 找不到而静默 skip)。setdefault 保证显式环境变量仍然优先。
_os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from . import datasets, metrics, report, runtime
from .interface import CompressionConfig, KVCompressor

__all__ = [
    "CompressionConfig", "KVCompressor",
    "metrics", "report", "datasets", "runtime",
]
