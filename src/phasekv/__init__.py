"""PhaseKV: 访问放大感知的相位异构 KV Cache 压缩 (CH-2 架构)。

导入本包即把 ``phasekv`` 注入 eval 注册表 (``eval.compressors.register``), 之后即可经
``build_compressor("phasekv", ...)`` 或 ``--methods-json '[{"method":"phasekv",...}]'``
被评估框架装载、对比与验收。

模块构成 (idea 文档第 3/5 节):
  - planner.py       roofline 预算分配器 (3.2)
  - quant.py         KIVI 式 per-channel K / per-token V 伪量化 + 逻辑字节折算
  - hetero_cache.py  三层异构缓存状态 + 底层算子 (3.1)
  - prompt_tier.py   prefill 注意力选择 + 低比特打包 (3.1-1)
  - decode_tier.py   decode FP16 热环 + 懒批冷压 (3.1-2/3)
  - compressor.py    PhaseKVCompressor(KVCompressor) 接 CH-3 接口 (阶段 A)

阶段 B (融合混精注意力 kernel) 落在 kernels/ (未实现)。本包仅提供阶段 A: 真实精度 +
真实逻辑/峰值显存, 不报告 decode 加速比 (遵守"不要模拟数据"红线)。
"""
from __future__ import annotations

from eval.compressors import register

from .compressor import PhaseKVCompressor
from .phasekv_cache import PhaseKVCache
from .qwen2_attention import install_qwen2_phasekv_attention

register("phasekv", PhaseKVCompressor)

__all__ = ["PhaseKVCompressor", "PhaseKVCache", "install_qwen2_phasekv_attention"]
