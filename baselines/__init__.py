"""后端竞品基线注册入口 (CH-5)。

导入本包即把竞品基线注入 eval 注册表 (eval/compressors 的 ``register``), 供
``python -m eval.run_benchmark --methods-json '[{"method":"snapkv",...}]'`` 调用。

eval/ 由 CH-3 维护、不改动; 让 eval 感知这些基线的方式是运行前先 ``import baselines``
(见 scripts/run_eval.py)。

已接入:
  - snapkv    : SnapKV (NeurIPS'24) prefill 观察窗口投票 + per-head 聚类 pooling
  - kivi      : KIVI (ICML'24) K per-channel / V per-token 量化 + 最近 FP16 残差
  - scope     : SCOPE (ACL'25) prefill 保留 + decode 滑动窗口 + heavy-hitter
  - pyramidkv : PyramidKV (2024) 层间金字塔预算 + 层内 SnapKV 选择 (逐层异长)
"""
from eval.compressors import register

from .kivi import KIVICompressor
from .pyramidkv import PyramidKVCompressor
from .scope import SCOPECompressor
from .snapkv import SnapKVCompressor

register("snapkv", SnapKVCompressor)
register("kivi", KIVICompressor)
register("scope", SCOPECompressor)
register("pyramidkv", PyramidKVCompressor)

__all__ = [
    "SnapKVCompressor",
    "KIVICompressor",
    "SCOPECompressor",
    "PyramidKVCompressor",
]
