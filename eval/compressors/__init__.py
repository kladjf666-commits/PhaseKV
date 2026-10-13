"""压缩器注册表与工厂 (QA / CH-3 维护)。

团队新增方法时, 实现 ``KVCompressor`` 子类后, 在此用 ``register`` 登记即可被
评估框架按名调用。也支持外部 (src/ 或 baselines/) 通过 ``register`` 动态注入。
"""
from __future__ import annotations

from typing import Callable, Dict, Type

from ..interface import CompressionConfig, KVCompressor
from .attention_based import HeterogeneousReference, ObservedAttentionEviction
from .baselines import FullCache, KVQuant, RecentWindow, StreamingLLM
from .segment_ablation import SegmentAblationEvict

REGISTRY: Dict[str, Type[KVCompressor]] = {}


def register(name: str, cls: Type[KVCompressor]) -> None:
    if not issubclass(cls, KVCompressor):
        raise TypeError(f"{cls} 必须是 KVCompressor 子类")
    REGISTRY[name] = cls


def available() -> list:
    return sorted(REGISTRY.keys())


def build_compressor(method: str, **params) -> KVCompressor:
    """按注册名构造压缩器。

    Args:
        method: 注册名 (见 ``available()``)。
        **params: 写入 CompressionConfig 的字段; 未知字段进入 ``extra``。
    """
    if method not in REGISTRY:
        raise KeyError(f"未注册的压缩方法: {method}; 可用: {available()}")
    known = set(CompressionConfig.__dataclass_fields__.keys())
    cfg_kwargs = {k: v for k, v in params.items() if k in known}
    extra = {k: v for k, v in params.items() if k not in known}
    cfg_kwargs.setdefault("name", method)
    if extra:
        cfg_kwargs.setdefault("extra", {})
        cfg_kwargs["extra"].update(extra)
    config = CompressionConfig(**cfg_kwargs)
    return REGISTRY[method](config)


register("full", FullCache)
register("recent", RecentWindow)
register("streaming", StreamingLLM)
register("kvquant", KVQuant)
register("h2o", ObservedAttentionEviction)
register("hetero", HeterogeneousReference)
register("segment_ablation", SegmentAblationEvict)

__all__ = [
    "REGISTRY", "register", "available", "build_compressor",
    "FullCache", "RecentWindow", "StreamingLLM", "KVQuant",
    "ObservedAttentionEviction", "HeterogeneousReference", "SegmentAblationEvict",
]
