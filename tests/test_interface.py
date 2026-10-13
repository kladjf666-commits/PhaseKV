"""接口契约单测 (QA / CH-3)。"""
import pytest

from conftest import make_fake_cache

from eval.compressors import REGISTRY, available, build_compressor
from eval.interface import CompressionConfig, KVCompressor


def test_registry_has_expected_methods():
    for name in ("full", "recent", "streaming", "kvquant", "h2o", "hetero"):
        assert name in REGISTRY
    assert set(available()) == set(REGISTRY.keys())


def test_all_registered_are_kvcompressor_subclasses():
    for cls in REGISTRY.values():
        assert issubclass(cls, KVCompressor)


def test_build_compressor_routes_extra_to_extra_field():
    comp = build_compressor("kvquant", bits=4, quant_axis=-1)
    assert comp.config.bits == 4
    assert comp.config.extra.get("quant_axis") == -1


def test_describe_contract():
    comp = build_compressor("recent", recent_window=16)
    d = comp.describe()
    assert d["name"] == "recent"
    assert d["class"] == "RecentWindow"
    assert d["requires_attention"] is False
    assert d["physically_evicts"] is True
    assert "config" in d


def test_requires_attention_flags():
    assert build_compressor("full").requires_attention is False
    assert build_compressor("recent", recent_window=8).requires_attention is False
    assert build_compressor("kvquant", bits=8).requires_attention is False
    assert build_compressor("h2o", prefill_keep_ratio=0.3, recent_window=8).requires_attention is True
    assert build_compressor("hetero", prefill_keep_ratio=0.3, recent_window=8).requires_attention is True


def test_default_cache_memory_bytes_matches_physical():
    cache = make_fake_cache(num_layers=2, num_kv_heads=2, seq_len=8, head_dim=4)
    comp = build_compressor("full")
    # 2 layers * (K+V) * 1*2*8*4 elems * 4B(float32 默认) = 2*2*64*4 = 1024 elems *4B
    expected = sum(t.numel() * t.element_size()
                   for t in list(cache.key_cache) + list(cache.value_cache))
    assert comp.cache_memory_bytes(cache) == expected


def test_config_validation():
    with pytest.raises(ValueError):
        CompressionConfig(name="x", prefill_keep_ratio=1.5).validate()
    with pytest.raises(ValueError):
        CompressionConfig(name="x", bits=5).validate()
    with pytest.raises(ValueError):
        CompressionConfig(name="x", recent_window=0).validate()
    CompressionConfig(name="ok", prefill_keep_ratio=0.5, bits=4, recent_window=8).validate()


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        build_compressor("does_not_exist")
