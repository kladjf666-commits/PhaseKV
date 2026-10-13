"""压缩器行为单测 (QA / CH-3): 在 CPU 小张量上验证驱逐 / 量化 / 内存折算。"""
import pytest
import torch

from conftest import make_fake_attentions, make_fake_cache

from eval.compressors import build_compressor
from eval.compressors import common


def _seq_len(cache):
    return cache.key_cache[0].shape[-2]


def test_full_cache_noop():
    cache = make_fake_cache(seq_len=8)
    comp = build_compressor("full")
    cache = comp.on_prefill(cache, prompt_len=8, model_config=None)
    cache = comp.on_decode_step(cache, step=0, model_config=None)
    assert _seq_len(cache) == 8


def test_recent_window_keeps_last_k():
    cache = make_fake_cache(seq_len=8, head_dim=4)
    original_last4 = cache.key_cache[0][:, :, -4:, :].clone()
    comp = build_compressor("recent", recent_window=4)
    cache = comp.on_prefill(cache, prompt_len=8, model_config=None)
    assert _seq_len(cache) == 4
    assert torch.allclose(cache.key_cache[0], original_last4)


def test_streaming_keeps_sink_and_recent():
    cache = make_fake_cache(seq_len=8, head_dim=4)
    k0 = cache.key_cache[0].clone()
    comp = build_compressor("streaming", sink_tokens=2, recent_window=4)
    cache = comp.on_prefill(cache, prompt_len=8, model_config=None)
    assert _seq_len(cache) == 6  # 2 sink + 4 recent
    # 保留的应为下标 [0,1,4,5,6,7]
    expected = k0[:, :, [0, 1, 4, 5, 6, 7], :]
    assert torch.allclose(cache.key_cache[0], expected)


def test_recent_window_decode_trims_growth():
    cache = make_fake_cache(seq_len=4, head_dim=4)
    comp = build_compressor("recent", recent_window=4)
    # 模拟 decode 增长: 追加 3 个 token 后裁剪
    for _ in range(3):
        k = torch.randn(1, 2, 1, 4)
        v = torch.randn(1, 2, 1, 4)
        cache.update(k, v, 0)
        cache.update(k, v, 1)
        cache = comp.on_decode_step(cache, step=0, model_config=None)
        assert _seq_len(cache) <= 4


def test_kvquant_preserves_shape_and_reduces_logical_bytes():
    cache = make_fake_cache(seq_len=8, head_dim=8)
    comp = build_compressor("kvquant", bits=4)
    full_bytes = sum(t.numel() * t.element_size()
                     for t in list(cache.key_cache) + list(cache.value_cache))
    cache = comp.on_prefill(cache, prompt_len=8, model_config=None)
    assert _seq_len(cache) == 8  # 量化不改变形状
    logical = comp.cache_memory_bytes(cache)
    assert logical < full_bytes  # 4bit 逻辑占用应远小于 float32 物理


def test_fake_quantize_roundtrip_bounds():
    x = torch.randn(4, 16)
    dq = common.fake_quantize(x, bits=8, axis=-1)
    assert dq.shape == x.shape
    # 8bit 量化误差应有界 (相对幅度)
    assert (x - dq).abs().max() <= (x.amax() - x.amin())


def test_h2o_prefill_respects_budget():
    cache = make_fake_cache(num_layers=2, num_kv_heads=2, seq_len=12, head_dim=4)
    attn = make_fake_attentions(num_layers=2, num_heads=2, q_len=12, kv_len=12)
    comp = build_compressor("h2o", prefill_keep_ratio=0.5, sink_tokens=2, recent_window=2)
    cache = comp.on_prefill(cache, attentions=attn, prompt_len=12, model_config=None)
    # 预算 = round(12*0.5)=6
    assert _seq_len(cache) == 6


def test_h2o_requires_attentions():
    cache = make_fake_cache(seq_len=12)
    comp = build_compressor("h2o", prefill_keep_ratio=0.5, recent_window=2)
    with pytest.raises(ValueError):
        comp.on_prefill(cache, attentions=None, prompt_len=12, model_config=None)


def test_hetero_prefill_then_decode():
    cache = make_fake_cache(num_layers=2, num_kv_heads=2, seq_len=20, head_dim=4)
    attn = make_fake_attentions(num_layers=2, num_heads=2, q_len=20, kv_len=20)
    comp = build_compressor("hetero", prefill_keep_ratio=0.3, sink_tokens=2, recent_window=4)
    cache = comp.on_prefill(cache, attentions=attn, prompt_len=20, model_config=None)
    k0 = _seq_len(cache)
    assert k0 == 6  # round(20*0.3)=6
    # decode 增长: 追加 6 个生成 token, decode 窗口=4, prefill 块不动
    for step in range(6):
        for layer in range(2):
            k = torch.randn(1, 2, 1, 4)
            v = torch.randn(1, 2, 1, 4)
            cache.update(k, v, layer)
        cache = comp.on_decode_step(cache, step=step, model_config=None)
    # 最终 = k0(6) + recent_window(4) = 10
    assert _seq_len(cache) == 10


def test_slice_cache_per_layer_indices():
    cache = make_fake_cache(num_layers=2, seq_len=6, head_dim=4)
    keep = [torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])]
    cache = common.slice_cache(cache, keep)
    assert cache.key_cache[0].shape[-2] == 3
    assert cache.key_cache[1].shape[-2] == 3
