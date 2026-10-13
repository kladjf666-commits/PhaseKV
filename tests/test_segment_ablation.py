"""受控分段消融压缩器单测 (QA / CH-3): 验证只压某一段、另一段完全保留。"""
import pytest
import torch

from conftest import make_fake_attentions, make_fake_cache

from eval.compressors import build_compressor


def _seq_len(cache):
    return cache.key_cache[0].shape[-2]


def _append_token(cache, num_layers=2, num_kv_heads=2, head_dim=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    for layer in range(num_layers):
        k = torch.randn(1, num_kv_heads, 1, head_dim, generator=gen)
        v = torch.randn(1, num_kv_heads, 1, head_dim, generator=gen)
        cache.update(k, v, layer)


def test_requires_attention_flag():
    comp = build_compressor("segment_ablation", segment="prompt", prefill_keep_ratio=0.5)
    assert comp.requires_attention is True
    assert comp.physically_evicts is True


def test_missing_segment_raises():
    with pytest.raises(ValueError):
        build_compressor("segment_ablation", prefill_keep_ratio=0.5)


def test_missing_budget_raises():
    with pytest.raises(ValueError):
        build_compressor("segment_ablation", segment="prompt")


def test_prompt_segment_evicts_prompt_only():
    cache = make_fake_cache(num_layers=2, num_kv_heads=2, seq_len=12, head_dim=4)
    attn = make_fake_attentions(num_layers=2, num_heads=2, q_len=12, kv_len=12)
    comp = build_compressor("segment_ablation", segment="prompt",
                            prefill_keep_ratio=0.5, recent_window=4)
    cache = comp.on_prefill(cache, attentions=attn, prompt_len=12, model_config=None)
    assert _seq_len(cache) == 6  # round(12*0.5)=6

    # decode 段必须完全保留: 追加 3 个生成 token, on_decode_step 不应驱逐
    for step in range(3):
        _append_token(cache)
        a = make_fake_attentions(num_layers=2, num_heads=2, q_len=1, kv_len=_seq_len(cache))
        cache = comp.on_decode_step(cache, attentions=a, step=step, model_config=None)
    assert _seq_len(cache) == 6 + 3


def test_prompt_segment_requires_attentions_when_evicting():
    cache = make_fake_cache(num_layers=2, num_kv_heads=2, seq_len=12, head_dim=4)
    comp = build_compressor("segment_ablation", segment="prompt", prefill_keep_ratio=0.5)
    with pytest.raises(ValueError):
        comp.on_prefill(cache, attentions=None, prompt_len=12, model_config=None)


def test_generated_segment_keeps_prompt_intact_and_evicts_generated():
    cache = make_fake_cache(num_layers=2, num_kv_heads=2, seq_len=8, head_dim=4)
    prompt_k0 = cache.key_cache[0].clone()  # 完整 prompt 的 K (8 个)
    comp = build_compressor("segment_ablation", segment="generated",
                            max_cache_tokens=3, recent_window=2)
    cache = comp.on_prefill(cache, attentions=None, prompt_len=8, model_config=None)
    assert _seq_len(cache) == 8  # 只压 generated 时 prompt 全保留

    # decode 段增长到 6 个生成 token, generated 预算=3
    for step in range(6):
        _append_token(cache, seed=step + 1)
        a = make_fake_attentions(num_layers=2, num_heads=2, q_len=1, kv_len=_seq_len(cache))
        cache = comp.on_decode_step(cache, attentions=a, step=step, model_config=None)

    # prompt 段 (前 8 个) 必须原封不动
    assert torch.allclose(cache.key_cache[0][:, :, :8, :], prompt_k0)
    # 总长 = prompt(8) + generated 预算(3)
    assert _seq_len(cache) == 8 + 3


def test_generated_segment_keeps_recent_window():
    cache = make_fake_cache(num_layers=1, num_kv_heads=2, seq_len=4, head_dim=4)
    comp = build_compressor("segment_ablation", segment="generated",
                            decode_keep_ratio=0.5, recent_window=2)
    cache = comp.on_prefill(cache, attentions=None, prompt_len=4, model_config=None)
    last_recent = None
    for step in range(8):
        _append_token(cache, num_layers=1, seed=step + 100)
        last_recent = cache.key_cache[0][:, :, -1:, :].clone()  # 最新生成 token 的 K
        a = make_fake_attentions(num_layers=1, num_heads=2, q_len=1, kv_len=_seq_len(cache))
        cache = comp.on_decode_step(cache, attentions=a, step=step, model_config=None)
    # 强制 recent 窗口 => 最新生成的 token 必定仍在缓存末尾 (未被驱逐)
    assert torch.allclose(cache.key_cache[0][:, :, -1:, :], last_recent)


def test_keep_ratio_one_is_noop_for_prompt():
    cache = make_fake_cache(num_layers=2, num_kv_heads=2, seq_len=10, head_dim=4)
    attn = make_fake_attentions(num_layers=2, num_heads=2, q_len=10, kv_len=10)
    comp = build_compressor("segment_ablation", segment="prompt", prefill_keep_ratio=1.0)
    cache = comp.on_prefill(cache, attentions=attn, prompt_len=10, model_config=None)
    assert _seq_len(cache) == 10  # ratio=1.0 => 不驱逐 (全保留锚点)
