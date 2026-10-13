"""真实打包缓存的 CPU 行为测试，不依赖模型权重或 CUDA。"""
import unittest

import torch
from transformers.cache_utils import Cache, DynamicCache

from src.phasekv.phasekv_cache import PhaseKVCache
from src.phasekv.kernels.packing import unpack_dequantize
from src.phasekv.quant import quantize_key, quantize_value


def make_prefill(layers=2, length=20, batch=1, dtype=torch.float16):
    generator = torch.Generator().manual_seed(7)
    cache = DynamicCache()
    for layer in range(layers):
        key = torch.randn(batch, 2, length, 8, generator=generator, dtype=dtype)
        value = torch.randn(batch, 2, length, 8, generator=generator, dtype=dtype)
        cache.update(key, value, layer)
    return cache


class TestPhaseKVCache(unittest.TestCase):
    def test_bf16_prefill_is_packed_for_fp16_decode(self):
        cache = PhaseKVCache.from_dense_prefill(
            make_prefill(dtype=torch.bfloat16), prompt_len=20,
            prompt_bits=2, group_size=4)
        layer = cache.layer_tiers(0)
        self.assertEqual(layer.hot_k.dtype, torch.float16)
        self.assertEqual(layer.hot_v.dtype, torch.float16)
        self.assertEqual(layer.prompt.key.packed.dtype, torch.uint8)
        self.assertTrue(torch.isfinite(unpack_dequantize(layer.prompt.key)).all())

    def test_prefill_is_packed_with_key_and_value_on_different_axes(self):
        dense = make_prefill()
        indices = torch.tensor([[[0, 1, 4, 6, 9, 11, 15, 19],
                                 [0, 2, 3, 7, 9, 13, 17, 19]]])
        cache = PhaseKVCache.from_dense_prefill(
            dense, prompt_len=20, keep_indices=[indices, indices],
            prompt_bits=2, group_size=4, recent_window=2, lazy_g=2)

        self.assertIsInstance(cache, Cache)
        self.assertEqual(cache.get_seq_length(), 8)
        self.assertEqual(cache.true_seq_length, 20)
        layer = cache.layer_tiers(0)
        self.assertEqual(layer.prompt.key.axis, 2)
        self.assertEqual(layer.prompt.value.axis, 3)
        selected = indices.sort(dim=-1).values.unsqueeze(-1).expand(1, 2, 8, 8)
        raw_k = dense.key_cache[0].gather(2, selected)
        raw_v = dense.value_cache[0].gather(2, selected)
        self.assertTrue(torch.allclose(
            unpack_dequantize(layer.prompt.key), quantize_key(raw_k, 2, 4), atol=0.004))
        self.assertTrue(torch.allclose(
            unpack_dequantize(layer.prompt.value), quantize_value(raw_v, 2, 4), atol=0.004))
        self.assertEqual(dense.key_cache[0].shape[2], 20)

    def test_decode_cools_batches_without_repacking_old_segments(self):
        cache = PhaseKVCache.from_dense_prefill(
            make_prefill(), prompt_len=20, prompt_bits=2, cold_bits=4,
            group_size=4, recent_window=2, lazy_g=2, decode_keep_ratio=0.5)
        generator = torch.Generator().manual_seed(9)
        for _ in range(4):
            for layer_idx in range(len(cache)):
                key = torch.randn(1, 2, 1, 8, generator=generator, dtype=torch.float16)
                value = torch.randn(1, 2, 1, 8, generator=generator, dtype=torch.float16)
                cache.append_hot(layer_idx, key, value)
            cache.finish_decode_step()

        first_cold = cache.layer_tiers(0).cold[0]
        self.assertEqual(first_cold.length, 1)
        self.assertEqual(cache.layer_tiers(0).hot_len, 2)
        for _ in range(2):
            for layer_idx in range(len(cache)):
                key = torch.randn(1, 2, 1, 8, generator=generator, dtype=torch.float16)
                value = torch.randn(1, 2, 1, 8, generator=generator, dtype=torch.float16)
                cache.append_hot(layer_idx, key, value)
            cache.finish_decode_step()

        self.assertIs(cache.layer_tiers(0).cold[0], first_cold)
        self.assertEqual(len(cache.layer_tiers(0).cold), 2)
        self.assertEqual(cache.get_seq_length(), 24)
        self.assertEqual(cache.true_seq_length, 26)
        self.assertGreater(cache.physical_nbytes(), 0)

    def test_dense_update_is_rejected(self):
        cache = PhaseKVCache.from_dense_prefill(make_prefill(), prompt_len=20)
        key = torch.empty(1, 2, 1, 8, dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "不返回稠密"):
            cache.update(key, key, 0)
        cache.append_hot(0, key, key)
        with self.assertRaisesRegex(RuntimeError, "每一层"):
            cache.finish_decode_step()


if __name__ == "__main__":
    unittest.main()
