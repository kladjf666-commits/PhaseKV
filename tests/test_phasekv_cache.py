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

    def test_new_prompt_is_packed_without_rebuilding_old_cache(self):
        cache = PhaseKVCache.from_dense_prefill(
            make_prefill(), prompt_len=20, prompt_bits=4,
            recent_window=2, lazy_g=2)
        old_prompt = cache.layer_tiers(0).prompt
        key = torch.ones(1, 2, 1, 8, dtype=torch.float16)
        for layer_idx in range(len(cache)):
            cache.append_hot(layer_idx, key, key)
        cache.finish_decode_step()
        cache.begin_prompt_chunk(3)
        for layer_idx in range(len(cache)):
            for _ in range(3):
                cache.append_prompt_token(layer_idx, key, key)
        cache.finish_prompt_chunk()
        self.assertIs(cache.layer_tiers(0).prompt, old_prompt)
        self.assertEqual(cache.true_seq_length, 24)
        self.assertEqual(cache.get_seq_length(), 24)
        self.assertEqual(cache.layer_tiers(0).hot_len, 1)
        self.assertEqual(cache.layer_tiers(0).cold, [])
        self.assertTrue(torch.equal(cache.layer_tiers(0).hot_views()[0], key))
        self.assertTrue(torch.equal(cache.layer_tiers(0).hot_views()[1], key))
        self.assertEqual([s.length for s in cache.layer_tiers(0).prompt_extensions], [3])
        self.assertEqual(cache.layer_tiers(0).prompt_extensions[-1].key.bits, 4)

    def test_invalid_config_and_fractional_indices_fail_before_packing(self):
        dense = make_prefill()
        for name, value in (("prompt_bits", 3), ("cold_bits", 4.0),
                            ("group_size", 1.5), ("recent_window", True),
                            ("lazy_g", 0), ("decode_keep_ratio", float("nan"))):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    PhaseKVCache.from_dense_prefill(dense, prompt_len=20, **{name: value})
        indices = torch.tensor([[[0.5, 3.0], [1.0, 4.0]]])
        with self.assertRaisesRegex(ValueError, "整数张量"):
            PhaseKVCache.from_dense_prefill(
                dense, prompt_len=20, keep_indices=[indices, indices])
        with self.assertRaisesRegex(ValueError, "第 0 层.*prompt_len=19.*20"):
            PhaseKVCache.from_dense_prefill(dense, prompt_len=19)

    def test_invalid_layer_and_failed_writes_do_not_advance_state(self):
        cache = PhaseKVCache.from_dense_prefill(make_prefill(), prompt_len=20)
        key = torch.ones(1, 2, 1, 8, dtype=torch.float16)
        for layer_idx in (-1, 2, 0.5, True):
            with self.assertRaisesRegex(IndexError, "layer_idx"):
                cache.append_hot(layer_idx, key, key)
        with self.assertRaisesRegex(ValueError, "float16"):
            cache.append_hot(0, key.float(), key.float())
        self.assertEqual(cache.get_seq_length(), 20)
        cache.append_hot(0, key, key)
        self.assertEqual(cache.true_seq_length, 20)
        with self.assertRaisesRegex(RuntimeError, "第 0 层.*已追加"):
            cache.append_hot(0, key, key)
        with self.assertRaisesRegex(RuntimeError, r"尚未追加的层为 \[1\]"):
            cache.finish_decode_step()
        with self.assertRaisesRegex(RuntimeError, "decode 步内"):
            cache.begin_prompt_chunk(2)
        cache.append_hot(1, key, key)
        cache.finish_decode_step()
        self.assertEqual(cache.true_seq_length, 21)
        with self.assertRaisesRegex(RuntimeError, r"\[0, 1\]"):
            cache.finish_decode_step()

    def test_cooling_threshold_selection_and_hot_tail(self):
        cache = PhaseKVCache.from_dense_prefill(
            make_prefill(layers=1), prompt_len=20, group_size=4,
            recent_window=2, lazy_g=2, decode_keep_ratio=0.5)
        prompt = cache.layer_tiers(0).prompt
        tokens = []
        for step, strengths in enumerate(((9, 1), (1, 9), (3, 3), (4, 4))):
            value = torch.tensor(strengths, dtype=torch.float16).view(1, 2, 1, 1).expand(1, 2, 1, 8)
            key = torch.full_like(value, step + 1)
            tokens.append((key, value))
            cache.append_hot(0, key, value)
            # 当前 query 读取时，尚未执行本步冷却。
            self.assertEqual(cache.layer_tiers(0).hot_len, step + 1)
            self.assertEqual(cache.layer_tiers(0).cold_len, 0)
            cache.finish_decode_step()
            if step < 3:
                self.assertEqual(cache.layer_tiers(0).cold_len, 0)
        layer = cache.layer_tiers(0)
        # 两个 head 分别保留过期批的第 0 / 第 1 个 token。
        cold_key = unpack_dequantize(layer.cold[0].key)
        self.assertTrue(torch.equal(cold_key[0, :, 0, 0], torch.tensor([1., 2.])))
        self.assertTrue(torch.equal(unpack_dequantize(layer.cold[0].value), torch.full_like(cold_key, 9)))
        hot_k, hot_v = layer.hot_views()
        self.assertTrue(torch.equal(hot_k, torch.cat([t[0] for t in tokens[2:]], dim=2)))
        self.assertTrue(torch.equal(hot_v, torch.cat([t[1] for t in tokens[2:]], dim=2)))
        self.assertIs(layer.prompt, prompt)
        self.assertEqual(cache.true_seq_length, 24)
        self.assertEqual(cache.get_seq_length(), 23)

    def test_prompt_chunk_state_errors_and_decode_after_finish(self):
        cache = PhaseKVCache.from_dense_prefill(
            make_prefill(), prompt_len=20, recent_window=2, lazy_g=2)
        key = torch.ones(1, 2, 1, 8, dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "begin_prompt_chunk"):
            cache.append_prompt_token(0, key, key)
        for length in (0, 1.5, True):
            with self.assertRaisesRegex(ValueError, "length"):
                cache.begin_prompt_chunk(length)
        cache.begin_prompt_chunk(5)  # 超过正常热层容量，必须扩容。
        with self.assertRaisesRegex(RuntimeError, "prompt 块内"):
            cache.begin_prompt_chunk(1)
        with self.assertRaisesRegex(RuntimeError, "prompt 块内"):
            cache.reorder_cache(torch.tensor([0]))
        with self.assertRaisesRegex(RuntimeError, "finish_prompt_chunk"):
            cache.append_hot(0, key, key)
        with self.assertRaisesRegex(RuntimeError, "finish_prompt_chunk"):
            cache.finish_decode_step()
        with self.assertRaisesRegex(ValueError, "同一设备"):
            cache.append_prompt_token(0, key.to("meta"), key.to("meta"))
        with self.assertRaisesRegex(ValueError, "float16"):
            cache.append_prompt_token(0, key.float(), key.float())
        for _ in range(5):
            cache.append_prompt_token(0, key, key)
        with self.assertRaisesRegex(RuntimeError, "第 0 层已写满"):
            cache.append_prompt_token(0, key, key)
        with self.assertRaisesRegex(RuntimeError, "第 1 层.*期望 5 个，实际 0 个"):
            cache.finish_prompt_chunk()
        self.assertEqual(cache.true_seq_length, 20)
        self.assertEqual(cache.layer_tiers(0).prompt_extensions, [])
        for _ in range(5):
            cache.append_prompt_token(1, key, key)
        cache.finish_prompt_chunk()
        self.assertEqual(cache.true_seq_length, 25)
        self.assertEqual(cache.layer_tiers(0).hot_k.shape[2], 4)
        for layer_idx in range(len(cache)):
            cache.append_hot(layer_idx, key, key)
        cache.finish_decode_step()
        self.assertEqual(cache.true_seq_length, 26)

    def test_reorder_keeps_all_tiers_and_rejects_invalid_indices(self):
        cache = PhaseKVCache.from_dense_prefill(
            make_prefill(batch=2), prompt_len=20, recent_window=2, lazy_g=2)
        key = torch.arange(32, dtype=torch.float16).reshape(2, 2, 1, 8)
        for _ in range(4):
            for layer_idx in range(len(cache)):
                cache.append_hot(layer_idx, key, key)
            cache.finish_decode_step()
        cache.begin_prompt_chunk(1)
        for layer_idx in range(len(cache)):
            cache.append_prompt_token(layer_idx, key, key)
        cache.finish_prompt_chunk()
        for layer_idx in range(len(cache)):
            cache.append_hot(layer_idx, key, key)
        cache.finish_decode_step()
        for indices in (torch.tensor([-1]), torch.tensor([2]), torch.tensor([0.])):
            with self.assertRaisesRegex(ValueError, "beam_idx"):
                cache.reorder_cache(indices)
        layer = cache.layer_tiers(0)
        originals = [unpack_dequantize(s.key) for s in
                     [layer.prompt, *layer.prompt_extensions, *layer.cold]]
        original_hot = layer.hot_views()[0].clone()
        order = torch.tensor([1, 0, 1])
        cache.reorder_cache(order)
        for original, segment in zip(originals, [layer.prompt, *layer.prompt_extensions, *layer.cold]):
            self.assertTrue(torch.equal(unpack_dequantize(segment.key), original.index_select(0, order)))
        self.assertTrue(torch.equal(layer.hot_views()[0], original_hot.index_select(0, order)))
        self.assertEqual(cache.true_seq_length, 26)

    def test_prompt_chunks_preserve_decode_window_and_cooling_schedule(self):
        # 相同生成 KV 流，有/无穿插 prompt 时应得到完全一致的 hot/cold。
        # 分别覆盖热区不足窗口、已有待冷却 token、已有 cold，以及重复追加块。
        for keep_ratio in (0.5, 1.0):
            with self.subTest(keep_ratio=keep_ratio):
                options = dict(prompt_len=20, group_size=4, recent_window=2,
                               lazy_g=2, decode_keep_ratio=keep_ratio)
                baseline = PhaseKVCache.from_dense_prefill(make_prefill(), **options)
                cache = PhaseKVCache.from_dense_prefill(make_prefill(), **options)
                generator = torch.Generator().manual_seed(23)
                added_prompt_tokens = 0
                for step in range(8):
                    for layer_idx in range(len(cache)):
                        key = torch.randn(1, 2, 1, 8, generator=generator, dtype=torch.float16)
                        value = torch.randn(1, 2, 1, 8, generator=generator, dtype=torch.float16)
                        baseline.append_hot(layer_idx, key, value)
                        cache.append_hot(layer_idx, key, value)
                    baseline.finish_decode_step()
                    cache.finish_decode_step()
                    if step in (0, 2, 3, 6):
                        # 同时验证无需扩容与需要扩容的情况。
                        length = 1 if step == 0 else 5
                        frozen = [[layer.prompt, *layer.prompt_extensions, *layer.cold]
                                  for layer in cache.layers]
                        cache.begin_prompt_chunk(length)
                        for layer_idx in range(len(cache)):
                            for _ in range(length):
                                key = torch.randn(1, 2, 1, 8, generator=generator, dtype=torch.float16)
                                value = torch.randn(1, 2, 1, 8, generator=generator, dtype=torch.float16)
                                cache.append_prompt_token(layer_idx, key, value)
                        cache.finish_prompt_chunk()
                        added_prompt_tokens += length
                        self.assertEqual(cache.prompt_chunk_len, 0)
                        for layer, old_segments in zip(cache.layers, frozen):
                            retained = [layer.prompt, *layer.prompt_extensions[:-1], *layer.cold]
                            self.assertEqual(len(retained), len(old_segments))
                            for old, current in zip(old_segments, retained):
                                self.assertIs(old, current)
                            self.assertEqual(layer.prompt_extensions[-1].length, length)
                    for actual, expected in zip(cache.layers, baseline.layers):
                        self.assertEqual(actual.hot_len, expected.hot_len)
                        self.assertEqual(actual.hot_k.shape[2], 4)
                        for actual_hot, expected_hot in zip(actual.hot_views(), expected.hot_views()):
                            self.assertEqual(actual_hot.dtype, torch.float16)
                            self.assertTrue(torch.equal(actual_hot, expected_hot))
                        self.assertEqual(len(actual.cold), len(expected.cold))
                        for actual_seg, expected_seg in zip(actual.cold, expected.cold):
                            for name in ("key", "value"):
                                for field in ("packed", "scale", "mins"):
                                    self.assertTrue(torch.equal(
                                        getattr(getattr(actual_seg, name), field),
                                        getattr(getattr(expected_seg, name), field)))
                    self.assertEqual(cache.true_seq_length, 20 + step + 1 + added_prompt_tokens)
                    self.assertEqual(cache.get_seq_length(), baseline.get_seq_length() + added_prompt_tokens)


if __name__ == "__main__":
    unittest.main()
