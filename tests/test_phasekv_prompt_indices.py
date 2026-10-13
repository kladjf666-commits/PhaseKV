"""Prompt 保留下标的预算、顺序和去重检查。"""
import unittest
from types import SimpleNamespace

import torch

from src.phasekv.prompt_tier import select_prompt_indices


class TestPromptIndices(unittest.TestCase):
    def test_short_prompt_stays_within_budget(self):
        for prompt_len, sink, expected in ((20, 4, 10), (40, 4, 20), (40, 16, 20)):
            attention = torch.ones(1, 2, prompt_len, prompt_len) / prompt_len
            keep = select_prompt_indices(
                (attention,), prompt_len=prompt_len,
                model_config=SimpleNamespace(num_key_value_heads=2, num_attention_heads=2),
                prefill_keep_ratio=0.5, sink_tokens=sink,
                obs_window=32, pool_kernel=7, pooling="maxpool",
            )[0]
            self.assertEqual(keep.shape, (1, 2, expected))
            self.assertTrue(bool((keep[..., 1:] > keep[..., :-1]).all()))
            self.assertGreaterEqual(int(keep.min()), 0)
            self.assertLess(int(keep.max()), prompt_len)


if __name__ == "__main__":
    unittest.main()
