"""Qwen2 示例采样的数值检查。"""
import unittest

import torch

from qwen2 import sample_token


class TestQwen2Sampling(unittest.TestCase):
    def test_samples_finite_logits(self):
        token = sample_token(torch.tensor([[0.0, 1.0, 2.0]]), top_k=1)
        self.assertEqual(token.item(), 2)

    def test_rejects_nonfinite_logits_before_multinomial(self):
        with self.assertRaisesRegex(RuntimeError, "prefill logits 含 NaN/Inf"):
            sample_token(torch.tensor([[float("nan"), 0.0]]), stage="prefill")

    def test_repetition_penalty_uses_previous_tokens(self):
        token = sample_token(
            torch.tensor([[0.0, 1.0, 2.0]]), top_k=1,
            repetition_penalty=10.0, seen_token_ids=torch.tensor([[2]]))
        self.assertEqual(token.item(), 1)


if __name__ == "__main__":
    unittest.main()
