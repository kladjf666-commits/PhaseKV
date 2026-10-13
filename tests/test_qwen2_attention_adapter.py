"""Qwen2 适配器对普通前向的透明性测试，无需下载模型权重。"""
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.phasekv.qwen2_attention import (
    PhaseKVQwen2Attention,
    install_qwen2_phasekv_attention,
)


class TestQwen2Adapter(unittest.TestCase):
    def test_install_preserves_uncompressed_prefill(self):
        config = Qwen2Config(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
        model = Qwen2ForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            original = model(input_ids=input_ids, use_cache=False).logits
            install_qwen2_phasekv_attention(model)
            wrapped = model(input_ids=input_ids, use_cache=False).logits
        self.assertIsInstance(model.model.layers[0].self_attn, PhaseKVQwen2Attention)
        self.assertFalse(model.model.layers[0].self_attn.training)
        self.assertTrue(torch.equal(original, wrapped))
        install_qwen2_phasekv_attention(model)
        self.assertIsInstance(model.model.layers[0].self_attn, PhaseKVQwen2Attention)


if __name__ == "__main__":
    unittest.main()
