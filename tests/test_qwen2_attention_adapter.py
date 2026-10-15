"""Qwen2 适配器对普通前向的透明性测试，无需下载模型权重。"""
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from qwen2 import append_prompt_chunk, decode_token
from src.phasekv.phasekv_cache import PhaseKVCache

from src.phasekv.qwen2_attention import (
    PhaseKVQwen2Attention,
    install_qwen2_phasekv_attention,
)


class TestQwen2Adapter(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "真实 packed attention 需要 CUDA")
    @torch.inference_mode()
    def test_prompt_block_matches_causal_token_forwards_and_preserves_hot(self):
        torch.manual_seed(7)
        config = Qwen2Config(
            vocab_size=97, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, attention_dropout=0.0,
            attn_implementation="eager",
        )
        model = Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.float16).eval()
        install_qwen2_phasekv_attention(model)
        prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], device="cuda")
        prefill = model(input_ids=prompt, past_key_values=DynamicCache(), use_cache=True)
        keep = torch.tensor([[[0, 2, 5], [0, 3, 5]]], device="cuda")
        options = dict(prompt_len=6, keep_indices=[keep, keep], group_size=4,
                       recent_window=8, lazy_g=2)
        block_cache = PhaseKVCache.from_dense_prefill(prefill.past_key_values, **options)
        serial_cache = PhaseKVCache.from_dense_prefill(prefill.past_key_values, **options)
        # 真实位置与物理位置已经不同；先留下一个生成热 token。
        for cache in (block_cache, serial_cache):
            decode_token(model, cache, torch.tensor([7], device="cuda"))
        old_hot = [(k.clone(), v.clone()) for k, v in
                   (layer.hot_views() for layer in block_cache.layers)]
        new_ids = torch.tensor([[8, 9, 10]], device="cuda")
        block_cache.begin_prompt_chunk(3)
        block_output = model(
            input_ids=new_ids, past_key_values=block_cache, use_cache=True,
            position_ids=torch.arange(7, 10, device="cuda").unsqueeze(0),
            cache_position=torch.arange(4, 7, device="cuda"),
        )
        serial_logits = [decode_token(model, serial_cache, token)
                         for token in new_ids[0]]
        # 逐 token 前向的早期 query 不可能读取后续输入，作为块内因果性的对照。
        torch.testing.assert_close(block_output.logits, torch.stack(serial_logits, dim=1),
                                   atol=0.002, rtol=0.002)
        block_cache.finish_prompt_chunk()
        self.assertEqual(block_cache.true_seq_length, 10)
        for layer, (key, value) in zip(block_cache.layers, old_hot):
            self.assertEqual(layer.hot_len, 1)
            self.assertEqual(layer.cold_len, 0)
            self.assertEqual(layer.prompt_extensions[-1].length, 3)
            self.assertTrue(torch.equal(layer.hot_views()[0], key))
            self.assertTrue(torch.equal(layer.hot_views()[1], value))
        # 块结束后继续 decode，再追加单 token prompt，验证两种阶段可以交替。
        logits = decode_token(model, block_cache, torch.tensor([11], device="cuda"))
        self.assertTrue(torch.isfinite(logits).all())
        logits = append_prompt_chunk(model, block_cache, [12])
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(block_cache.true_seq_length, 12)
        self.assertEqual(block_cache.layer_tiers(0).hot_len, 2)

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
