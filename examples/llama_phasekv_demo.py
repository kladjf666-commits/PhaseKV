"""运行：python -m examples.llama_phasekv_demo

用随机小 Llama 演示完整接入顺序；无需下载模型权重或 tokenizer。
随机权重只验证接口与缓存流转，不代表真实模型的语言能力。
"""
from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from examples.llama_phasekv_adapter import install_llama_phasekv_attention
from src.phasekv.phasekv_cache import PhaseKVCache


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("真实 PhaseKV Triton 路径需要 CUDA")

    config = LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0,
        pretraining_tp=1, _attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).half().cuda().eval()
    install_llama_phasekv_attention(model)

    # 首轮：原 attention 产生完整 DynamicCache，然后一次性打包。
    first_ids = torch.tensor([[3, 8, 12, 14, 21, 9, 10, 11]], device="cuda")
    first = model(input_ids=first_ids, past_key_values=DynamicCache(),
                  use_cache=True, output_attentions=False)
    cache = PhaseKVCache.from_dense_prefill(
        first.past_key_values, prompt_len=first_ids.shape[1],
        prompt_bits=4, cold_bits=4, group_size=32,
        recent_window=2, lazy_g=2, decode_keep_ratio=0.5,
    )
    del first
    print("prefill:", cache.true_seq_length, "真实 token，", cache.get_seq_length(), "保留 KV")

    # decode：每次只前向新 token；整个模型处理完所有层后结束这一步。
    for token_id in (22, 23, 24, 25, 26):
        true_pos = cache.true_seq_length
        physical_pos = cache.get_seq_length()
        output = model(
            input_ids=torch.tensor([[token_id]], device="cuda"),
            past_key_values=cache, use_cache=True, output_attentions=False,
            position_ids=torch.tensor([[true_pos]], device="cuda"),
            cache_position=torch.tensor([physical_pos], device="cuda"),
        )
        cache.finish_decode_step()
        assert torch.isfinite(output.logits).all()
    print("decode:", cache.true_seq_length, "真实 token，", cache.get_seq_length(), "保留 KV")

    # 下一轮：只前向新增的 prompt token，之前的 KV 不重新 prefill。
    new_ids = torch.tensor([[30, 31, 32]], device="cuda")
    true_pos = cache.true_seq_length
    physical_pos = cache.get_seq_length()
    cache.begin_prompt_chunk(new_ids.shape[1])
    output = model(
        input_ids=new_ids, past_key_values=cache,
        use_cache=True, output_attentions=False,
        position_ids=torch.arange(true_pos, true_pos + new_ids.shape[1], device="cuda").unsqueeze(0),
        cache_position=torch.arange(physical_pos, physical_pos + new_ids.shape[1], device="cuda"),
    )
    cache.finish_prompt_chunk()
    assert torch.isfinite(output.logits).all()
    print("新增 prompt:", cache.true_seq_length, "真实 token，", cache.get_seq_length(), "保留 KV")
    print("prompt 扩展段:", len(cache.layer_tiers(0).prompt_extensions),
          "cold 段:", len(cache.layer_tiers(0).cold))


if __name__ == "__main__":
    main()
