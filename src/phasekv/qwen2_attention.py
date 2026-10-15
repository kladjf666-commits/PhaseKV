"""Transformers 4.46.x Qwen2 的 PhaseKV decode 注意力适配器。

在模型加载完成后调用 ``install_qwen2_phasekv_attention(model)``。首次 prefill
仍由原注意力模块执行；已有 ``PhaseKVCache`` 上的新增 prompt 按块输入，
块内逐 token 做因果注意力。限制 batch=1、CUDA、无 padding。
"""
from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
)

from .phasekv_cache import PhaseKVCache


class PhaseKVQwen2Attention(nn.Module):
    """复用原投影权重，处理 PhaseKV 单 token decode 和后续 prompt 块。

    本层只写入 KV、计算注意力。finish_decode_step/finish_prompt_chunk
    必须由外层在整个模型前向返回后调用，不能在单个 attention 层中调用。
    """

    def __init__(self, original: Qwen2Attention):
        super().__init__()
        self.original = original
        self.layer_idx = original.layer_idx
        self.train(original.training)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        if not isinstance(past_key_value, PhaseKVCache):
            return self.original(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        batch, q_len, _ = hidden_states.shape
        if batch != 1 or hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("PhaseKV Qwen2 目前只支持 batch=1、float16/bfloat16")
        if hidden_states.device.type != "cuda":
            raise ValueError("PhaseKV Triton decode 需要 CUDA")
        if self.training or output_attentions or not use_cache:
            raise ValueError("PhaseKV decode 需要 eval()、use_cache=True、output_attentions=False")
        if self.layer_idx is None:
            raise ValueError("Qwen2 attention 缺少 layer_idx")

        # 在投影、写入之前确定阶段。新增 prompt 即使只有一个 token，也走块接口。
        cache = past_key_value
        if cache.prompt_chunk_len:
            if q_len != cache.prompt_chunk_len:
                raise ValueError(
                    f"新增 prompt 需要 {cache.prompt_chunk_len} 个 token，实际为 {q_len}")
            append_token = cache.append_prompt_token
        else:
            if q_len != 1:
                raise ValueError(
                    f"普通 PhaseKV decode 需要 q_len=1，实际为 {q_len}；"
                    "新增 prompt 请先调用 begin_prompt_chunk(length)")
            append_token = cache.append_hot

        base = self.original
        q = base.q_proj(hidden_states).view(batch, q_len, base.num_heads, base.head_dim).transpose(1, 2)
        k = base.k_proj(hidden_states).view(
            batch, q_len, base.num_key_value_heads, base.head_dim).transpose(1, 2)
        v = base.v_proj(hidden_states).view(
            batch, q_len, base.num_key_value_heads, base.head_dim).transpose(1, 2)

        if position_embeddings is None:
            if position_ids is None:
                raise ValueError("必须提供真实绝对 position_ids 以计算 RoPE")
            cos, sin = base.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # packed kernel 与热层当前使用 FP16；QK 点积在 kernel 内以 FP32 累加。
        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)

        from .kernels.attention_triton import fused_phasekv_decode

        # 当前 kernel 是单 query 接口：写一个、算一个，不能提前写入未来 token。
        # 无 padding 的因果可见性由该顺序保证，不依赖模型生成的稠密 causal mask。
        tiers = cache.layer_tiers(self.layer_idx)
        outputs = []
        for index in range(q_len):
            append_token(
                self.layer_idx, k[:, :, index:index + 1, :],
                v[:, :, index:index + 1, :])
            outputs.append(fused_phasekv_decode(q[0, :, index, :], tiers, base.num_heads))
        output = (outputs[0] if q_len == 1 else torch.stack(outputs, dim=0))
        output = output.reshape(batch, q_len, base.hidden_size)
        return base.o_proj(output.to(hidden_states.dtype)), None, cache


def install_qwen2_phasekv_attention(model) -> None:
    """就地包装 Qwen2ForCausalLM 的每层注意力，不修改 Transformers 安装文件。"""
    if getattr(model.config, "model_type", None) != "qwen2":
        raise TypeError("当前适配器仅支持 model_type='qwen2' 的模型")
    for layer in model.model.layers:
        original = layer.self_attn
        if isinstance(original, PhaseKVQwen2Attention):
            continue
        if not isinstance(original, Qwen2Attention):
            raise TypeError(f"不支持的 Qwen2 attention 类型: {type(original).__name__}")
        layer.self_attn = PhaseKVQwen2Attention(original)
