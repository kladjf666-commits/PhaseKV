"""教学示例：将 Transformers 4.46.3 的 Llama eager attention 接到 PhaseKV。

这是 attention 适配层，不包含模型下载、chat template 或采样循环。
首次 prefill 仍走原始 LlamaAttention；传入 PhaseKVCache 后才走打包 KV 路径。
限制：CUDA、batch=1、无 padding、pretraining_tp=1、标准全局因果注意力。
"""
from __future__ import annotations

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

from src.phasekv.kernels.attention_triton import fused_phasekv_decode
from src.phasekv.phasekv_cache import PhaseKVCache


class PhaseKVLlamaAttention(nn.Module):
    def __init__(self, original: LlamaAttention):
        super().__init__()
        # 持有原模块，直接复用训练好的 q/k/v/o_proj 权重。
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
        **kwargs,
    ):
        # 普通 DynamicCache（包括首次 prefill）必须原样走官方实现。
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
                **kwargs,
            )

        batch, q_len, _ = hidden_states.shape   #原样复制Llama源码
        base = self.original
        if batch != 1 or hidden_states.device.type != "cuda":
            raise ValueError("示例仅支持单条无 padding 的 CUDA 输入")
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("示例仅支持 FP16/BF16 模型")
        if self.training or not use_cache or output_attentions:
            raise ValueError("PhaseKV 路径需要 eval、use_cache=True、output_attentions=False")
        if self.layer_idx is None or base.config.pretraining_tp != 1:
            raise ValueError("示例需要 layer_idx，且 pretraining_tp=1")

        # 1. 必须与目标模型原始 attention 使用相同的投影和维度布局。
        q = base.q_proj(hidden_states).view(batch, q_len, base.num_heads, base.head_dim).transpose(1, 2)
        k = base.k_proj(hidden_states).view(batch, q_len, base.num_key_value_heads, base.head_dim).transpose(1, 2)
        v = base.v_proj(hidden_states).view(batch, q_len, base.num_key_value_heads, base.head_dim).transpose(1, 2)

        # 2. 使用模型计算的真实绝对位置对 Q/K 施加 RoPE，然后才写入 KV。
        if position_embeddings is None:
            if position_ids is None:
                raise ValueError("缺少真实绝对 position_ids")
            cos, sin = base.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)

        # 3. 先写当前 token 的 KV，再让当前 Q 看见它以及所有保留的过去 KV。
        if past_key_value.prompt_chunk_len: #
            if q_len != past_key_value.prompt_chunk_len:
                raise ValueError("新增 prompt 长度与缓存预留长度不一致")
            parts = []#用于临时保存结果
            for index in range(q_len):#一次只能处理一个 token，逐个 token 追加到 PhaseKV 缓存，速度慢！后续可以考虑加入mask矩阵，改成一次性处理多个 token
                past_key_value.append_prompt_token(self.layer_idx, k[:, :, index:index + 1, :], v[:, :, index:index + 1, :])
                parts.append(fused_phasekv_decode(q[0, :, index, :], past_key_value.layer_tiers(self.layer_idx), base.num_heads))
            context = torch.stack(parts, dim=0).reshape(batch, q_len, base.num_heads * base.head_dim)
        else:
            if q_len != 1:
                raise ValueError("普通 decode 每次只处理一个 token")
            past_key_value.append_hot(self.layer_idx, k, v)
            context = fused_phasekv_decode(
                q[0, :, 0, :], past_key_value.layer_tiers(self.layer_idx), base.num_heads,
            ).reshape(batch, 1, base.num_heads * base.head_dim)

        # 4. 把各 head 拼回 hidden size，执行原模型的输出投影。
        return base.o_proj(context.to(hidden_states.dtype)), None, past_key_value


def install_llama_phasekv_attention(model) -> None:
    """模型加载完成后调用一次；保持原权重和普通 prefill 行为。"""
    if model.config.model_type != "llama" or model.config._attn_implementation != "eager":
        raise TypeError("示例仅适配 transformers 4.46.3 的 Llama eager attention")
    if model.config.pretraining_tp != 1:
        raise ValueError("示例尚未实现 pretraining_tp > 1 的投影切分")
    for layer in model.model.layers:
        original = layer.self_attn
        if isinstance(original, PhaseKVLlamaAttention):
            continue
        if type(original) is not LlamaAttention:
            raise TypeError(f"不支持的 attention 模块: {type(original).__name__}")
        layer.self_attn = PhaseKVLlamaAttention(original)
