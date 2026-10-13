"""模型运行时: 加载模型、带压缩的生成循环、流式困惑度 (QA / CH-3 维护)。

关键点 (与 transformers 4.44.x DynamicCache 对齐):
  - 压缩 (eviction) 会使物理缓存长度 < 真实已见 token 数。本运行时对每次 decode/chunk
    前向显式传入"真实绝对位置"的 ``position_ids`` 以保证 RoPE 正确; 而 ``cache_position``
    交由模型按物理长度推断, 用于把新 token 写入物理缓存末尾并构建因果掩码。
  - 需要注意力分数的压缩器 (requires_attention=True) 以 eager attention 加载, 否则用 sdpa。
"""
from __future__ import annotations

import time
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from . import metrics
from .interface import KVCompressor


def _physical_len(cache) -> int:
    if cache is None or len(cache.key_cache) == 0 or cache.key_cache[0] is None:
        return 0
    return cache.key_cache[0].shape[-2]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample(logits: torch.Tensor, greedy: bool, temperature: float, top_p: float,
            generator: Optional[torch.Generator]) -> torch.Tensor:
    """从 [batch, vocab] logits 采样下一个 token, 返回 [batch] 的 LongTensor。"""
    if greedy:
        return torch.argmax(logits, dim=-1)#永远找概率最大的返回
    logits = logits.float() / max(1e-6, temperature)
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum - sorted_probs <= top_p
        keep[..., 0] = True
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
        return sorted_idx.gather(-1, choice).squeeze(-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def load_model_and_tokenizer(model_name: str, *, dtype: str = "float16",
                             device: str = "cuda", attn_implementation: str = "sdpa",
                             trust_remote_code: bool = True):
    """加载模型与 tokenizer。dtype ∈ {float16,bfloat16,float32}。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_with_compression(model, tokenizer, input_ids: torch.Tensor,
                              compressor: KVCompressor, *, max_new_tokens: int = 64,
                              greedy: bool = True, temperature: float = 1.0,
                              top_p: float = 1.0, stop_on_eos: bool = True,
                              measure_memory: bool = True, seed: int = 1234) -> dict:
    """带 KV 压缩的自回归生成, 返回生成结果与各项指标。"""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    #将维度确定为[batch_size,seq_length]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    prompt_len = input_ids.shape[1]
    out_attn = compressor.requires_attention
    eos_id = tokenizer.eos_token_id if stop_on_eos else None
    generator = torch.Generator(device=device).manual_seed(seed) if not greedy else None

    compressor.reset(model.config)
    if measure_memory:
        metrics.reset_peak_gpu_memory(device)

    _sync(device)
    t0 = time.perf_counter()
    # 显式传入空 DynamicCache (而非 None), 否则 transformers 会返回 legacy 元组缓存。
    # 新式 RoPE: position_ids 驱动 cos/sin, cache_position 驱动写入与因果掩码 (解耦)。
    prefill_positions = torch.arange(prompt_len, device=device).unsqueeze(0)
    out = model(input_ids=input_ids, use_cache=True, output_attentions=out_attn,
                past_key_values=DynamicCache(), position_ids=prefill_positions,
                cache_position=torch.arange(prompt_len, device=device))
    _sync(device)
    ttft = time.perf_counter() - t0

    cache = out.past_key_values
    #压缩promote cache
    cache = compressor.on_prefill(
        cache, attentions=(out.attentions if out_attn else None),
        prompt_len=prompt_len, model_config=model.config)

    next_token = _sample(out.logits[:, -1, :], greedy, temperature, top_p, generator)
    generated = [int(next_token.item())]
    stats = metrics.LatencyStats(ttft_s=ttft, num_generated=1)
    cache_bytes_trace = [compressor.cache_memory_bytes(cache)]
    full_bytes_trace = [_full_cache_bytes(cache)]
    true_pos = prompt_len

    for step in range(max_new_tokens - 1):
        if eos_id is not None and next_token.item() == eos_id:
            break
        # 真实位置驱动 RoPE; 物理 cache_position 驱动缓存写入与因果掩码 (二者解耦)。
        position_ids = torch.tensor([[true_pos]], device=device, dtype=torch.long)
        phys = _physical_len(cache)
        cache_position = torch.tensor([phys], device=device, dtype=torch.long)
        _sync(device)
        t1 = time.perf_counter()
        out = model(input_ids=next_token.view(1, 1), past_key_values=cache,
                    use_cache=True, output_attentions=out_attn,
                    position_ids=position_ids, cache_position=cache_position)
        _sync(device)
        stats.decode_step_times_s.append(time.perf_counter() - t1)

        cache = out.past_key_values
        cache = compressor.on_decode_step(
            cache, attentions=(out.attentions if out_attn else None),
            step=step, model_config=model.config)

        next_token = _sample(out.logits[:, -1, :], greedy, temperature, top_p, generator)
        generated.append(int(next_token.item()))
        stats.num_generated += 1
        cache_bytes_trace.append(compressor.cache_memory_bytes(cache))
        full_bytes_trace.append(_full_cache_bytes(cache))
        true_pos += 1

    peak = metrics.peak_gpu_memory_bytes(device) if measure_memory else 0
    return {
        "generated_ids": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "latency": stats,
        "cache_bytes_trace": cache_bytes_trace,
        "final_cache_bytes": cache_bytes_trace[-1],
        "full_cache_bytes": full_bytes_trace[-1],
        "peak_gpu_bytes": peak,
        "prompt_len": prompt_len,
        "compressor": compressor.describe(),
    }


def _full_cache_bytes(cache) -> int:
    """对照: 同样 token 数下未压缩 (fp16 物理) 的字节数。"""
    total = 0
    for tensor in list(cache.key_cache) + list(cache.value_cache):
        if tensor is None:
            continue
        total += tensor.numel() * 2  # fp16 基准
    return total


@torch.no_grad()
def streaming_perplexity(model, compressor: KVCompressor, input_ids: torch.Tensor, *,
                         chunk_size: int = 512) -> dict:
    """流式困惑度: 分块处理长序列, 块间压缩缓存, 累计 NLL。

    复用压缩器接口: 第一块视为 prefill (on_prefill), 后续块视为 decode 维护
    (on_decode_step), 因此压缩器应实现与 chunk 粒度无关的预算式裁剪。
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    seq_len = input_ids.shape[1]
    out_attn = compressor.requires_attention

    compressor.reset(model.config)
    acc = metrics.PerplexityAccumulator()
    cache = None
    prev_last_logits = None  # 上一块最后位置的 logits, 预测当前块第一个 token
    true_pos = 0
    chunk_index = 0

    pos = 0
    while pos < seq_len:
        chunk = input_ids[:, pos:pos + chunk_size]
        clen = chunk.shape[1]
        position_ids = torch.arange(true_pos, true_pos + clen, device=device).unsqueeze(0)
        phys = _physical_len(cache)
        cache_position = torch.arange(phys, phys + clen, device=device)
        if cache is None:
            cache = DynamicCache()
        out = model(input_ids=chunk, past_key_values=cache, use_cache=True,
                    output_attentions=out_attn, position_ids=position_ids,
                    cache_position=cache_position)
        logits = out.logits  # [1, clen, vocab]

        if prev_last_logits is not None and clen >= 1:
            acc.update(prev_last_logits.unsqueeze(1), chunk[:, :1])
        if clen >= 2:
            acc.update(logits[:, :-1, :], chunk[:, 1:])
        prev_last_logits = logits[:, -1, :]

        cache = out.past_key_values
        if chunk_index == 0:
            cache = compressor.on_prefill(
                cache, attentions=(out.attentions if out_attn else None),
                prompt_len=clen, model_config=model.config)
        else:
            cache = compressor.on_decode_step(
                cache, attentions=(out.attentions if out_attn else None),
                step=chunk_index, model_config=model.config)

        true_pos += clen
        pos += clen
        chunk_index += 1

    return {
        "perplexity": acc.perplexity,
        "mean_nll": acc.mean_nll,
        "total_tokens": acc.total_tokens,
        "final_cache_bytes": compressor.cache_memory_bytes(cache) if cache is not None else 0,
        "full_cache_bytes": _full_cache_bytes(cache) if cache is not None else 0,
        "compressor": compressor.describe(),
    }
