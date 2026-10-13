"""Qwen2-1.5B 的 PhaseKV 交互式对话示例（单 GPU、batch=1）。

先用普通 Qwen2 attention 完成 prefill 并取观察窗口注意力分数；随后从原始
DynamicCache 直接选择、打包 Prompt KV。decode 逐 token 使用 PhaseKVCache
和自定义 Qwen2 attention。每轮以完整对话历史重新 prefill，复用已加载的模型。
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.cache_utils import DynamicCache

from src.phasekv import PhaseKVCache, install_qwen2_phasekv_attention
from src.phasekv.prompt_tier import select_prompt_indices


MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
MAX_NEW_TOKENS = 512
PROMPT_BITS = 2
COLD_BITS = 4
PREFILL_KEEP_RATIO = 0.5
DECODE_KEEP_RATIO = 0.5
SINK_TOKENS = 4
OBS_WINDOW = 32
RECENT_WINDOW = 32
LAZY_G = 16
GROUP_SIZE = 32


def sample_token(logits: torch.Tensor, *, temperature: float = 0.7,
                 top_k: int = 50, top_p: float = 0.95,
                 repetition_penalty: float = 1.0,
                 seen_token_ids: torch.Tensor | None = None,
                 generator: torch.Generator | None = None,
                 stage: str = "decode") -> torch.Tensor:
    """从 [1,vocab] logits 采样，返回形状 [1] 的 token。"""
    logits = logits.float()
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError(f"{stage} logits 含 NaN/Inf，无法采样；请检查模型计算精度")
    if temperature <= 0 or repetition_penalty <= 0:
        raise ValueError("temperature 和 repetition_penalty 必须为正")
    if repetition_penalty != 1.0 and seen_token_ids is not None:
        seen_token_ids = seen_token_ids.to(device=logits.device, dtype=torch.long)
        previous_scores = logits.gather(1, seen_token_ids)
        adjusted_scores = torch.where(
            previous_scores < 0,
            previous_scores * repetition_penalty,
            previous_scores / repetition_penalty,
        )
        logits = logits.scatter(1, seen_token_ids, adjusted_scores)
    logits = logits / temperature
    if top_k > 0:
        threshold = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < threshold, -float("inf"))
    sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
    probs = sorted_logits.softmax(dim=-1)
    if 0.0 < top_p < 1.0:
        remove = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        probs = sorted_logits.softmax(dim=-1)
    picked = torch.multinomial(probs, num_samples=1, generator=generator)
    return sorted_indices.gather(-1, picked).squeeze(-1)


def build_phasekv_cache(prefill, model, prompt_len: int, *,
                        prompt_bits: int = PROMPT_BITS, cold_bits: int = COLD_BITS,
                        prefill_keep_ratio: float = PREFILL_KEEP_RATIO,
                        decode_keep_ratio: float = DECODE_KEEP_RATIO,
                        sink_tokens: int = SINK_TOKENS, obs_window: int = OBS_WINDOW,
                        recent_window: int = RECENT_WINDOW, lazy_g: int = LAZY_G,
                        group_size: int = GROUP_SIZE) -> PhaseKVCache:
    """从普通 prefill 结果构建真实打包缓存，供对话和基准共用。"""
    keep = select_prompt_indices(
        prefill.attentions, prompt_len=prompt_len, model_config=model.config,
        prefill_keep_ratio=prefill_keep_ratio, sink_tokens=sink_tokens,
        obs_window=obs_window, pool_kernel=7, pooling="maxpool",
    )
    return PhaseKVCache.from_dense_prefill(
        prefill.past_key_values, prompt_len=prompt_len, keep_indices=keep,
        prompt_bits=prompt_bits, cold_bits=cold_bits, group_size=group_size,
        recent_window=recent_window, lazy_g=lazy_g,
        decode_keep_ratio=decode_keep_ratio,
    )


def eos_token_ids(generation_config) -> set[int]:
    eos = generation_config.eos_token_id
    return {int(token_id) for token_id in (eos if isinstance(eos, (list, tuple)) else [eos])
            if token_id is not None}


def sampling_options(generation_config) -> dict:
    return {
        "temperature": generation_config.temperature,
        "top_k": generation_config.top_k,
        "top_p": generation_config.top_p,
        "repetition_penalty": generation_config.repetition_penalty,
    }


@torch.inference_mode()
def generate_reply(model, tokenizer, messages: list[dict[str, str]]) -> tuple[str, int]:
    """基于本轮完整对话做 prefill，然后用 PhaseKVCache 逐 token 解码。"""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to("cuda")
    prompt_len = input_ids.shape[1]
    remaining = model.config.max_position_embeddings - prompt_len
    if remaining <= 0:
        raise ValueError("对话已超过模型上下文长度，请输入 /clear 开始新对话")
    max_new_tokens = min(MAX_NEW_TOKENS, remaining)
    positions = torch.arange(prompt_len, device=input_ids.device)
    generation_config = model.generation_config
    eos_ids = eos_token_ids(generation_config)
    seen_token_ids = input_ids
    sample_options = sampling_options(generation_config)

    # 按模型原生 BF16 做 prefill；output_attentions=True 仅用于此阶段的选择。
    prefill = model(
        input_ids=input_ids, past_key_values=DynamicCache(), use_cache=True,
        output_attentions=True, position_ids=positions.unsqueeze(0),
        cache_position=positions,
    )
    next_token = sample_token(
        prefill.logits[:, -1, :], seen_token_ids=seen_token_ids,
        stage="prefill", **sample_options)
    cache = build_phasekv_cache(prefill, model, prompt_len)
    del prefill

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    streamer.put(input_ids.cpu())
    generated_ids: list[int] = []
    for step in range(max_new_tokens):
        token_id = next_token.item()
        if token_id in eos_ids:
            break
        generated_ids.append(token_id)
        streamer.put(next_token.cpu())
        if step == max_new_tokens - 1:
            break
        seen_token_ids = torch.cat((seen_token_ids, next_token.view(1, 1)), dim=1)
        true_position = cache.true_seq_length
        physical_position = cache.get_seq_length()
        output = model(
            input_ids=next_token.view(1, 1), past_key_values=cache,
            use_cache=True, output_attentions=False,
            position_ids=torch.tensor([[true_position]], device=input_ids.device),
            cache_position=torch.tensor([physical_position], device=input_ids.device),
        )
        cache.finish_decode_step()
        next_token = sample_token(
            output.logits[:, -1, :], seen_token_ids=seen_token_ids,
            **sample_options)
        del output
    streamer.end()
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), cache.physical_nbytes()


def chat_loop(model, tokenizer) -> None:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    print("输入消息开始对话；输入 /clear 清空历史，/exit 退出。")
    while True:
        try:
            user_text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n对话结束。")
            return
        if user_text.lower() in {"/exit", "/quit"}:
            print("对话结束。")
            return
        if user_text.lower() == "/clear":
            messages = messages[:1]
            print("对话历史已清空。")
            continue
        if not user_text:
            continue
        messages.append({"role": "user", "content": user_text})
        print("模型：", end="", flush=True)
        reply, physical_bytes = generate_reply(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"[PhaseKV 物理 KV 字节: {physical_bytes:,}]")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PhaseKV Triton 注意力需要 CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("此 Qwen2 模型需要 BF16 prefill，但当前 GPU 不支持 BF16")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    install_qwen2_phasekv_attention(model)
    chat_loop(model, tokenizer)


if __name__ == "__main__":
    main()
