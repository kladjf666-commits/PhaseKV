"""Qwen2-1.5B 的 PhaseKV 交互式对话示例（单 GPU、batch=1）。

先用普通 Qwen2 attention 完成首次 prefill；需要裁剪 Prompt 时才收集注意力分数。
随后从原始 DynamicCache 直接选择、打包 Prompt KV。后续轮次只把新增的聊天模板 token
送入同一份 PhaseKVCache；decode 逐 token 使用自定义 Qwen2 attention。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.cache_utils import DynamicCache

from src.phasekv import PhaseKVCache, install_qwen2_phasekv_attention
from src.phasekv.prompt_tier import select_prompt_indices


MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
#MAX_NEW_TOKENS = 512
#PROMPT_BITS = 4
#COLD_BITS = 4
#PREFILL_KEEP_RATIO = 1.0
#DECODE_KEEP_RATIO = 0.5
#SINK_TOKENS = 4
#OBS_WINDOW = 32
#RECENT_WINDOW = 32
#LAZY_G = 16
#GROUP_SIZE = 32


MAX_NEW_TOKENS = 2048
PROMPT_BITS = 4
COLD_BITS = 4
PREFILL_KEEP_RATIO = 1.0
DECODE_KEEP_RATIO = 1.0
SINK_TOKENS = 4
OBS_WINDOW = 32
RECENT_WINDOW = 128
LAZY_G = 128
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


@dataclass
class ChatSession:
    cache: PhaseKVCache | None = None
    token_ids: list[int] = field(default_factory=list)


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


def build_phasekv_cache(prefill, model, prompt_len: int, *,
                        prompt_bits: int = PROMPT_BITS, cold_bits: int = COLD_BITS,
                        prefill_keep_ratio: float = PREFILL_KEEP_RATIO,
                        decode_keep_ratio: float = DECODE_KEEP_RATIO,
                        sink_tokens: int = SINK_TOKENS, obs_window: int = OBS_WINDOW,
                        recent_window: int = RECENT_WINDOW, lazy_g: int = LAZY_G,
                        group_size: int = GROUP_SIZE) -> PhaseKVCache:
    """首次 prefill 后建立打包缓存；完整保留时无需注意力分数。"""
    keep = (None if prefill_keep_ratio == 1.0 else select_prompt_indices(
        prefill.attentions, prompt_len=prompt_len, model_config=model.config,
        prefill_keep_ratio=prefill_keep_ratio, sink_tokens=sink_tokens,
        obs_window=obs_window, pool_kernel=7, pooling="maxpool"))
    return PhaseKVCache.from_dense_prefill(
        prefill.past_key_values, prompt_len=prompt_len, keep_indices=keep,
        prompt_bits=prompt_bits, cold_bits=cold_bits, group_size=group_size,
        recent_window=recent_window, lazy_g=lazy_g,
        decode_keep_ratio=decode_keep_ratio)


def assistant_end_tokens(tokenizer) -> list[int]:
    template = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": ""}], tokenize=True,
        add_generation_prompt=False)
    if len(template) < 2 or template[-2] != tokenizer.eos_token_id:
        raise ValueError("当前 Qwen2 chat 模板的 assistant 结束符不受支持")
    return template[-2:]


def new_user_prompt_tokens(tokenizer, system_message: dict, user_message: dict) -> list[int]:
    """从 chat 模板截取用户轮次，避免单独模板自动补入 system 消息。"""
    system_prefix = tokenizer.apply_chat_template(
        [system_message], tokenize=True, add_generation_prompt=False)
    with_user = tokenizer.apply_chat_template(
        [system_message, user_message], tokenize=True, add_generation_prompt=True)
    if with_user[:len(system_prefix)] != system_prefix:
        raise ValueError("chat 模板的 system 前缀不一致")
    return with_user[len(system_prefix):]


@torch.inference_mode()
def append_prompt_chunk(model, cache: PhaseKVCache, token_ids: list[int]) -> torch.Tensor:
    """只前向新增 token；旧 Prompt 与已生成 KV 保持在原缓存中。"""
    if not token_ids:
        raise ValueError("新增 prompt 不能为空")
    device = next(model.parameters()).device
    true_start = cache.true_seq_length
    physical_start = cache.get_seq_length()
    length = len(token_ids)
    cache.begin_prompt_chunk(length)
    output = model(
        input_ids=torch.tensor([token_ids], device=device), past_key_values=cache,
        use_cache=True, output_attentions=False,
        position_ids=torch.arange(true_start, true_start + length, device=device).unsqueeze(0),
        cache_position=torch.arange(physical_start, physical_start + length, device=device),
    )
    cache.finish_prompt_chunk()
    logits = output.logits[:, -1, :].clone()
    del output
    return logits


@torch.inference_mode()
def decode_token(model, cache: PhaseKVCache, token: torch.Tensor) -> torch.Tensor:
    """前向一个生成 token，并在所有模型层返回后提交本步；返回下一 token 的 logits。"""
    device = next(model.parameters()).device
    output = model(
        input_ids=token.reshape(1, 1), past_key_values=cache,
        use_cache=True, output_attentions=False,
        position_ids=torch.tensor([[cache.true_seq_length]], device=device),
        cache_position=torch.tensor([cache.get_seq_length()], device=device),
    )
    cache.finish_decode_step()
    return output.logits[:, -1, :]


@torch.inference_mode()
def generate_reply(model, tokenizer, messages: list[dict[str, str]],
                   session: ChatSession, *, max_new_tokens: int = MAX_NEW_TOKENS) -> tuple[str, int]:
    """首次 prefill；后续轮次只前向新用户消息并复用 PhaseKVCache。"""
    if not messages or messages[0]["role"] != "system" or messages[-1]["role"] != "user":
        raise ValueError("对话必须以 system 开始、以 user 结束")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须为正")
    device = next(model.parameters()).device
    generation_config = model.generation_config
    eos_ids = eos_token_ids(generation_config)
    options = sampling_options(generation_config)
    ending = assistant_end_tokens(tokenizer)
    if session.cache is None:
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]
        if prompt_len + len(ending) >= model.config.max_position_embeddings:
            raise ValueError("对话已超过模型上下文长度，请输入 /clear 开始新对话")
        positions = torch.arange(prompt_len, device=device)
        prefill = model(
            input_ids=input_ids, past_key_values=DynamicCache(), use_cache=True,
            output_attentions=PREFILL_KEEP_RATIO < 1.0,
            position_ids=positions.unsqueeze(0), cache_position=positions)
        session.cache = build_phasekv_cache(prefill, model, prompt_len)
        session.token_ids = input_ids[0].tolist()
        # clone 避免最后位置的视图继续持有整个 [B,prompt_len,vocab] logits。
        logits = prefill.logits[:, -1, :].clone()
        del prefill
    else:
        new_ids = new_user_prompt_tokens(tokenizer, messages[0], messages[-1])
        if session.cache.true_seq_length + len(new_ids) + len(ending) >= model.config.max_position_embeddings:
            raise ValueError("对话已超过模型上下文长度，请输入 /clear 开始新对话")
        logits = append_prompt_chunk(model, session.cache, new_ids)
        session.token_ids.extend(new_ids)

    cache = session.cache
    remaining = model.config.max_position_embeddings - cache.true_seq_length - len(ending)
    if remaining <= 0:
        raise ValueError("对话已超过模型上下文长度，请输入 /clear 开始新对话")
    max_new_tokens = min(max_new_tokens, remaining)
    seen_token_ids = torch.tensor([session.token_ids], device=device)
    next_token = sample_token(logits, seen_token_ids=seen_token_ids,
                              stage="prefill", **options)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    streamer.put(torch.tensor([session.token_ids], dtype=torch.long))
    generated_ids: list[int] = []
    for step in range(max_new_tokens):
        token_id = next_token.item()
        if token_id in eos_ids:
            break
        generated_ids.append(token_id)
        streamer.put(next_token.cpu())
        # 每个输出 token 都立即前向写入缓存，包括达到生成上限的最后一个。
        # 所以下轮可以直接追加用户输入，无需另设“补写最后一个 token”分支。
        logits = decode_token(model, cache, next_token)
        seen_token_ids = torch.cat((seen_token_ids, next_token.view(1, 1)), dim=1)
        if step == max_new_tokens - 1:
            break
        next_token = sample_token(
            logits, seen_token_ids=seen_token_ids, **options)
    streamer.end()
    # 模板结束符走新增输入接口，只打包该块，不清空生成热窗口。
    append_prompt_chunk(model, cache, ending)
    session.token_ids.extend([*generated_ids, *ending])
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), cache.physical_nbytes()


def chat_loop(model, tokenizer, *, max_new_tokens: int = MAX_NEW_TOKENS) -> None:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    session = ChatSession()
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
            session = ChatSession()
            print("对话历史已清空。")
            continue
        if not user_text:
            continue
        messages.append({"role": "user", "content": user_text})
        print("模型：", end="", flush=True)
        reply, physical_bytes = generate_reply(
            model, tokenizer, messages, session, max_new_tokens=max_new_tokens)
        messages.append({"role": "assistant", "content": reply})
        print(f"[PhaseKV 物理 KV 字节: {physical_bytes:,}]")


def check_cuda() -> str:
    """检查当前进程能否执行 CUDA；失败不等同于机器没有显卡或驱动损坏。"""
    if torch.version.cuda is None:
        raise RuntimeError(f"当前 Python 的 PyTorch 不含 CUDA：{sys.executable}")
    try:
        torch.cuda.init()
        torch.ones(1, device="cuda").add_(1)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise RuntimeError(
            f"当前进程无法使用 CUDA：{exc}\n"
            f"Python：{sys.executable}\n"
            "请在本机终端检查 nvidia-smi；若本机正常而隔离环境失败，"
            "需要让运行环境访问 GPU 设备，而不是重装驱动。"
        ) from exc
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("此对话入口使用 BF16 模型，当前 GPU 不支持；不会自动切换为 FP16")
    return torch.cuda.get_device_name()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Qwen2 PhaseKV 多轮对话（CUDA、batch=1）")
    parser.add_argument("--model", default=MODEL_ID, help="模型目录或 Hugging Face 模型 ID")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--local-files-only", action="store_true", help="仅使用本地权重，不访问网络")
    parser.add_argument("--check-cuda", action="store_true", help="只检查 GPU，不加载模型")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens 必须为正")
    try:
        gpu = check_cuda()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"GPU：{gpu}；PyTorch：{torch.__version__}；CUDA runtime：{torch.version.cuda}")
    if args.check_cuda:
        return
    print(f"正在加载 {args.model} …", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=args.local_files_only).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    install_qwen2_phasekv_attention(model)
    chat_loop(model, tokenizer, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
