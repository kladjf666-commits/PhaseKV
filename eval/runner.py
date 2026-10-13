"""基准编排 (QA / CH-3 维护): 串联模型加载、各方法评测与指标汇总。

对一组压缩方法在同一模型 / 同一 prompt 下运行:
  1. 生成任务 -> KV 显存节省、峰值显存、TTFT/TPOT/decode 吞吐、对 full-cache 输出一致率。
  2. (可选) 流式困惑度 -> 精度。
显存节省的分母为"未压缩全量缓存"的解析字节数 (按模型维度 + 总 token 数计算), 更准确。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from . import metrics, runtime
from .compressors import build_compressor


@dataclass
class BenchmarkConfig:
    model_name: str
    methods: List[dict]                      # 每项: {"method": name, ...params}
    prompt: str
    max_new_tokens: int = 64
    dtype: str = "float16"
    device: str = "cuda"
    greedy: bool = True
    seed: int = 1234
    ppl_text: Optional[str] = None           # 非空则跑流式困惑度
    ppl_chunk_size: int = 512
    ppl_max_chars: int = 20000
    extra_meta: dict = field(default_factory=dict)


def _kv_dims(model_config):
    num_layers = getattr(model_config, "num_hidden_layers")
    num_heads = getattr(model_config, "num_attention_heads")
    num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        head_dim = model_config.hidden_size // num_heads
    return num_layers, num_kv_heads, head_dim


def _needs_attention(methods: List[dict]) -> bool:
    for spec in methods:
        params = {k: v for k, v in spec.items() if k != "method"}
        if build_compressor(spec["method"], **params).requires_attention:
            return True
    return False


def run_benchmark(cfg: BenchmarkConfig) -> dict:
    attn_impl = "eager" if _needs_attention(cfg.methods) else "sdpa"
    model, tokenizer = runtime.load_model_and_tokenizer(
        cfg.model_name, dtype=cfg.dtype, device=cfg.device, attn_implementation=attn_impl)

    num_layers, num_kv_heads, head_dim = _kv_dims(model.config)
    input_ids = tokenizer(cfg.prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    ppl_ids = None
    if cfg.ppl_text:
        ppl_ids = tokenizer(cfg.ppl_text[:cfg.ppl_max_chars], return_tensors="pt").input_ids

    # full-cache 参考输出 (用于 token 一致率)
    full_gen = runtime.generate_with_compression(
        model, tokenizer, input_ids, build_compressor("full"),
        max_new_tokens=cfg.max_new_tokens, greedy=cfg.greedy, seed=cfg.seed)
    ref_ids = full_gen["generated_ids"]

    rows = []
    for spec in cfg.methods:
        params = {k: v for k, v in spec.items() if k != "method"}
        compressor = build_compressor(spec["method"], **params)
        gen = runtime.generate_with_compression(
            model, tokenizer, input_ids, compressor,
            max_new_tokens=cfg.max_new_tokens, greedy=cfg.greedy, seed=cfg.seed)

        total_tokens = prompt_len + gen["latency"].num_generated
        full_ref_bytes = metrics.kv_cache_dtype_bytes(
            num_layers, num_kv_heads, head_dim, total_tokens, dtype_bytes=2)
        method_bytes = gen["final_cache_bytes"]

        row = {
            "method": compressor.name,
            "kv_saving_pct": metrics.memory_saving_ratio(method_bytes, full_ref_bytes) * 100.0,
            "compression_ratio": metrics.compression_ratio(method_bytes, full_ref_bytes),
            "peak_gpu_mb": gen["peak_gpu_bytes"] / 1e6,
            "ttft_ms": gen["latency"].ttft_s * 1000.0,
            "tpot_ms": gen["latency"].tpot_s * 1000.0,
            "decode_tps": gen["latency"].decode_throughput,
            "token_match_vs_full": metrics.token_match_rate(gen["generated_ids"], ref_ids),
            "perplexity": None,
            "final_cache_bytes": method_bytes,
            "full_ref_cache_bytes": full_ref_bytes,
            "num_generated": gen["latency"].num_generated,
            "text": gen["text"],
        }

        if ppl_ids is not None:
            ppl = runtime.streaming_perplexity(
                model, build_compressor(spec["method"], **params), ppl_ids,
                chunk_size=cfg.ppl_chunk_size)
            row["perplexity"] = ppl["perplexity"]
            row["ppl_total_tokens"] = ppl["total_tokens"]

        rows.append(row)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    meta = {
        "model": cfg.model_name,
        "dtype": cfg.dtype,
        "attn_implementation": attn_impl,
        "prompt_len": prompt_len,
        "max_new_tokens": cfg.max_new_tokens,
        "kv_dims": {"num_layers": num_layers, "num_kv_heads": num_kv_heads, "head_dim": head_dim},
        "device": cfg.device,
        "greedy": cfg.greedy,
        **cfg.extra_meta,
    }
    return {"rows": rows, "meta": meta}
