"""评估数据集加载 (QA / CH-3 维护)。

提供三类评估数据:
  - 困惑度: WikiText-2 (HF datasets 下载) + 内置离线样本 (无需联网, 供 smoke)。
  - 长上下文检索: 合成 needle-in-a-haystack (确定性生成, 无需联网)。
  - 长上下文任务: LongBench 子任务 (HF datasets 下载)。

联网下载统一建议 ``HF_ENDPOINT=https://hf-mirror.com`` 并缓存到 /root/autodl-tmp/hf-cache。
"""
from __future__ import annotations

import random
from typing import List, Optional


BUILTIN_SAMPLE_TEXT = (
    "Large language model inference is dominated by two phases. During the prefill phase, "
    "the model processes the entire input prompt in parallel and populates the key-value "
    "cache for every layer and attention head. During the decode phase, the model generates "
    "tokens one at a time, and each new token attends to the growing key-value cache. Because "
    "the decode phase is memory bound, the size and bandwidth of the key-value cache directly "
    "determine throughput and latency. Compressing the key-value cache is therefore a central "
    "problem for efficient serving of large language models on modern accelerators. A promising "
    "direction is to apply different compression strategies to the prefill cache and the decode "
    "cache, because the static prompt and the dynamically generated tokens exhibit very different "
    "attention patterns. The prompt tokens are fixed and can be analyzed once, while the decode "
    "tokens arrive incrementally and benefit from cheap streaming policies. This document is used "
    "as a small offline corpus so that the perplexity smoke test can run without any network access. "
) * 6


def load_wikitext(split: str = "test", n_chars: Optional[int] = None) -> str:
    """加载 WikiText-2 (raw) 指定 split 的拼接文本。"""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    if n_chars is not None:
        text = text[:n_chars]
    return text


def make_needle_haystack(num_samples: int = 5, *, haystack_sentences: int = 200,
                         seed: int = 1234) -> List[dict]:
    """确定性生成 needle-in-a-haystack 样本 (无需联网)。

    每条样本: 在大段无关文本 (haystack) 中嵌入一句含"魔法数字"的 needle, 并提问该数字。
    返回 [{context, question, answers, depth_ratio}]。
    """
    rng = random.Random(seed)
    filler = (
        "The grass is green and the sky is blue on an ordinary calm afternoon. "
        "People walk slowly through the quiet park while the gentle wind moves the leaves. "
    )
    samples = []
    for i in range(num_samples):
        magic = rng.randint(100000, 999999)
        depth = (i + 1) / (num_samples + 1)
        insert_at = max(1, int(haystack_sentences * depth))
        needle = f"The special magic number for session {i} is {magic}. "
        sentences = [filler for _ in range(haystack_sentences)]
        sentences.insert(insert_at, needle)
        context = "".join(sentences)
        question = f"What is the special magic number for session {i}? Answer with only the number."
        samples.append({
            "context": context,
            "question": question,
            "answers": [str(magic)],
            "depth_ratio": round(depth, 3),
        })
    return samples


def load_longbench(task: str = "narrativeqa", n_samples: int = 20) -> List[dict]:
    """加载 LongBench 单任务 (HF: THUDM/LongBench)。

    返回 [{prompt, answers, length, dataset}]; prompt 已套用任务模板。
    """
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench", task, split="test")
    prompt_templates = _longbench_templates()
    template = prompt_templates.get(task, "{context}\n\n{input}\n\nAnswer:")
    out = []
    for i, ex in enumerate(ds):
        if i >= n_samples:
            break
        prompt = template.format(context=ex.get("context", ""), input=ex.get("input", ""))
        out.append({
            "prompt": prompt,
            "answers": ex.get("answers", []),
            "length": ex.get("length", len(prompt)),
            "dataset": task,
        })
    return out


def _longbench_templates() -> dict:
    return {
        "narrativeqa": (
            "You are given a story and a question. Answer the question concisely.\n\n"
            "Story: {context}\n\nQuestion: {input}\n\nAnswer:"),
        "qasper": (
            "You are given a scientific paper and a question.\n\n"
            "Paper: {context}\n\nQuestion: {input}\n\nAnswer:"),
        "triviaqa": (
            "Answer the question based on the given passages.\n\n"
            "Passages: {context}\n\nQuestion: {input}\n\nAnswer:"),
    }


def build_long_prompt_from_text(text: str, tokenizer, target_tokens: int) -> str:
    """把任意长文本裁剪到约 ``target_tokens`` 个 token 的 prompt (用于显存/时延扫描)。"""
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    if len(ids) <= target_tokens:
        repeat = (target_tokens // max(1, len(ids))) + 1
        ids = (ids * repeat)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)
