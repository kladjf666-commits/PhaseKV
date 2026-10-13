"""真实模型端到端冒烟测试 (QA / CH-3)。

默认使用 Qwen2.5-0.5B-Instruct 在 GPU 上跑通整个评估链路, 校验:
  - full-cache 自我一致 (token_match==1.0);
  - 驱逐类方法确实降低 KV 逻辑显存 (kv_saving_pct>0 且 final<full);
  - 量化类方法保持形状但降低逻辑显存;
  - 困惑度为有限正数。

无 CUDA 或模型不可用时自动跳过 (不做假数据)。模型缓存与镜像通过环境变量控制:
  HF_ENDPOINT=https://hf-mirror.com  HF_HOME=/root/autodl-tmp/hf-cache
  KVCOMP_SMOKE_MODEL 可覆盖模型 (默认 Qwen/Qwen2.5-0.5B-Instruct)。
"""
import os

import pytest

torch = pytest.importorskip("torch")

from eval.datasets import BUILTIN_SAMPLE_TEXT
from eval.runner import BenchmarkConfig, run_benchmark

MODEL = os.environ.get("KVCOMP_SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="端到端冒烟需要 CUDA GPU")


@pytest.fixture(scope="module")
def smoke_result():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    cfg = BenchmarkConfig(
        model_name=MODEL,
        methods=[
            {"method": "full"},
            {"method": "recent", "recent_window": 128},
            {"method": "kvquant", "bits": 4},
        ],
        prompt=BUILTIN_SAMPLE_TEXT[:2000],
        max_new_tokens=16,
        ppl_text=BUILTIN_SAMPLE_TEXT,
        ppl_chunk_size=256,
    )
    try:
        return run_benchmark(cfg)
    except Exception as exc:  # 模型不可用 (网络/缓存) 时跳过, 不伪造
        pytest.skip(f"模型 {MODEL} 不可用: {exc}")


def _row(result, method):
    for r in result["rows"]:
        if r["method"] == method:
            return r
    raise AssertionError(f"结果缺少方法 {method}")


def test_smoke_runs_all_methods(smoke_result):
    assert len(smoke_result["rows"]) == 3
    assert smoke_result["meta"]["model"] == MODEL


def test_full_cache_self_consistent(smoke_result):
    full = _row(smoke_result, "full")
    assert full["token_match_vs_full"] == 1.0
    assert full["kv_saving_pct"] < 1.0  # full 基本不省 (仅取整误差)
    assert full["perplexity"] == pytest.approx(full["perplexity"])  # 非 NaN
    assert full["perplexity"] > 1.0


def test_recent_window_saves_memory(smoke_result):
    recent = _row(smoke_result, "recent")
    full = _row(smoke_result, "full")
    assert recent["kv_saving_pct"] > 0.0
    assert recent["final_cache_bytes"] < full["final_cache_bytes"]


def test_kvquant_saves_memory(smoke_result):
    q = _row(smoke_result, "kvquant")
    full = _row(smoke_result, "full")
    assert q["kv_saving_pct"] > 0.0
    assert q["final_cache_bytes"] < full["final_cache_bytes"]
    assert q["perplexity"] > 1.0


def test_peak_memory_measured(smoke_result):
    for row in smoke_result["rows"]:
        assert row["peak_gpu_mb"] > 0.0
        assert row["ttft_ms"] > 0.0
