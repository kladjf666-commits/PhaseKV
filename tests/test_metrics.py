"""指标计算单测 (QA / CH-3)。"""
import math

import pytest
import torch

from eval import metrics


def test_kv_cache_dtype_bytes_formula():
    # 2(K,V) * batch1 * layers2 * kvheads2 * seq10 * dim4 * 2B = 640
    assert metrics.kv_cache_dtype_bytes(2, 2, 4, 10, dtype_bytes=2, batch=1) == 640


def test_memory_saving_and_compression_ratio():
    assert metrics.memory_saving_ratio(250, 1000) == pytest.approx(0.75)
    assert metrics.compression_ratio(250, 1000) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        metrics.memory_saving_ratio(1, 0)


def test_latency_stats():
    s = metrics.LatencyStats(ttft_s=0.1, decode_step_times_s=[0.02, 0.04, 0.06], num_generated=4)
    assert s.tpot_s == pytest.approx(0.04)
    assert s.decode_throughput == pytest.approx(3 / 0.12)
    assert s.total_time_s == pytest.approx(0.22)
    assert s.end_to_end_throughput == pytest.approx(4 / 0.22)
    assert s.percentile_decode_ms(50) == pytest.approx(40.0)
    assert s.percentile_decode_ms(0) == pytest.approx(20.0)
    assert s.percentile_decode_ms(100) == pytest.approx(60.0)


def test_perplexity_uniform():
    # 均匀分布 (vocab=4) 下每 token NLL = ln4, ppl = 4
    logits = torch.zeros(1, 3, 4)
    labels = torch.tensor([[0, 1, 2]])
    acc = metrics.PerplexityAccumulator()
    acc.update(logits, labels)
    assert acc.total_tokens == 3
    assert acc.perplexity == pytest.approx(4.0, rel=1e-5)
    assert acc.mean_nll == pytest.approx(math.log(4), rel=1e-5)


def test_perplexity_ignore_index():
    logits = torch.zeros(1, 2, 4)
    labels = torch.tensor([[-100, 1]])
    acc = metrics.PerplexityAccumulator()
    acc.update(logits, labels)
    assert acc.total_tokens == 1


def test_perplexity_shape_mismatch_raises():
    acc = metrics.PerplexityAccumulator()
    with pytest.raises(ValueError):
        acc.update(torch.zeros(1, 3, 4), torch.tensor([[0, 1]]))


def test_token_match_and_divergence():
    assert metrics.token_match_rate([1, 2, 3], [1, 2, 9]) == pytest.approx(2 / 3)
    assert metrics.first_divergence_index([1, 2, 3], [1, 2, 3]) is None
    assert metrics.first_divergence_index([1, 2, 3], [1, 9, 3]) == 1
    assert metrics.first_divergence_index([1, 2], [1, 2, 3]) == 2


def test_exact_substring_score():
    assert metrics.exact_substring_score("The number is 12345.", ["12345"]) == 1.0
    assert metrics.exact_substring_score("no match here", ["12345"]) == 0.0
