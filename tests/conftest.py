"""测试公共夹具 (QA / CH-3)。"""
import os
import sys

import torch

# 使 ``import eval`` 可用 (kvcomp/ 作为包根)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def make_fake_cache(num_layers: int = 2, num_kv_heads: int = 2, seq_len: int = 8,
                    head_dim: int = 4, seed: int = 0):
    """用真实 transformers DynamicCache 构造小型缓存 (CPU)。"""
    from transformers.cache_utils import DynamicCache

    gen = torch.Generator().manual_seed(seed)
    cache = DynamicCache()
    for layer in range(num_layers):
        k = torch.randn(1, num_kv_heads, seq_len, head_dim, generator=gen)
        v = torch.randn(1, num_kv_heads, seq_len, head_dim, generator=gen)
        cache.update(k, v, layer)
    return cache


def make_fake_attentions(num_layers: int = 2, num_heads: int = 2, q_len: int = 8,
                         kv_len: int = 8, seed: int = 1):
    """构造合法的注意力张量元组 (行和为 1)。"""
    gen = torch.Generator().manual_seed(seed)
    attentions = []
    for _ in range(num_layers):
        a = torch.rand(1, num_heads, q_len, kv_len, generator=gen)
        a = a / a.sum(dim=-1, keepdim=True)
        attentions.append(a)
    return tuple(attentions)
