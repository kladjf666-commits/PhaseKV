"""Launch planning for the KIVI-layout PhaseKV decode kernel (no CUDA required)."""
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class DecodeLaunchConfig:
    num_splits: int
    split_len: int
    block_n: int
    num_warps: int


def _ceildiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


@lru_cache(maxsize=2048)
def plan_kivi_decode(seq_len: int, num_heads: int, head_dim: int, sm_count: int,
                     num_splits: int | None = None, block_n: int | None = None,
                     num_warps: int | None = None) -> DecodeLaunchConfig:
    """Plan each packed segment from its retained length, head shape and GPU size.

    Keep auto tiles near 1024 elements to limit live dequantization registers.
    Start with four blocks per SM; long segments target <=16 tiles per split,
    up to 32 blocks per SM / 128 splits to bound the PyTorch reduction cost.
    These are bounded heuristics, not a runtime autotuning search. Explicit
    options override their respective heuristics independently.
    """
    if min(seq_len, num_heads, head_dim, sm_count) <= 0:
        raise ValueError("sequence, heads, head_dim and SM count must be positive")
    if head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two")
    if num_splits is not None and num_splits <= 0:
        raise ValueError("num_splits must be positive")
    if block_n is not None and (block_n <= 0 or block_n & (block_n - 1)):
        raise ValueError("block_n must be a positive power of two")
    if num_warps is not None and num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be one of 1, 2, 4, 8, 16, 32")

    if block_n is None:
        max_tile = max(1, min(32, 1024 // head_dim, seq_len))
        block_n = 1 << (max_tile.bit_length() - 1)
    if num_warps is None:
        # Explicit large tiles also get a larger thread block unless overridden.
        num_warps = min(32, max(4, _next_power_of_two(_ceildiv(block_n * head_dim, 256))))

    if num_splits is None:
        tiles = _ceildiv(seq_len, block_n)
        base = _ceildiv(4 * sm_count, num_heads)
        limit = min(128, _next_power_of_two(_ceildiv(32 * sm_count, num_heads)))
        requested = min(tiles, limit, _next_power_of_two(max(base, _ceildiv(tiles, 16))))
        # Tile-aligned boundaries avoid duplicated tail work in every split.
        split_len = _ceildiv(tiles, requested) * block_n
    else:
        # Preserve the manual split-length convention used by existing callers.
        split_len = _ceildiv(seq_len, min(seq_len, num_splits))
    # Do not launch empty trailing splits (also covers uneven manual splits).
    splits = _ceildiv(seq_len, split_len)
    return DecodeLaunchConfig(splits, split_len, block_n, num_warps)
