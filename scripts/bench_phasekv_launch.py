"""Compare old and automatic launch parameters on the real KIVI packed kernel.

CUDA graph timing excludes Python launch overhead and reuses resident data;
this microbenchmark does not measure model tokens/s.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton
from triton.testing import do_bench_cudagraph

from src.phasekv.kernels import attention_triton as attention
from src.phasekv.kernels.launch_config import plan_kivi_decode
from src.phasekv.kernels.packing import quantize_pack


class CaptureLaunch:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def call(*args, **kwargs):
            self.replay = lambda: launch(*args, **kwargs)
            self.compiled = self.replay()
            return self.compiled

        return call


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', type=int, nargs='+', default=[128, 512, 8192, 28672])
    parser.add_argument('--output', type=Path, default=Path('results/phasekv_auto_launch.json'))
    args = parser.parse_args()
    torch.manual_seed(42)
    props = torch.cuda.get_device_properties(0)
    payload = {'gpu': props.name, 'sm_count': props.multi_processor_count,
               'torch': torch.__version__, 'triton': triton.__version__,
               'measurement': 'CUDA graph, resident random 4-bit KV, single layer, Hq=12 Hkv=2 HD=128; excludes Python overhead',
               'results': []}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for length in args.lengths:
            q = torch.randn(12, 128, device='cuda', dtype=torch.float16)
            k = quantize_pack(torch.randn(2, length, 128, device='cuda', dtype=torch.float16), 4, 1, 32)
            v = quantize_pack(torch.randn(2, length, 128, device='cuda', dtype=torch.float16), 4, 2, 32)
            for mode, options in [('old', dict(num_splits=8, block_n=64, num_warps=4)), ('auto', {})]:
                config = plan_kivi_decode(length, 12, 128, props.multi_processor_count, **options)
                capture = CaptureLaunch(attention._decode_attn_kivi_split_kernel)
                with patch.object(attention, '_decode_attn_kivi_split_kernel', capture):
                    attention._kivi_layer_partial(q, k, v, 12, **options)
                kernel_ms = do_bench_cudagraph(capture.replay, rep=100)
                partial_ms = do_bench_cudagraph(lambda: attention._kivi_layer_partial(q, k, v, 12, **options), rep=100)
                row = dict(length=length, mode=mode, config=asdict(config), kernel_ms=kernel_ms,
                           partial_with_reduce_ms=partial_ms, registers=capture.compiled.n_regs,
                           spills=capture.compiled.n_spills)
                payload['results'].append(row)
                print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
