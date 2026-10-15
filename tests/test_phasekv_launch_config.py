"""Coverage of automatic split boundaries and actual packed attention outputs."""
import unittest

import torch

from src.phasekv.kernels.launch_config import plan_kivi_decode


class TestLaunchPlanning(unittest.TestCase):
    def test_boundaries_cover_segment_without_empty_splits(self):
        for length in (1, 7, 31, 127, 128, 129, 511, 512, 8193, 28672):
            for heads, dim, sms in ((12, 128, 46), (4, 8, 1), (32, 256, 132)):
                c = plan_kivi_decode(length, heads, dim, sms)
                self.assertGreaterEqual(c.num_splits, 1)
                self.assertLessEqual(c.num_splits, 128)
                self.assertLess((c.num_splits - 1) * c.split_len, length)
                self.assertGreaterEqual(c.num_splits * c.split_len, length)
                self.assertEqual(c.split_len % c.block_n, 0)
                self.assertLessEqual(c.block_n * dim, 1024)

    def test_length_and_hardware_change_parallelism(self):
        short = plan_kivi_decode(128, 12, 128, 46)
        medium = plan_kivi_decode(8192, 12, 128, 46)
        long = plan_kivi_decode(28672, 12, 128, 46)
        self.assertLess(short.num_splits, medium.num_splits)
        self.assertLess(medium.num_splits, long.num_splits)
        self.assertLess(plan_kivi_decode(512, 12, 128, 16).num_splits,
                        plan_kivi_decode(512, 12, 128, 80).num_splits)
        self.assertLess(plan_kivi_decode(512, 12, 256, 46).block_n,
                        plan_kivi_decode(512, 12, 64, 46).block_n)

    def test_overrides_and_uneven_splits(self):
        old = plan_kivi_decode(8192, 12, 128, 46, 8, 64, 4)
        self.assertEqual((old.num_splits, old.split_len, old.block_n, old.num_warps),
                         (8, 1024, 64, 4))
        uneven = plan_kivi_decode(10, 12, 128, 46, num_splits=8)
        self.assertEqual((uneven.num_splits, uneven.split_len), (5, 2))
        self.assertEqual(plan_kivi_decode(512, 12, 128, 46, block_n=16).block_n, 16)
        self.assertEqual(plan_kivi_decode(512, 12, 128, 46, num_warps=8).num_warps, 8)
        self.assertEqual(plan_kivi_decode(1, 12, 128, 46, num_splits=100).num_splits, 1)

    def test_invalid_parameters(self):
        for kwargs in ({'num_splits': 0}, {'block_n': 0}, {'block_n': 3}, {'num_warps': 3}):
            with self.assertRaises(ValueError):
                plan_kivi_decode(128, 12, 128, 46, **kwargs)
        with self.assertRaises(ValueError):
            plan_kivi_decode(0, 12, 128, 46)


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class TestPackedAutoLaunch(unittest.TestCase):
    @torch.inference_mode()
    def test_auto_and_manual_match_dense_quantized_reference(self):
        from src.phasekv.kernels.attention_triton import _kivi_layer_partial
        from src.phasekv.kernels.packing import quantize_pack, unpack_low_bits

        def dense_float(pt):
            codes = unpack_low_bits(pt.packed, pt.bits, pt.axis_len).float()
            groups = torch.arange(pt.axis_len, device='cuda') // pt.group_size
            return (codes * pt.scale.float()[..., groups] +
                    pt.mins.float()[..., groups]).movedim(-1, pt.axis)

        torch.manual_seed(42)
        for length, heads, dim, kb, vb in ((1, 4, 8, 2, 4), (10, 4, 64, 8, 8),
                                         (127, 12, 128, 4, 4), (513, 12, 128, 2, 2),
                                         (8193, 12, 128, 4, 4), (28672, 12, 128, 4, 4),
                                         (129, 4, 256, 8, 4)):
            with self.subTest(length=length, dim=dim, bits=(kb, vb)):
                q = torch.randn(heads, dim, device='cuda', dtype=torch.float16)
                k = quantize_pack(torch.randn(2, length, dim, device='cuda', dtype=torch.float16), kb, 1, 32)
                v = quantize_pack(torch.randn(2, length, dim, device='cuda', dtype=torch.float16), vb, 2, 32)
                dk = dense_float(k).repeat_interleave(heads // 2, dim=0)
                dv = dense_float(v).repeat_interleave(heads // 2, dim=0)
                probs = torch.softmax(torch.einsum('hd,hsd->hs', q.float(), dk) / dim**0.5, dim=-1)
                expected = torch.einsum('hs,hsd->hd', probs, dv)
                for options in ({}, {'num_splits': 8, 'block_n': 64, 'num_warps': 4},
                                {'num_splits': 3}):
                    _, total, acc = _kivi_layer_partial(q, k, v, heads, **options)
                    torch.testing.assert_close(acc / total[:, None], expected, atol=3e-4, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
