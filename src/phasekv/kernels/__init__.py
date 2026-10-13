"""PhaseKV 阶段 B: 单遍混精度融合注意力 kernel 及其数据地基 (CH-2 架构)。

阶段 A (src/phasekv 上层) 用 fake-quant 得到*真实精度 + 逻辑显存*, 物理张量仍是 fp16,
不报告 decode 加速。阶段 B 在此包内落地"真加速"的两块:

  - packing.py           真实低比特打包 (2/4/8 bit 塞入 uint8) + 紧凑 scale/min, 使
                         prompt/冷层以压缩形态存于 HBM (阶段 A 逻辑显存 => 阶段 B 物理显存)。
  - attention_triton.py  单遍混精融合 decode 注意力 kernel: 一次读 [packed prompt] +
                         [packed 冷层] + [FP16 热层], 在片上 (寄存器/共享内存) 反量化后做
                         online-softmax 融合, 不把整份 KV 反量化回 HBM, 不做 dense 回退。

红线: 加速比只来自阶段 B 真实 kernel 的端到端测量, 严禁模型估算冒充; 打包无损于量化栅格。
"""
from __future__ import annotations

from . import packing, reference

__all__ = ["packing", "reference"]
