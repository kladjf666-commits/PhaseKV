"""KV Cache 压缩方法的统一接口契约 (QA / CH-3 维护)。

任何 KV Cache 压缩方法 (架构 src/ 或后端 baselines/ 实现) 都必须继承
``KVCompressor`` 并实现其抽象方法, 才能被本评估框架 (eval/) 装载、对比与验收。

设计目标:
1. 方法无关: prefill 阶段与 decode 阶段可使用完全不同的压缩策略 (用户核心 idea)。
2. 可度量: 压缩器需如实汇报其 KV Cache 的"逻辑显存占用", 量化类方法据 bit 数折算。
3. 与 HuggingFace ``transformers`` 的 ``Cache`` (DynamicCache) 协议对齐, 直接操作
   ``key_cache`` / ``value_cache`` 张量, 不引入额外的影子缓存。

约定 (与 transformers 4.44.x 对齐):
- ``cache`` 为 ``transformers.cache_utils.Cache`` 实例; ``cache.key_cache[i]`` /
  ``cache.value_cache[i]`` 形状为 ``[batch, num_kv_heads, seq_len, head_dim]``。
- 压缩器对 cache 的修改应"就地或返回新 cache 二选一", 统一以返回值为准。
- 涉及 token 驱逐 (eviction) 时, 物理缓存长度会小于真实已见 token 数, 评估运行时
  ``runtime.py`` 负责用真实绝对位置 (true position) 传入 ``position_ids`` 以保证 RoPE
  正确; 压缩器只需保证保留的 K/V 张量自身一致。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CompressionConfig:
    """压缩方法的统一配置。

    不同方法各取所需; 框架据此构造压缩器并写入实验记录, 保证可复现。

    Attributes:
        name: 方法名 (用于报告与注册表)。
        prefill_keep_ratio: prefill 结束后保留的 KV 比例 (0,1]; None 表示不在 prefill 压缩。
        decode_keep_ratio: decode 过程中维持的 KV 比例 (0,1]; None 表示 decode 不驱逐。
        sink_tokens: 始终保留的起始 (attention sink) token 数。
        recent_window: 始终保留的最近窗口长度 (token 数)。
        bits: 量化方法的目标位宽 (如 8/4/2); None 表示非量化。
        max_cache_tokens: 物理缓存允许的最大 token 数 (None 表示无硬上限)。
        seed: 方法内部随机性的种子 (保证可复现)。
        extra: 方法私有的附加超参。
    """

    name: str = "base"
    prefill_keep_ratio: Optional[float] = None
    decode_keep_ratio: Optional[float] = None
    sink_tokens: int = 0
    recent_window: Optional[int] = None
    bits: Optional[int] = None
    max_cache_tokens: Optional[int] = None
    seed: int = 1234
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        for field_name in ("prefill_keep_ratio", "decode_keep_ratio"):
            value = getattr(self, field_name)
            if value is not None and not (0.0 < value <= 1.0):
                raise ValueError(f"{field_name} 必须落在 (0, 1], 实际为 {value}")
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens 不能为负")
        if self.recent_window is not None and self.recent_window <= 0:
            raise ValueError("recent_window 必须为正整数")
        if self.bits is not None and self.bits not in (2, 3, 4, 8, 16):
            raise ValueError(f"bits 仅支持 2/3/4/8/16, 实际为 {self.bits}")
        if self.max_cache_tokens is not None and self.max_cache_tokens <= 0:
            raise ValueError("max_cache_tokens 必须为正整数")


class KVCompressor(abc.ABC):#一份供其他类继承的规范，通常不应该直接拿来创建对象
    """KV Cache 压缩方法的抽象基类 (评估框架的集成契约)。

    生命周期 (由 runtime.py 调用):
        1. ``reset(model_config)``       -- 每条样本生成前调用, 清空内部状态。
        2. ``on_prefill(cache, ...)``    -- prefill 前向后调用一次, 压缩 prompt KV。
        3. 循环: ``on_decode_step(cache, ...)`` -- 每个 decode step 后调用, 维持缓存。
    度量:
        - ``cache_memory_bytes(cache)``  -- 汇报当前 KV Cache 的逻辑显存占用 (字节)。
    """

    #: 方法是否需要注意力分数 (eviction 类如 H2O/SnapKV 需要; runtime 据此开启
    #: output_attentions, 注意会禁用 SDPA/Flash 并显著增加显存, 仅在必要时开启)。
    requires_attention: bool = False

    #: 是否物理改变缓存张量形状 (eviction=True; 纯量化=False, 仅改变逻辑位宽)。
    physically_evicts: bool = True

    def __init__(self, config: CompressionConfig):
        config.validate()
        self.config = config
        self.name = config.name

    def reset(self, model_config: Any) -> None:
        """每条样本开始前重置压缩器内部状态 (默认无状态)。"""

    @abc.abstractmethod
    def on_prefill(
        self,
        cache: Any,
        *,
        attentions: Optional[tuple] = None,
        prompt_len: int,
        model_config: Any,
    ) -> Any:
        """prefill 前向结束后压缩 prompt 的 KV Cache。

        Args:
            cache: prefill 后的 transformers Cache (含完整 prompt 的 K/V)。
            attentions: 若 ``requires_attention`` 为真, 为各层注意力张量元组,
                每个形状 ``[batch, num_heads, q_len, kv_len]``; 否则为 None。
            prompt_len: prompt 的真实 token 数。
            model_config: 模型 config (含 num_hidden_layers / num_key_value_heads 等)。

        Returns:
            压缩后的 Cache (可与入参为同一对象)。
        """

    @abc.abstractmethod
    def on_decode_step(
        self,
        cache: Any,
        *,
        attentions: Optional[tuple] = None,
        step: int,
        model_config: Any,
    ) -> Any:
        """每个 decode step 前向结束后维持/压缩增长的 KV Cache。

        Args:
            cache: 含新生成 token 的 Cache。
            attentions: 同 ``on_prefill``。
            step: 当前 decode 步序号 (从 0 开始)。
            model_config: 模型 config。

        Returns:
            压缩后的 Cache。
        """

    def cache_memory_bytes(self, cache: Any) -> int:
        """汇报当前 KV Cache 的逻辑显存占用 (字节)。

        默认实现按物理张量真实字节累加 (适用于 eviction 类方法)。量化类方法应覆盖
        本方法, 按目标位宽折算 (物理张量可能仍是 fp16, 但逻辑占用应反映压缩后位宽)。
        """
        total = 0
        key_cache = getattr(cache, "key_cache", None)
        value_cache = getattr(cache, "value_cache", None)
        if key_cache is None or value_cache is None:
            raise TypeError("cache 必须暴露 key_cache / value_cache (transformers Cache 协议)")
        for tensor in list(key_cache) + list(value_cache):
            if tensor is None:
                continue
            total += tensor.numel() * tensor.element_size()
        return total

    def describe(self) -> dict:
        """返回用于报告/复现的方法描述。"""
        return {
            "name": self.name,
            "class": type(self).__name__,
            "requires_attention": self.requires_attention,
            "physically_evicts": self.physically_evicts,
            "config": {
                k: v for k, v in vars(self.config).items() if v is not None and v != {}
            },
        }
