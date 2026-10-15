# Qwen2 PhaseKV 冷层准确性消融

固定 Prompt 全保留、4-bit；对所有配置逐 token 输入相同的长回复前缀，随后贪心补全编号。
`no_cold` 保持生成段 FP16 热缓存；`keepall` 只量化冷层；`keep50` 还会筛掉一半冷层 token。
命中表示回答包含预设六位编号。前缀由实验脚本固定输入，不是模型自然生成。
本测试只覆盖同一轮长回复形成的冷层，不评估多轮会话缓存复用。

| 配置 | 命中 | 相对 prompt4_no_cold 的 token 一致率 | 回答前冷层 token 数 |
|---|---:|---:|---:|
| full | 3/3 | 0.905 | — |
| prompt4_no_cold | 3/3 | 1.000 | 0 |
| prompt4_cold8_keepall | 3/3 | 1.000 | 192 |
| prompt4_cold4_keepall | 3/3 | 1.000 | 192 |
| prompt4_cold4_keep50 | 3/3 | 1.000 | 96 |

## 逐题回答

| 题目 | 标准答案 | full | prompt4_no_cold | prompt4_cold8_keepall | prompt4_cold4_keepall | prompt4_cold4_keep50 |
|---|---:|---:|---:|---:|---:|---:|
| record_a | 739284 | 739284 | 739284。 | 739284。 | 739284。 | 739284。 |
| record_b | 815603 | 815603。 | 815603。 | 815603。 | 815603。 | 815603。 |
| record_c | 224906 | 224906 | 224906。 | 224906。 | 224906。 | 224906。 |

## 自然长回复

使用内置 `long_followup`，每种配置独立贪心生成；此题没有标准答案，token 一致率只表示输出是否分叉。

| 配置 | 生成 token | 冷层 token | 相对 no_cold 首次分叉 token | token 一致率 |
|---|---:|---:|---:|---:|
| full | 128 | — | 13 | 0.297 |
| prompt4_no_cold | 114 | 0 | — | 1.000 |
| prompt4_cold8_keepall | 114 | 80 | — | 1.000 |
| prompt4_cold4_keepall | 114 | 80 | — | 1.000 |
| prompt4_cold4_keep50 | 122 | 40 | 77 | 0.631 |
