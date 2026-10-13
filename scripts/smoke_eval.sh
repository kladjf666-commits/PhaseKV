#!/usr/bin/env bash
# KV 压缩评估 · 可复现冒烟脚本 (CH-3)。在 release/kvcomp/ 下运行: bash scripts/smoke_eval.sh
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
MODEL="${KVCOMP_SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

cd "$(dirname "$0")/.."

echo "===== 单元测试 (纯逻辑, 无需 GPU) ====="
python -m pytest tests/test_metrics.py tests/test_interface.py tests/test_compressors.py -q

echo "===== full preset (sdpa) + 困惑度 ====="
python -m eval.run_benchmark --model "$MODEL" --preset full \
    --prompt-tokens 512 --max-new-tokens 32 --ppl-source builtin \
    --out results/smoke_full

echo "===== attention preset (eager) ====="
python -m eval.run_benchmark --model "$MODEL" --preset attention \
    --prompt-tokens 512 --max-new-tokens 32 --ppl-source none \
    --out results/smoke_attention

echo "===== 端到端测试 (真实模型) ====="
python -m pytest tests/test_smoke_e2e.py -q

echo "全部冒烟通过。报告见 results/smoke_*.{md,csv,json}"
