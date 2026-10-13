"""带基线注入的评估入口 (CH-5)。

eval/run_benchmark 默认只注册 eval 自带的参考方法 (full/recent/streaming/kvquant/h2o/
hetero)。本脚本在调用它之前先 ``import baselines`` 注入竞品基线 (snapkv/kivi/scope/
pyramidkv); 若 ``src/phasekv`` 已就绪也一并注入, 从而让 ``--methods-json`` 能引用全部
团队方法 (用于 M1 精度 Pareto 对比)。命令行参数与 eval.run_benchmark 完全一致。

在 release/kvcomp/ 下运行, 例:
  HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf-cache \
  python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct \
      --methods-json '[{"method":"full"},{"method":"snapkv","prefill_keep_ratio":0.3,"obs_window":16}]' \
      --max-new-tokens 32 --ppl-source builtin --out results/baselines_smoke
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import baselines  # noqa: E402,F401  导入即注入 snapkv/kivi/scope(/pyramidkv)

try:
    import src.phasekv  # noqa: E402,F401  架构侧就绪则一并注入 phasekv
    _PHASEKV = True
except Exception as exc:  # 架构侧未完成时只注入基线, 不阻塞 (非算法回退, 仅插件发现)
    _PHASEKV = False
    print(f"[run_eval] 未注入 src.phasekv: {exc.__class__.__name__}: {exc}", file=sys.stderr)

from eval.compressors import available  # noqa: E402
from eval.run_benchmark import main  # noqa: E402

if __name__ == "__main__":
    print(f"[run_eval] phasekv 注入={_PHASEKV}; 可用方法={available()}", file=sys.stderr)
    main()
