"""模型下载脚本 (CH-10 补齐, 供后端/全队复用)。

支持两种下载源 (国内优先魔塔):
  - modelscope (默认): 魔塔 ModelScope, 国内直连, 推荐。
  - hf: HuggingFace, 走 HF_ENDPOINT=https://hf-mirror.com 镜像。

缓存目录:
  - modelscope -> MODELSCOPE_CACHE=/root/autodl-tmp/modelscope-cache
  - hf         -> HF_HOME=/root/autodl-tmp/hf-cache

下载前做磁盘空间检查; 下载后打印团队记忆登记提醒 (共享 50G 盘)。

模型别名:
  0.5b   -> Qwen/Qwen2.5-0.5B-Instruct   (冒烟, ~1GB)
  7b     -> Qwen/Qwen2.5-7B-Instruct     (主力, fp16 ~15GB)
  llama8b-> meta-llama/Llama-3.1-8B-Instruct (泛化; 魔塔走非gated的
           LLM-Research/Meta-Llama-3.1-8B-Instruct 镜像, HF 源需 gated 授权)

默认忽略 original/*.pth 等重复的原始权重, 只拉 safetensors 省盘 (--include-original 关闭)。

在 release/kvcomp/ 下运行, 例:
  python scripts/download_models.py 7b                        # 魔塔 (默认)
  python scripts/download_models.py 7b --source modelscope
  python scripts/download_models.py 0.5b --source hf
  python scripts/download_models.py llama8b                   # 魔塔非gated泛化模型
  python scripts/download_models.py Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_HOME = "/root/autodl-tmp/hf-cache"
DEFAULT_MODELSCOPE_CACHE = "/root/autodl-tmp/modelscope-cache"

ALIASES = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
}

# 魔塔与 HF 的 model id 映射。Qwen 系列两边同步用同名; meta-llama 在魔塔的
# gated 命名空间需授权, 改指向魔塔上非 gated 的 LLM-Research 镜像仓库。
MODELSCOPE_IDS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
}

APPROX_GB = {
    "Qwen/Qwen2.5-0.5B-Instruct": 1.0,
    "Qwen/Qwen2.5-7B-Instruct": 15.5,
    "meta-llama/Llama-3.1-8B-Instruct": 16.5,
    "LLM-Research/Meta-Llama-3.1-8B-Instruct": 16.5,
}

# 部分仓库同时提供 safetensors 与 original/*.pth 原始权重 (体积翻倍且 transformers
# 不需要); 默认忽略以省盘。gguf 同理。可用 --include-original 关闭忽略。
DEFAULT_IGNORE_PATTERNS = ["original/*", "*.pth", "*.gguf"]


def _free_gb(path: str) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def _resolve(name: str) -> str:
    return ALIASES.get(name.lower(), name)


def _check_disk(cache_root: str, repo_id: str, margin_gb: float) -> None:
    need = APPROX_GB.get(repo_id, 5.0)
    free = _free_gb(cache_root)
    print(f"[download_models] {repo_id}: 预计 ~{need:.1f}GB, 当前可用 {free:.1f}GB "
          f"(缓存根={cache_root})", flush=True)
    if free < need + margin_gb:
        raise RuntimeError(
            f"磁盘余量不足: 需 ~{need:.1f}GB(+{margin_gb:.0f}GB 余量), 仅 {free:.1f}GB。")


def download_hf(repo_id: str, *, hf_home: str, ignore_patterns=None) -> str:
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    os.environ["HF_HOME"] = hf_home
    os.makedirs(hf_home, exist_ok=True)
    from huggingface_hub import snapshot_download

    print(f"[download_models] HF 源: {repo_id} (端点={os.environ['HF_ENDPOINT']}, "
          f"忽略={ignore_patterns})", flush=True)
    local = snapshot_download(repo_id=repo_id, repo_type="model",
                              ignore_patterns=ignore_patterns)
    print(f"[download_models] 完成: {repo_id} -> {local}", flush=True)
    return local


def download_modelscope(repo_id: str, *, cache_dir: str, ignore_patterns=None) -> str:
    os.environ["MODELSCOPE_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    ms_id = MODELSCOPE_IDS.get(repo_id, repo_id)
    from modelscope import snapshot_download

    print(f"[download_models] 魔塔源: {ms_id} (MODELSCOPE_CACHE={cache_dir}, "
          f"忽略={ignore_patterns})", flush=True)
    local = snapshot_download(ms_id, cache_dir=cache_dir,
                              ignore_patterns=ignore_patterns)
    print(f"[download_models] 完成: {ms_id} -> {local}", flush=True)
    return local


def download_one(repo_id: str, *, source: str, cache_root: str, margin_gb: float,
                 ignore_patterns=None) -> str:
    os.makedirs(cache_root, exist_ok=True)
    _check_disk(cache_root, repo_id, margin_gb)
    if source == "modelscope":
        if repo_id not in MODELSCOPE_IDS and repo_id.startswith("meta-llama/"):
            raise RuntimeError(f"{repo_id} 在魔塔需单独申请权限, 请用 --source hf")
        return download_modelscope(repo_id, cache_dir=cache_root,
                                   ignore_patterns=ignore_patterns)
    return download_hf(repo_id, hf_home=cache_root, ignore_patterns=ignore_patterns)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="模型下载 (魔塔/HF, 缓存到数据盘)")
    p.add_argument("models", nargs="+",
                   help="模型别名(0.5b/7b/llama8b)或完整 repo id")
    p.add_argument("--source", choices=["modelscope", "hf"], default="modelscope",
                   help="下载源: modelscope=魔塔(默认, 国内快), hf=hf-mirror")
    p.add_argument("--cache-dir", default=None,
                   help="覆盖默认缓存根目录")
    p.add_argument("--margin-gb", type=float, default=3.0, help="磁盘安全余量 GB")
    p.add_argument("--include-original", action="store_true",
                   help="连同 original/*.pth 原始权重一起下载 (默认忽略, 只拉 safetensors 省盘)")
    p.add_argument("--ignore", nargs="*", default=None,
                   help=f"自定义忽略 glob 模式; 缺省={DEFAULT_IGNORE_PATTERNS}")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.cache_dir:
        cache_root = args.cache_dir
    elif args.source == "modelscope":
        cache_root = DEFAULT_MODELSCOPE_CACHE
    else:
        cache_root = DEFAULT_HF_HOME

    if args.include_original:
        ignore_patterns = args.ignore  # None 或用户显式指定
    else:
        ignore_patterns = args.ignore if args.ignore is not None else DEFAULT_IGNORE_PATTERNS

    targets = [_resolve(m) for m in args.models]
    print(f"[download_models] 源={args.source} 待下载={targets} 缓存={cache_root} "
          f"忽略={ignore_patterns}", flush=True)
    paths = []
    for repo_id in targets:
        paths.append(download_one(repo_id, source=args.source, cache_root=cache_root,
                                  margin_gb=args.margin_gb, ignore_patterns=ignore_patterns))
    print("[download_models] 全部完成。本地路径:", flush=True)
    for p in paths:
        print(f"  {p}", flush=True)
    print("[download_models] 登记提醒: 请在团队记忆登记已下载模型 (共享 50G 盘)。", flush=True)


if __name__ == "__main__":
    sys.exit(main())
