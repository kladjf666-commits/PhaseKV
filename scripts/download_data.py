"""评估数据下载脚本 (CH-10 补齐, 供全队复用)。

预拉取并缓存评估数据:
  - WikiText-2 (raw): 流式困惑度语料 (eval.datasets.load_wikitext 使用)。
  - LongBench 子集: 长上下文任务分数 (M1 主结果, scripts/run_longbench.py 使用)。

LongBench 注意点 (实测踩坑): THUDM/LongBench 的 HF 加载脚本把 data.zip 的 URL **硬编码
为 huggingface.co**, 绕过 HF_ENDPOINT, 在仅 hf-mirror 直连可用的机器上必然 Network
unreachable。故本脚本**直接从 HF_ENDPOINT(hf-mirror) 拉取 data.zip 并解压**到
LONGBENCH_DIR, run_longbench.py 直接读本地 JSONL, 不依赖该脆弱加载器。

在 release/kvcomp/ 下运行:
  python scripts/download_data.py                       # wikitext + LongBench data.zip
  python scripts/download_data.py --no-wikitext         # 只下 LongBench
  python scripts/download_data.py --longbench narrativeqa,qasper   # (校验这些任务存在)
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_HOME = "/root/autodl-tmp/hf-cache"
# LongBench 解压目录 (含 data/<task>.jsonl); run_longbench.py 默认读这里。
LONGBENCH_DIR = os.environ.get("LONGBENCH_DIR", "/root/autodl-tmp/hf-cache/longbench")

# M1 默认 LongBench 子集: 英文 QA (qa_f1 口径, "允许改写"语义), 与 idea 文档第6节对齐。
DEFAULT_LONGBENCH = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"]


def _setup_env() -> str:
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    hf_home = os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    os.makedirs(hf_home, exist_ok=True)
    return hf_home


def download_wikitext() -> None:
    from datasets import load_dataset

    print("[download_data] WikiText-2-raw-v1 (test+validation) ...", flush=True)
    for split in ("test", "validation"):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        print(f"[download_data]   wikitext/{split}: {len(ds)} 行", flush=True)


def download_longbench(tasks) -> None:
    endpoint = os.environ["HF_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/datasets/THUDM/LongBench/resolve/main/data.zip"
    os.makedirs(LONGBENCH_DIR, exist_ok=True)
    zip_path = os.path.join(LONGBENCH_DIR, "data.zip")
    data_dir = os.path.join(LONGBENCH_DIR, "data")

    if not os.path.isdir(data_dir) or not os.listdir(data_dir):
        print(f"[download_data] 拉取 LongBench data.zip: {url}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "kvcomp-downloader"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            got = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r[download_data]   data.zip {got/1e6:.0f}/{total/1e6:.0f}MB",
                          end="", flush=True)
        print("\n[download_data] 解压 data.zip ...", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(LONGBENCH_DIR)
        os.remove(zip_path)
    else:
        print(f"[download_data] LongBench 已存在: {data_dir} (跳过下载)", flush=True)

    for task in tasks:
        path = os.path.join(data_dir, f"{task}.jsonl")
        n = sum(1 for _ in open(path, encoding="utf-8")) if os.path.exists(path) else -1
        status = f"{n} 条样本" if n >= 0 else "缺失!"
        print(f"[download_data]   LongBench/{task}: {status} ({path})", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="评估数据下载 (WikiText + LongBench)")
    p.add_argument("--longbench", default=",".join(DEFAULT_LONGBENCH),
                   help="逗号分隔的 LongBench 任务名; 空字符串则跳过")
    p.add_argument("--no-wikitext", action="store_true", help="跳过 WikiText 下载")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _setup_env()
    print(f"[download_data] 端点={os.environ['HF_ENDPOINT']} 缓存={os.environ['HF_HOME']}",
          flush=True)
    if not args.no_wikitext:
        download_wikitext()
    tasks = [t.strip() for t in args.longbench.split(",") if t.strip()]
    if tasks:
        download_longbench(tasks)
    print("[download_data] 全部完成。", flush=True)


if __name__ == "__main__":
    sys.exit(main())
