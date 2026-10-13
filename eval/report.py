"""结果汇总与报告 (QA / CH-3 维护): 生成 Markdown 表 + CSV + JSON。"""
from __future__ import annotations

import csv
import io
import json
from typing import List


REPORT_COLUMNS = [
    ("method", "方法"),
    ("kv_saving_pct", "KV显存节省%"),
    ("compression_ratio", "压缩比x"),
    ("peak_gpu_mb", "峰值显存MB"),
    ("ttft_ms", "TTFT(ms)"),
    ("tpot_ms", "TPOT(ms)"),
    ("decode_tps", "decode吞吐(tok/s)"),
    ("perplexity", "困惑度"),
    ("token_match_vs_full", "对full一致率"),
]


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def to_markdown(rows: List[dict], title: str = "KV 压缩评估结果") -> str:
    header = "| " + " | ".join(label for _, label in REPORT_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in REPORT_COLUMNS) + " |"
    lines = [f"### {title}", "", header, sep]
    for row in rows:
        cells = []
        for key, _ in REPORT_COLUMNS:
            digits = 0 if key in ("peak_gpu_mb",) else 2
            cells.append(_fmt(row.get(key), digits))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_csv(rows: List[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[k for k, _ in REPORT_COLUMNS], extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k, _ in REPORT_COLUMNS})
    return buf.getvalue()


def save_all(rows: List[dict], out_prefix: str, title: str = "KV 压缩评估结果",
             meta: dict | None = None) -> dict:
    """保存 .md/.csv/.json 三种格式, 返回各文件路径。"""
    md_path = f"{out_prefix}.md"
    csv_path = f"{out_prefix}.csv"
    json_path = f"{out_prefix}.json"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(rows, title))
        f.write("\n")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(to_csv(rows))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta or {}, "rows": rows}, f, ensure_ascii=False, indent=2)
    return {"markdown": md_path, "csv": csv_path, "json": json_path}
