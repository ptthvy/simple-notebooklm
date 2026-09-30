"""Tổng hợp kết quả thực nghiệm thành bảng Markdown + biểu đồ (từ các file JSON trong results/)."""
from __future__ import annotations

import json
from pathlib import Path

from .ragas_evaluator import METRICS, RESULTS_DIR

ORDER = [
    "rc_1500_200",
    "rc_1000_150",
    "rc_800_100",
    "rc_500_50",
    "semantic_interquartile",
    "semantic_std_dev",
    "semantic_percentile",
]
COLS = list(METRICS)  # faithfulness, answer_relevancy, context_precision, context_recall


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def load_chunking(results_dir: Path) -> list[dict]:
    rows = []
    for f in results_dir.glob("chunking_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("summary_metrics") and not d.get("error"):
            rows.append(d)
    rows.sort(key=lambda d: ORDER.index(d["strategy_id"]) if d["strategy_id"] in ORDER else 99)
    return rows


def load_reranking(results_dir: Path) -> dict | None:
    f = results_dir / "reranking.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def table_chunking(rows: list[dict]) -> str:
    head = "| Method | Config | Chunks | Faith. | Relev. | Prec. | Recall |\n|---|---|---|---|---|---|---|\n"
    body = ""
    for d in rows:
        m = d["summary_metrics"]
        body += (
            f"| {d['method']} | {d['config']} | {d['chunk_count']} | "
            + " | ".join(_fmt(m.get(c)) for c in COLS)
            + " |\n"
        )
    return head + body


def table_reranking(chunk_rows: list[dict], rerank: dict | None) -> str:
    head = (
        "| Phương pháp | Chunks | Faithfulness | Answer Relevance | Context Precision | Context Recall |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = []
    if rerank:
        runs = {r["name"]: r for r in rerank["runs"] if r.get("summary_metrics") and not r.get("error")}
        if "baseline" in runs:
            m = runs["baseline"]["summary_metrics"]
            lines.append(("Recursive (1000/150)", rerank["chunk_count"], m))
        if "reranker" in runs:
            m = runs["reranker"]["summary_metrics"]
            lines.append(("Recursive (1000/150) + Reranker", rerank["chunk_count"], m))
    sem = [d for d in chunk_rows if d["method"] == "Semantic"]
    if sem:
        best = max(sem, key=lambda d: sum(v for v in (d["summary_metrics"].get(c) for c in COLS) if v is not None))
        lines.insert(1 if lines else 0, (f"Semantic ({best['config']})", best["chunk_count"], best["summary_metrics"]))
    body = "".join(
        f"| {name} | {n} | " + " | ".join(_fmt(m.get(c)) for c in COLS) + " |\n" for name, n, m in lines
    )
    return head + body if lines else "_Chưa có kết quả reranking._\n"


def _bar_chart(labels: list[str], series: dict[str, list[float]], title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(labels))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.6), 4.5))
    for i, (name, vals) in enumerate(series.items()):
        vals = [0 if v is None else v for v in vals]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Điểm")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def build_report(results_dir: Path = RESULTS_DIR) -> Path:
    results_dir = Path(results_dir)
    chunk_rows = load_chunking(results_dir)
    rerank = load_reranking(results_dir)
    md = ["# Kết quả thực nghiệm", ""]

    if chunk_rows:
        md += ["## Chunking", "", table_chunking(chunk_rows)]
        rec = [d for d in chunk_rows if d["method"] == "Recursive"]
        _bar_chart(
            [f"{d['method'][:3]} {d['config']}" for d in chunk_rows],
            {METRICS[c]: [d["summary_metrics"].get(c) for d in chunk_rows] for c in COLS},
            "So sánh chỉ số Ragas theo chiến lược chunking",
            results_dir / "chart_chunking.png",
        )
        md += ["![chunking](chart_chunking.png)", ""]
        md += ["Số liệu rút ra từ bảng trên (tự động):", ""]
        for c in COLS:
            valid = [(d, d["summary_metrics"].get(c)) for d in chunk_rows if d["summary_metrics"].get(c) is not None]
            if valid:
                d, v = max(valid, key=lambda t: t[1])
                md.append(f"- Cao nhất về {METRICS[c]}: {d['method']} {d['config']} ({v:.4f})")
        md.append("")

    if rerank:
        md += ["## Reranking", "", table_reranking(chunk_rows, rerank)]
        md += [
            f"Cấu hình: initial_k={rerank['initial_k']} -> rerank_k={rerank['rerank_k']}, "
            f"model `{rerank['reranker_model']}`.",
            "",
        ]
        runs = [r for r in rerank["runs"] if r.get("summary_metrics") and not r.get("error")]
        if runs:
            _bar_chart(
                list(METRICS.values()),
                {r["name"]: [r["summary_metrics"].get(c) for c in COLS] for r in runs},
                "Baseline vs Reranker",
                results_dir / "chart_reranking.png",
            )
            md += ["![reranking](chart_reranking.png)", ""]
        for r in rerank["runs"]:
            if r.get("error"):
                md.append(f"> Lỗi khi chạy `{r['name']}`: {r['error']}")

    n = max(
        [d.get("n_cases", 0) for d in chunk_rows] + ([rerank["runs"][0].get("n_cases", 0)] if rerank else [0])
    )
    md += ["", f"_Số câu hỏi benchmark: {n}. Với n nhỏ, chênh lệch < ~0.05 giữa các cấu hình chưa đủ kết luận._", ""]

    out = results_dir / "report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"Đã ghi báo cáo: {out}")
    return out


if __name__ == "__main__":
    build_report()
