"""Đánh giá RAG bằng Ragas (LLM-as-judge) trên benchmark câu hỏi / đáp án chuẩn."""
from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..schemas import RagAnswer

log = logging.getLogger(__name__)

BENCHMARK_FILE = Path(__file__).parent / "benchmark_rag.csv"
RESULTS_DIR = Path(__file__).parent / "results"
MIN_RECOMMENDED_CASES = 10

# tên cột Ragas -> nhãn ngắn dùng trong bảng
METRICS = {
    "faithfulness": "Faith.",
    "answer_relevancy": "Relev.",
    "context_precision": "Prec.",
    "context_recall": "Recall",
}


def load_benchmark(path: Path | str | None = None, limit: int | None = None) -> list[dict[str, str]]:
    path = Path(path) if path else BENCHMARK_FILE
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [
            {"question": r["question"].strip(), "ground_truth": r["ground_truth"].strip()}
            for r in csv.DictReader(f)
            if (r.get("question") or "").strip() and (r.get("ground_truth") or "").strip()
        ]
    if limit:
        rows = rows[:limit]
    if len(rows) < MIN_RECOMMENDED_CASES:
        log.warning(
            "Benchmark chỉ có %d câu (khuyến nghị >= %d). Điểm Ragas sẽ rất nhiễu — hãy bổ sung câu hỏi "
            "khớp với tài liệu trong data/ (xem: python -m src.evaluation.make_benchmark).",
            len(rows),
            MIN_RECOMMENDED_CASES,
        )
    return rows


def default_answer_fn(question: str) -> RagAnswer:
    from ..rag import answer

    return answer(question)


def get_ragas_metrics(llm, embeddings):
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings
    return metrics


@dataclass
class EvaluationOutcome:
    df: "object"  # pandas.DataFrame (mỗi dòng = 1 câu hỏi)
    failed: list[dict] = field(default_factory=list)  # câu hỏi lỗi khi sinh câu trả lời


def build_eval_records(test_cases: list[dict[str, str]], answer_fn: Callable[[str], RagAnswer]):
    """Chạy pipeline RAG cho từng câu hỏi; câu nào lỗi thì ghi lại và bỏ qua (không đưa điểm giả)."""
    records, failed = [], []
    for case in test_cases:
        try:
            rag_response = answer_fn(case["question"])
        except Exception as exc:  # noqa: BLE001
            log.error("Lỗi khi trả lời %r: %s", case["question"], exc)
            failed.append({"question": case["question"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        records.append(
            {
                "user_input": case["question"],
                "response": rag_response.answer,
                "retrieved_contexts": [c.text for c in rag_response.chunks],
                "reference": case["ground_truth"],
            }
        )
    return records, failed


def run_evaluation(
    test_cases: list[dict[str, str]],
    *,
    answer_fn: Callable[[str], RagAnswer] = default_answer_fn,
    llm_provider: str | None = None,
    timeout_s: int = 180,
    max_retries: int = 3,
    max_workers: int = 2,
) -> EvaluationOutcome:
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    from ..llm import get_llm
    from ..store import get_embeddings

    records, failed = build_eval_records(test_cases, answer_fn)
    if not records:
        raise RuntimeError("Không có câu hỏi nào trả lời được để đánh giá.")

    dataset = EvaluationDataset.from_list(records)
    llm = LangchainLLMWrapper(get_llm(provider=llm_provider))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = get_ragas_metrics(llm, embeddings)
    config = RunConfig(timeout=timeout_s, max_retries=max_retries, max_workers=max_workers)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=config,
        show_progress=True,
    )
    return EvaluationOutcome(df=result.to_pandas(), failed=failed)


def _clean(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def summary_metrics(df) -> dict[str, object]:
    """Trung bình từng chỉ số (bỏ qua NaN do judge lỗi) + số câu hợp lệ cho mỗi chỉ số."""
    import pandas as pd

    out: dict[str, object] = {"n_cases": int(len(df))}
    n_valid: dict[str, int] = {}
    for col in METRICS:
        if col not in df.columns:
            out[col], n_valid[col] = None, 0
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        out[col] = _clean(values.mean()) if values.notna().any() else None
        n_valid[col] = int(values.notna().sum())
    out["n_valid"] = n_valid
    return out


def all_metrics_missing(summary: dict[str, object]) -> bool:
    """True nếu judge không chấm được gì (thường do sai API key / hết quota / model trả rác)."""
    return all(summary.get(c) is None for c in METRICS)


JUDGE_FAILED_MSG = (
    "Judge LLM không chấm được câu nào (tất cả chỉ số là NaN). Kiểm tra GOOGLE_API_KEY, "
    "quota/rate limit (giảm --max-workers) hoặc chọn --judge-provider khác."
)


def write_json(path: Path | str, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
