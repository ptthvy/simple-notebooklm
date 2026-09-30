import json

import numpy as np
import pandas as pd
import pytest
from langchain_core.documents import Document

from src.evaluation import report
from src.evaluation.chunking_strategies import RecursiveChunker, SemanticChunkerWrapper, get_strategies
from src.evaluation.ragas_evaluator import build_eval_records, load_benchmark, summary_metrics
from src.schemas import RagAnswer

from .conftest import BowEmbeddings

TEXT = " ".join(
    ["The cat sat on the warm mat and purred softly."] * 3
    + ["A kitten chased the small ball of yarn across the mat."] * 3
    + ["Stock markets rallied as investors bought bank shares today."] * 3
    + ["The central bank raised interest rates to fight inflation."] * 3
)


def test_seven_strategies_with_expected_ids():
    ids = [s.strategy_id for s in get_strategies(embeddings=BowEmbeddings())]
    assert ids == [
        "rc_500_50", "rc_800_100", "rc_1000_150", "rc_1500_200",
        "semantic_percentile", "semantic_std_dev", "semantic_interquartile",
    ]
    s = get_strategies(embeddings=BowEmbeddings(), ids=["rc_1000_150"])[0]
    assert (s.method, s.label) == ("Recursive", "1000/150")
    with pytest.raises(ValueError):
        get_strategies(embeddings=BowEmbeddings(), ids=["nope"])


def test_bigger_recursive_chunks_mean_fewer_chunks():
    doc = [Document(page_content=TEXT * 6, metadata={"page": 1})]
    counts = [len(RecursiveChunker(size, size // 10).split_documents(doc)) for size in (200, 500, 1500)]
    assert counts[0] > counts[1] > counts[2] >= 1


def test_semantic_chunker_splits_at_topic_boundary_and_keeps_metadata():
    sc = SemanticChunkerWrapper(BowEmbeddings(), breakpoint_type="percentile", breakpoint_amount=90)
    pieces = sc.split_text(TEXT)
    assert len(pieces) >= 2
    assert " ".join(pieces).split() == TEXT.split()  # không mất/đổi chữ nào
    assert len(pieces) == 2  # đúng 2 chủ đề: mèo / tài chính
    assert pieces[0].endswith("across the mat.") and pieces[1].startswith("Stock markets")
    docs = sc.split_documents([Document(page_content=TEXT, metadata={"page": 7, "filename": "f.pdf"})])
    assert all(d.metadata["page"] == 7 for d in docs)


@pytest.mark.parametrize("kind", ["percentile", "standard_deviation", "interquartile"])
def test_semantic_thresholds_run(kind):
    assert SemanticChunkerWrapper(BowEmbeddings(), breakpoint_type=kind).split_text(TEXT)


def test_semantic_caps_overlong_chunks():
    sc = SemanticChunkerWrapper(BowEmbeddings(), max_chunk_chars=400)
    long_text = "word " * 500  # 1 câu duy nhất, không dấu chấm
    assert all(len(p) <= 400 for p in sc.split_text(long_text))


def test_summary_metrics_ignores_nan_and_reports_valid_counts():
    df = pd.DataFrame(
        {
            "faithfulness": [1.0, np.nan, 0.5],
            "answer_relevancy": [0.8, 0.6, 0.7],
            "context_precision": [np.nan, np.nan, np.nan],
            "context_recall": [1.0, 1.0, 0.0],
        }
    )
    m = summary_metrics(df)
    assert m["faithfulness"] == 0.75 and m["context_precision"] is None
    assert m["n_valid"] == {"faithfulness": 2, "answer_relevancy": 3, "context_precision": 0, "context_recall": 3}
    assert m["n_cases"] == 3


def test_load_benchmark_ships_valid_csv(caplog):
    with caplog.at_level("WARNING"):
        rows = load_benchmark()
    assert len(rows) == 3 and rows[0]["ground_truth"]
    assert "Benchmark chỉ có 3 câu" in caplog.text


def test_build_eval_records_skips_failed_questions():
    def answer_fn(q):
        if q == "bad":
            raise RuntimeError("boom")
        return RagAnswer(question=q, answer="ans")

    cases = [{"question": "ok", "ground_truth": "gt"}, {"question": "bad", "ground_truth": "gt"}]
    records, failed = build_eval_records(cases, answer_fn)
    assert len(records) == 1 and records[0]["reference"] == "gt"
    assert failed == [{"question": "bad", "error": "RuntimeError: boom"}]


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_report_builds_tables_and_charts(tmp_path):
    def metrics(f, r, p, c):
        return {"faithfulness": f, "answer_relevancy": r, "context_precision": p, "context_recall": c}

    _write(tmp_path / "chunking_rc_1000_150.json", {
        "strategy_id": "rc_1000_150", "method": "Recursive", "config": "1000/150", "chunk_count": 34,
        "n_cases": 12, "summary_metrics": metrics(0.8, 0.7, 0.9, 0.95)})
    _write(tmp_path / "chunking_semantic_std_dev.json", {
        "strategy_id": "semantic_std_dev", "method": "Semantic", "config": "Std_dev", "chunk_count": 29,
        "n_cases": 12, "summary_metrics": metrics(0.85, 0.75, 0.8, 0.9)})
    _write(tmp_path / "chunking_rc_500_50.json", {"strategy_id": "rc_500_50", "error": "boom", "summary_metrics": {}})
    _write(tmp_path / "reranking.json", {
        "collection": "c", "initial_k": 15, "rerank_k": 5, "reranker_model": "m", "chunk_count": 34,
        "runs": [
            {"name": "baseline", "n_cases": 12, "summary_metrics": metrics(0.8, 0.7, 0.9, 0.95)},
            {"name": "reranker", "n_cases": 12, "summary_metrics": metrics(0.9, 0.8, 0.95, 0.97)},
        ]})

    out = report.build_report(tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "| Recursive | 1000/150 | 34 | 0.8000 | 0.7000 | 0.9000 | 0.9500 |" in text
    assert "rc_500_50" not in text and "500/50" not in text  # cấu hình lỗi không lọt vào bảng
    assert "Recursive (1000/150) + Reranker" in text and "Semantic (Std_dev)" in text
    assert (tmp_path / "chart_chunking.png").stat().st_size > 1000
    assert (tmp_path / "chart_reranking.png").stat().st_size > 1000
