"""Đánh giá tác động của Cross-Encoder reranking.

  python -m src.evaluation.run_reranking --demo      # xem điểm vector vs rerank cho 1 câu hỏi (không cần Ragas)
  python -m src.evaluation.run_reranking             # đánh giá Ragas: baseline vs baseline + reranker
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ..config import settings
from ..indexing import ingest
from ..rag import answer, retrieve
from ..store import collection_count
from .chunking_strategies import BASELINE_STRATEGY_ID, get_strategies
from .ragas_evaluator import (
    JUDGE_FAILED_MSG,
    RESULTS_DIR,
    all_metrics_missing,
    load_benchmark,
    run_evaluation,
    summary_metrics,
    write_json,
)
from .run_chunking import collection_for

log = logging.getLogger("run_reranking")


def demo(query: str, collection_name: str | None) -> None:
    print(f"Query: {query}")
    print(f"Reranker: {settings.reranker_model}")
    print(f"Initial K: {settings.rerank_initial_k} -> Rerank K: {settings.rerank_top_k}")
    for enabled in (False, True):
        print("\n" + "=" * 70)
        print("Rerank:", enabled)
        hits = retrieve(query, collection_name=collection_name, rerank=enabled, rerank_strict=True)
        for i, h in enumerate(hits, 1):
            rr = f"{h.rerank_score:.4f}" if h.rerank_score is not None else "-"
            print(f"{i:>2}. vector={h.score:.4f} rerank={rr} page={h.metadata.page} text={h.text[:160]!r}")


def _run(name: str, cases, collection_name: str, rerank: bool, judge: str | None, workers: int) -> dict:
    t0 = time.time()

    def answer_fn(q: str):
        return answer(q, collection_name=collection_name, rerank=rerank, rerank_strict=True)

    out: dict = {"name": name, "rerank": rerank, "n_cases": len(cases)}
    try:
        outcome = run_evaluation(cases, answer_fn=answer_fn, llm_provider=judge, max_workers=workers)
        out["summary_metrics"] = summary_metrics(outcome.df)
        out["failed_questions"] = outcome.failed
        if all_metrics_missing(out["summary_metrics"]):
            out["error"] = JUDGE_FAILED_MSG
    except Exception as exc:  # noqa: BLE001
        log.exception("Lỗi khi đánh giá %s", name)
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--query", default="RAG là gì và tại sao project sử dụng RAG?")
    p.add_argument("--collection", default=None, help="Collection dùng cho --demo (mặc định: collection chính)")
    p.add_argument("--benchmark", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--judge-provider", default=None, choices=["gemini", "vllm", "hf_local"])
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--reindex", action="store_true", help="Index lại collection baseline (Recursive 1000/150)")
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.demo:
        demo(args.query, args.collection)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = collection_for(BASELINE_STRATEGY_ID)
    if args.reindex or collection_count(name) == 0:
        log.info("Index collection baseline %s ...", name)
        baseline = get_strategies(ids=[BASELINE_STRATEGY_ID])[0]
        ingest(recreate=True, collection_name=name, chunker=baseline.chunker)

    cases = load_benchmark(args.benchmark, limit=args.limit)
    results = {
        "collection": name,
        "initial_k": settings.rerank_initial_k,
        "rerank_k": settings.rerank_top_k,
        "reranker_model": settings.reranker_model,
        "chunk_count": collection_count(name),
        "runs": [
            _run("baseline", cases, name, False, args.judge_provider, args.max_workers),
            _run("reranker", cases, name, True, args.judge_provider, args.max_workers),
        ],
    }
    write_json(args.output_dir / "reranking.json", results)

    from .report import build_report

    build_report(args.output_dir)


if __name__ == "__main__":
    main()
