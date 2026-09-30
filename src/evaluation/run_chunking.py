"""So sánh các chiến lược chunking bằng Ragas.

Ví dụ:
  python -m src.evaluation.run_chunking                       # chạy cả 7 cấu hình
  python -m src.evaluation.run_chunking --strategies rc_1000_150,semantic_interquartile --limit 5
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ..config import settings
from ..indexing import ingest
from ..rag import answer
from ..schemas import RagAnswer
from .chunking_strategies import ChunkingStrategy, get_strategies
from .ragas_evaluator import (
    JUDGE_FAILED_MSG,
    RESULTS_DIR,
    all_metrics_missing,
    load_benchmark,
    run_evaluation,
    summary_metrics,
    write_json,
)

log = logging.getLogger("run_chunking")


def collection_for(strategy_id: str) -> str:
    return f"{settings.qdrant_collection}__{strategy_id}"


def _evaluate_strategy(
    strategy: ChunkingStrategy,
    output_dir: Path,
    test_cases: list[dict],
    judge_provider: str | None,
    max_workers: int,
) -> dict[str, object]:
    collection_name = collection_for(strategy.strategy_id)
    t0 = time.time()
    chunk_count = ingest(recreate=True, collection_name=collection_name, chunker=strategy.chunker)

    result_out: dict[str, object] = {
        "strategy_id": strategy.strategy_id,
        "method": strategy.method,
        "config": strategy.label,
        "params": strategy.params,
        "chunk_count": chunk_count,
        "n_cases": len(test_cases),
        "judge_provider": judge_provider or settings.llm_provider,
        "summary_metrics": {},
    }

    try:

        def answer_fn(q: str) -> RagAnswer:
            return answer(q, collection_name=collection_name)

        outcome = run_evaluation(
            test_cases, answer_fn=answer_fn, llm_provider=judge_provider, max_workers=max_workers
        )
        result_out["summary_metrics"] = summary_metrics(outcome.df)
        result_out["failed_questions"] = outcome.failed
        if all_metrics_missing(result_out["summary_metrics"]):
            result_out["error"] = JUDGE_FAILED_MSG
        outcome.df.to_csv(output_dir / f"chunking_{strategy.strategy_id}_per_question.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        log.exception("Lỗi khi đánh giá %s", strategy.strategy_id)
        result_out["error"] = f"{type(exc).__name__}: {exc}"

    result_out["elapsed_s"] = round(time.time() - t0, 1)
    write_json(output_dir / f"chunking_{strategy.strategy_id}.json", result_out)
    return result_out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategies", help="Danh sách id, cách nhau bởi dấu phẩy (mặc định: tất cả)")
    p.add_argument("--benchmark", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Chỉ dùng N câu đầu (chạy thử cho rẻ)")
    p.add_argument("--judge-provider", default=None, choices=["gemini", "vllm", "hf_local"])
    p.add_argument("--max-workers", type=int, default=2, help="Số luồng Ragas (giảm nếu bị rate limit)")
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--skip-existing", action="store_true", help="Bỏ qua cấu hình đã có kết quả hợp lệ")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ids = [s.strip() for s in args.strategies.split(",")] if args.strategies else None
    strategies = get_strategies(ids=ids)
    cases = load_benchmark(args.benchmark, limit=args.limit)

    for s in strategies:
        out_file = args.output_dir / f"chunking_{s.strategy_id}.json"
        if args.skip_existing and out_file.exists() and '"error"' not in out_file.read_text(encoding="utf-8"):
            log.info("Bỏ qua %s (đã có kết quả)", s.strategy_id)
            continue
        log.info("=== %s (%s %s) ===", s.strategy_id, s.method, s.label)
        res = _evaluate_strategy(s, args.output_dir, cases, args.judge_provider, args.max_workers)
        log.info("chunks=%s metrics=%s", res["chunk_count"], res.get("summary_metrics") or res.get("error"))

    from .report import build_report

    build_report(args.output_dir)


if __name__ == "__main__":
    main()
