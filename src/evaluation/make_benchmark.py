"""Sinh BẢN NHÁP benchmark (question, ground_truth) từ chính các tài liệu đã index.

  python -m src.evaluation.make_benchmark --n 30 --out src/evaluation/benchmark_rag.csv

LƯU Ý: đáp án do LLM viết từ chính chunk nguồn nên có thể thiên vị (dễ đạt recall cao).
Hãy đọc lại, sửa hoặc xóa câu chưa tốt (cột `source` giúp đối chiếu) trước khi dùng để đánh giá.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from ..learning import LLMOutputError, _call_json, _sample_evenly
from ..rag import fetch_all_chunks
from .ragas_evaluator import BENCHMARK_FILE

log = logging.getLogger("make_benchmark")

PROMPT = """Bạn đang xây dựng bộ câu hỏi kiểm thử cho hệ thống hỏi đáp tài liệu.
Chỉ dựa vào ĐOẠN VĂN dưới đây, hãy viết 1 câu hỏi và 1 đáp án chuẩn.

Yêu cầu:
- Câu hỏi phải tự chứa ngữ cảnh: người đọc hiểu được mà không cần thấy đoạn văn (không dùng "đoạn trên", "tác giả").
- Câu hỏi hỏi về một kiến thức/khái niệm/quy trình cụ thể có trong đoạn văn.
- Đáp án chuẩn: 1-2 câu, chính xác, chỉ dùng thông tin trong đoạn văn.
- Nếu đoạn văn chỉ là mục lục, tiêu đề, danh sách tên, tài liệu tham khảo hoặc không chứa kiến thức, trả về {"skip": true}.
- Viết bằng ngôn ngữ của đoạn văn.

ĐOẠN VĂN:
%s

Chỉ trả về MỘT đối tượng JSON: {"question": "...", "ground_truth": "..."} hoặc {"skip": true}
"""


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=30, help="Số câu hỏi muốn có")
    p.add_argument("--document", default=None, help="Chỉ lấy từ 1 file")
    p.add_argument("--out", type=Path, default=BENCHMARK_FILE)
    p.add_argument("--min-chars", type=int, default=300, help="Bỏ qua chunk ngắn hơn ngưỡng này")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chunks = fetch_all_chunks(filters={"filename": args.document} if args.document else None)
    chunks = [c for c in chunks if len(c.text) >= args.min_chars]
    if not chunks:
        raise SystemExit("Không có chunk phù hợp. Hãy chạy `ingest` trước.")

    candidates = _sample_evenly(chunks, min(len(chunks), int(args.n * 1.6)))
    rows, seen = [], set()
    for c in candidates:
        if len(rows) >= args.n:
            break
        try:
            obj = _call_json(PROMPT % c.text)
        except LLMOutputError as exc:
            log.warning("Bỏ qua chunk %s: %s", c.metadata.chunk_id, exc)
            continue
        if not isinstance(obj, dict) or obj.get("skip"):
            continue
        q, a = str(obj.get("question", "")).strip(), str(obj.get("ground_truth", "")).strip()
        if not q or not a or q.lower() in seen:
            continue
        seen.add(q.lower())
        rows.append({"question": q, "ground_truth": a, "source": f"{c.metadata.filename}:p{c.metadata.page}"})
        log.info("[%d/%d] %s", len(rows), args.n, q)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "ground_truth", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"Đã ghi {len(rows)} câu vào {args.out}. Hãy đọc lại và chỉnh sửa trước khi đánh giá!")


if __name__ == "__main__":
    main()
