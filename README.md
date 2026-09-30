# Simple NotebookLM (RAG học tập trên PDF cá nhân)

Hỏi đáp có trích dẫn, tóm tắt (map-reduce), quiz, flashcards trên PDF của bạn; có UI Streamlit, CLI, REST API,
và bộ đánh giá Ragas (so sánh chunking + reranking). Cấu trúc bám theo đề *AIO2025 – Building a Simple NotebookLM*.

```
PDF -> chunk -> embedding -> Qdrant ─┐
câu hỏi -> retrieve (15) -> [Cross-Encoder rerank -> 5] -> prompt (Jinja2) -> LLM -> câu trả lời + [S1][S2]...
```

## Minh chứng chạy thực tế

Ứng dụng đã được chạy với một PDF 38 trang (87 chunks), sử dụng Gemini làm LLM backend.

| Hỏi đáp RAG có trích dẫn | Tóm tắt có ý chính |
| --- | --- |
| ![Câu trả lời RAG và các trích dẫn nguồn](assets/screenshots/rag-answer.png) | ![Kết quả tóm tắt tài liệu](assets/screenshots/summary.png) |

| Quiz trắc nghiệm | Flashcards |
| --- | --- |
| ![Quiz được tạo từ tài liệu](assets/screenshots/quiz.png) | ![Flashcards được tạo từ tài liệu](assets/screenshots/flashcards.png) |

## 1. Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-evaluation.txt              # chỉ cần khi chạy đánh giá
cp .env.example .env                                    # rồi điền GOOGLE_API_KEY
```

Đặt PDF vào `data/`. Lần chạy đầu sẽ tải model embedding (và reranker) từ HuggingFace, mất một lúc.

## 2. Sử dụng

```bash
# Cách A: CLI
python -m src.interfaces.cli ingest --recreate
python -m src.interfaces.cli ask "RAG là gì?" --rerank
python -m src.interfaces.cli debug-retrieval "RAG là gì?"          # xem chunk, không gọi LLM
python -m src.interfaces.cli summarize -d ten_file.pdf --fmt md -o out/summary.md
python -m src.interfaces.cli quiz -d ten_file.pdf -n 8 --fmt json -o out/quiz.json
python -m src.interfaces.cli flashcards -q "LoRA" -n 10

# Cách B: Web (2 terminal)
uvicorn src.interfaces.api:app --port 8000
streamlit run src/interfaces/ui.py
```

Bộ lọc: `--filters '{"filename":"a.pdf","page":3}'` hoặc `{"filenames":["a.pdf","b.pdf"]}` (nhiều file thì bỏ `page`).

## 3. Đánh giá (Ragas)

1. **Benchmark**: `src/evaluation/benchmark_rag.csv` (cột `question`, `ground_truth`). File đi kèm chỉ có 3 dòng ví dụ
   từ đề, **không khớp tài liệu của bạn**. Sinh bản nháp từ chính PDF rồi *đọc lại, sửa, xóa câu dở*:
   `python -m src.evaluation.make_benchmark --n 30`
2. **Thử nhanh cho rẻ**: `python -m src.evaluation.run_chunking --strategies rc_1000_150 --limit 5`
3. **Chunking đầy đủ** (7 cấu hình, mỗi cấu hình một collection): `python -m src.evaluation.run_chunking`
4. **Reranking**: `python -m src.evaluation.run_reranking` (xem điểm vector vs rerank: thêm `--demo`)
5. Kết quả: `src/evaluation/results/` gồm JSON từng cấu hình, CSV theo từng câu, `report.md` (Bảng 2, Bảng 3),
   `chart_chunking.png`, `chart_reranking.png`. Dựng lại báo cáo: `python -m src.evaluation.report`.

Ước lượng thô mỗi câu hỏi tốn ~10–15 lời gọi LLM cho mỗi cấu hình (câu trả lời + 4 chỉ số Ragas). 30 câu × 9 lượt chạy là
vài nghìn lời gọi: hãy dùng `--limit`, giảm `--max-workers` nếu gặp rate limit, và `--skip-existing` để chạy tiếp.
Câu nào lỗi sẽ bị **bỏ qua và ghi lại**, chỉ số nào judge không chấm được là `n/a` (không tính là 0). Nếu judge hỏng
hoàn toàn (sai key, hết quota) kết quả được đánh dấu `error` thay vì ra bảng toàn `n/a`.

Đọc kết quả: nếu số câu benchmark nhỏ, chênh lệch < ~0.05 giữa các cấu hình chỉ là nhiễu. Đừng chép số từ đề bài; hãy
dùng số của bạn.

## 4. Lưu ý kỹ thuật

- **Qdrant local chỉ cho một tiến trình mở thư mục `storage/qdrant`.** Không chạy API/UI cùng lúc với CLI hay script
  đánh giá (lỗi *"already accessed by another instance"*). Muốn chạy song song: chạy Qdrant server
  (`docker run -p 6333:6333 qdrant/qdrant`) và đặt `RAG_QDRANT_URL=http://localhost:6333`.
- **Ghim `langchain-community==0.3.31`**: `ragas 0.4.x` lỗi import với community ≥ 0.4, và `langchain-experimental` kéo
  community lên ≥ 0.4. Vì vậy `SemanticChunker` được cài lại trong `chunking_strategies.py` (cùng thuật toán).
- **Reranker không âm thầm fallback**: trong đánh giá, nếu Cross-Encoder không tải được thì báo lỗi (để không đo nhầm
  bằng lexical). Trong app (`--rerank`) có fallback lexical kèm cảnh báo log.
- PDF scan (không có text) sẽ bị từ chối khi upload; cần OCR trước.
- Đổi LLM: `RAG_LLM_PROVIDER=gemini|vllm|hf_local`. Judge của Ragas có thể khác LLM trả lời: `--judge-provider`.

## 5. Kiểm thử

```bash
pip install -r requirements-dev.txt
pytest -q          # 51 test, chạy offline (embedding/LLM giả), khoảng vài giây
```

Bao phủ: chunking + metadata + idempotent re-ingest, filter, retrieve, RAG có trích dẫn, rerank, tóm tắt single/map-reduce,
quiz/flashcards (validate, khử trùng, lọc nhãn nguồn), export, CLI, API (kể cả ánh xạ lỗi), UI Streamlit (AppTest nối
FastAPI), chiến lược chunking, ghép nối Ragas, báo cáo.
**Chưa được kiểm chứng tự động**: chất lượng thật của model embedding/reranker/LLM và điểm Ragas thật, vì cần tải model và
gọi API bằng key của bạn.

## 6. Cấu trúc

```
src/config.py schemas.py filters.py store.py indexing.py rag.py reranker.py learning.py llm.py export.py
src/prompts/*.jinja2            src/interfaces/{api,cli,ui,styles}.py
src/evaluation/{chunking_strategies,ragas_evaluator,run_chunking,run_reranking,report,make_benchmark}.py
tests/
```
