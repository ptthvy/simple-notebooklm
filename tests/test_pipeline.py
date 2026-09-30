import pytest

from src import rag
from src.export import export
from src.indexing import build_chunks, index_chunks, save_and_ingest_pdf
from src.learning import (
    LLMOutputError,
    NoContentError,
    _parse_json,
    generate_flashcards,
    generate_quiz,
    summarize,
)
from src.store import collection_count, ensure_collection, list_documents


def test_chunk_ids_are_unique_and_metadata_complete(sample_pdf):
    chunks = build_chunks([sample_pdf], chunk_size=300, chunk_overlap=30)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)) > 3
    assert {c.metadata["page"] for c in chunks} == {1, 2, 3}
    assert all(c.metadata["filename"] == "sample.pdf" for c in chunks)


def test_zero_overlap_is_respected(sample_pdf):
    a = build_chunks([sample_pdf], chunk_size=300, chunk_overlap=0)
    b = build_chunks([sample_pdf], chunk_size=300, chunk_overlap=100)
    assert len(a) < len(b)


def test_reingest_is_idempotent(sample_pdf):
    ensure_collection(recreate=True)
    n = index_chunks(build_chunks([sample_pdf], chunk_size=300, chunk_overlap=30))
    index_chunks(build_chunks([sample_pdf], chunk_size=300, chunk_overlap=30))
    assert collection_count() == n


def test_retrieve_ranks_relevant_page_first_and_filters(indexed):
    hits = rag.retrieve("low rank adaptation LoRA", k=3)
    assert hits[0].metadata.page == 2
    only_p3 = rag.retrieve("RAG", k=5, filters={"filename": "sample.pdf", "page": 3})
    assert only_p3 and {h.metadata.page for h in only_p3} == {3}
    assert rag.retrieve("RAG", filters={"filename": "nope.pdf"}) == []


def test_list_documents(indexed):
    docs = list_documents()
    assert len(docs) == 1 and docs[0].filename == "sample.pdf" and docs[0].pages == 3
    assert docs[0].chunks == len(indexed)


def test_answer_has_citations(indexed, fake_llm):
    res = rag.answer("What is RAG?")
    assert "[S1]" in res.answer and len(res.citations) == len(res.chunks) > 0
    assert res.citations[0].source_marker == "S1"
    assert "NGỮ CẢNH" in fake_llm[-1] and "What is RAG?" in fake_llm[-1]


def test_answer_without_context_skips_llm(indexed, fake_llm):
    res = rag.answer("anything", filters={"filename": "nope.pdf"})
    assert res.answer == rag.NO_CONTEXT_MESSAGE and res.chunks == [] and fake_llm == []


def test_answer_with_rerank_uses_initial_k(indexed, fake_llm, monkeypatch):
    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            return [float(len(t)) for _, t in pairs]

    monkeypatch.setattr("src.reranker.get_reranker", lambda name=None: FakeCE())
    res = rag.answer("What is RAG?", rerank=True, rerank_strict=True)
    assert len(res.chunks) == 5
    scores = [c.rerank_score for c in res.chunks]
    assert scores == sorted(scores, reverse=True) and all(s is not None for s in scores)


def test_parse_json_variants():
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json('blah {"a": [1, 2]} blah') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        _parse_json("not json at all")


def test_summarize_single(indexed, fake_llm, monkeypatch):
    monkeypatch.setattr("src.learning.settings.summarize_batch_size", 100)
    s = summarize(document="sample.pdf")
    assert s.scope == "document" and s.summary == "single summary [S1]"
    assert s.key_points == ["a", "b"]  # bỏ mục rỗng
    assert len(fake_llm) == 1 and len(s.citations) == len(s.chunks)


def test_summarize_map_reduce(indexed, fake_llm, monkeypatch):
    monkeypatch.setattr("src.learning.settings.summarize_batch_size", 3)
    s = summarize()
    n_batches = -(-len(indexed) // 3)
    assert s.scope == "corpus" and s.summary == "final summary [S1]"
    assert len(fake_llm) == n_batches + 1
    assert "[S4]" in fake_llm[1]  # batch 2 giữ đánh số nhãn nguồn toàn cục


def test_summarize_by_query_uses_retrieval(indexed, fake_llm, monkeypatch):
    monkeypatch.setattr("src.learning.settings.summarize_retrieval_k", 4)
    s = summarize(query="low rank adaptation")
    assert s.scope == "query" and s.target == "low rank adaptation" and len(s.chunks) <= 4


def test_quiz_validates_dedupes_and_filters_markers(indexed, fake_llm):
    quiz = generate_quiz(document="sample.pdf", count=8)
    assert [i.question for i in quiz.items] == ["What does RAG retrieve?", "What is LoRA?"]
    assert quiz.items[0].source_markers == ["S1"]  # S99 không tồn tại -> bị loại
    assert quiz.items[1].source_markers == ["S2"]


def test_quiz_count_truncates(indexed, fake_llm):
    assert len(generate_quiz(count=1).items) == 1


def test_flashcards_validate(indexed, fake_llm):
    deck = generate_flashcards()
    assert [c.front for c in deck.cards] == ["RAG", "LoRA"]
    assert deck.cards[1].source_markers == ["S2"]


def test_generation_caps_context(indexed, fake_llm, monkeypatch):
    monkeypatch.setattr("src.learning.settings.max_generation_chunks", 4)
    quiz = generate_quiz()
    assert len(quiz.chunks) == 4 and len(quiz.citations) == 4


def test_no_content_and_bad_llm_output(indexed, monkeypatch):
    with pytest.raises(NoContentError):
        generate_quiz(document="missing.pdf")
    monkeypatch.setattr("src.learning.invoke_llm", lambda p, provider=None: "sorry, no json")
    with pytest.raises(LLMOutputError):
        generate_flashcards()


def test_export_formats(indexed, fake_llm, tmp_path):
    quiz = generate_quiz()
    md = export(quiz, fmt="md")
    assert "### Câu 1." in md and "**Đáp án:** A. Chunks" in md
    assert '"items"' in export(quiz, fmt="json")
    out = export(generate_flashcards(), fmt="md", output=tmp_path / "x" / "cards.md")
    assert out.read_text(encoding="utf-8").startswith("# Flashcards")
    ans = export(rag.answer("What is RAG?"), fmt="text")
    assert "## Nguồn" in ans and "sample.pdf" in ans
    with pytest.raises(ValueError):
        export(quiz, fmt="pdf")


def test_save_and_ingest_validation(sample_pdf):
    with pytest.raises(ValueError):
        save_and_ingest_pdf(b"abc", "notes.txt")
    res = save_and_ingest_pdf(sample_pdf.read_bytes(), "../../evil/sample2.pdf")
    assert res["filename"] == "sample2.pdf" and res["chunks_indexed"] > 0
