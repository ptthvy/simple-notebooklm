import pytest

from src.reranker import RerankerUnavailable, rerank
from src.schemas import ChunkMetadata, RetrievedChunk


def _chunk(text, score=0.0):
    meta = ChunkMetadata(document_id="d", filename="f.pdf", source="/f.pdf", page=1, chunk_id="d:1:0")
    return RetrievedChunk(text=text, score=score, metadata=meta)


class Model:
    def predict(self, pairs, show_progress_bar=False):
        return [len(t) for _, t in pairs]


class Broken:
    def predict(self, *a, **k):
        raise OSError("cannot load model")


def test_rerank_orders_by_score_and_truncates_without_mutating():
    chunks = [_chunk("aa"), _chunk("aaaa"), _chunk("a")]
    out = rerank("q", chunks, top_k=2, model=Model())
    assert [c.text for c in out] == ["aaaa", "aa"]
    assert out[0].rerank_score == 4.0
    assert all(c.rerank_score is None for c in chunks)  # bản gốc không bị đổi


def test_strict_mode_raises_instead_of_silent_fallback():
    with pytest.raises(RerankerUnavailable, match="cannot load model"):
        rerank("q", [_chunk("x")], model=Broken())


def test_fallback_mode_warns_and_uses_lexical(caplog):
    chunks = [_chunk("nothing here"), _chunk("rag is retrieval augmented generation")]
    with caplog.at_level("WARNING"):
        out = rerank("what is rag", chunks, top_k=1, model=Broken(), allow_fallback=True)
    assert "lexical" in caplog.text
    assert out[0].text.startswith("rag is")


def test_empty_input():
    assert rerank("q", []) == []
