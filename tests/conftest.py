import hashlib
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src import config, store


class BowEmbeddings(Embeddings):
    """Embedding giả dạng bag-of-words băm (đủ để test xếp hạng theo từ khóa, không cần tải model)."""

    dim = 256

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


PAGES = [
    "Retrieval Augmented Generation, or RAG, retrieves relevant chunks from a vector database and adds them "
    "to the prompt. This grounds the answer in the source documents and reduces hallucination. "
    "The retriever uses semantic search over embeddings stored in Qdrant. " * 3,
    "LoRA, short for low rank adaptation, freezes the original model weights and trains only a small low rank "
    "update. This makes fine tuning cheap in memory. Supervised fine tuning teaches the model how to answer. " * 3,
    "Ragas evaluates a RAG pipeline with context recall, context precision, faithfulness and answer relevancy. "
    "Faithfulness checks that the answer is supported by the retrieved context. " * 3,
]


def make_pdf(path: Path, pages: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    for text in pages:
        y = 800
        for line in textwrap.wrap(text, 90):
            c.drawString(40, y, line)
            y -= 14
        c.showPage()
    c.save()
    return path


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path / "qdrant")
    monkeypatch.setattr(config.settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(config.settings, "qdrant_url", None)
    store.get_client.cache_clear()
    monkeypatch.setattr(store, "get_embeddings", lambda: BowEmbeddings())
    yield
    try:
        store.get_client().close()
    except Exception:  # noqa: BLE001
        pass
    store.get_client.cache_clear()


@pytest.fixture
def sample_pdf():
    return make_pdf(config.settings.data_dir / "sample.pdf", PAGES)


@pytest.fixture
def indexed(sample_pdf):
    from src.indexing import build_chunks, index_chunks
    from src.store import ensure_collection

    ensure_collection(recreate=True)
    chunks = build_chunks([sample_pdf], chunk_size=300, chunk_overlap=30)
    index_chunks(chunks)
    return chunks


QUIZ_JSON = {
    "items": [
        {"question": "What does RAG retrieve?", "options": ["Chunks", "Images", "Audio", "Video"],
         "correct_index": 0, "explanation": "It retrieves chunks.", "source_markers": ["S1", "S99"]},
        {"question": "what does rag retrieve?", "options": ["a", "b", "c", "d"],
         "correct_index": 1, "explanation": "duplicate (case-insensitive)", "source_markers": []},
        {"question": "Only three options", "options": ["a", "b", "c"],
         "correct_index": 0, "explanation": "invalid", "source_markers": []},
        {"question": "Bad index", "options": ["a", "b", "c", "d"],
         "correct_index": 7, "explanation": "invalid", "source_markers": []},
        {"question": "What is LoRA?", "options": ["Low rank adaptation", "A DB", "A GPU", "A prompt"],
         "correct_index": 0, "explanation": "Low rank adaptation.", "source_markers": ["S2"]},
    ]
}
CARDS_JSON = {
    "cards": [
        {"front": "RAG", "back": "Retrieval Augmented Generation", "source_markers": ["S1"]},
        {"front": "rag", "back": "duplicate", "source_markers": []},
        {"front": "LoRA", "back": "Low rank adaptation", "hint": "low rank", "source_markers": ["S2", "S42"]},
        {"back": "missing front"},
    ]
}


@pytest.fixture
def fake_llm(monkeypatch):
    calls: list[str] = []

    def fake(prompt: str, provider=None) -> str:
        calls.append(prompt)
        if "trắc nghiệm" in prompt:
            return "```json\n" + json.dumps(QUIZ_JSON) + "\n```"
        if "thẻ ghi nhớ" in prompt:
            return json.dumps(CARDS_JSON)
        if "MỘT PHẦN" in prompt:
            return json.dumps({"summary": "partial summary [S1]", "key_points": ["kp [S1]"]})
        if "tóm tắt thành phần" in prompt:
            return json.dumps({"summary": "final summary [S1]", "key_points": ["final kp"]})
        if "tóm tắt nội dung" in prompt:
            return 'Đây là kết quả: {"summary": "single summary [S1]", "key_points": ["a", "", "b"]} hết.'
        return "RAG truy xuất các chunk liên quan [S1]."

    monkeypatch.setattr("src.rag.invoke_llm", fake)
    monkeypatch.setattr("src.learning.invoke_llm", fake)
    return calls
