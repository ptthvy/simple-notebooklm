"""Pipeline RAG: retrieve -> (rerank) -> prompt -> LLM -> RagAnswer."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .config import settings
from .filters import filters_to_qdrant
from .llm import invoke_llm
from .reranker import rerank as rerank_chunks
from .schemas import ChunkMetadata, Citation, RagAnswer, RetrievedChunk
from .store import get_vector_store, scroll_all

PROMPTS_DIR = Path(__file__).parent / "prompts"
ANSWER_TEMPLATE = "answer.jinja2"
SUMMARY_SINGLE_TEMPLATE = "summary_single.jinja2"
SUMMARY_MAP_TEMPLATE = "summary_map.jinja2"
SUMMARY_REDUCE_TEMPLATE = "summary_reduce.jinja2"
QUIZ_TEMPLATE = "quiz.jinja2"
FLASHCARDS_TEMPLATE = "flashcards.jinja2"

NO_CONTEXT_MESSAGE = "Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời."


def retrieve(
    query: str,
    k: int | None = None,
    filters=None,
    collection_name: str | None = None,
    rerank: bool = False,
    rerank_strict: bool = False,
) -> list[RetrievedChunk]:
    """Semantic search. Khi rerank=True: lấy `rerank_initial_k` ứng viên rồi chọn `k`/`rerank_top_k` tốt nhất."""
    final_k = k or (settings.rerank_top_k if rerank else settings.top_k)
    fetch_k = max(settings.rerank_initial_k, final_k) if rerank else final_k

    hits = get_vector_store(collection_name).similarity_search_with_score(
        query=query,
        k=fetch_k,
        filter=filters_to_qdrant(filters),
    )
    chunks = [
        RetrievedChunk(text=doc.page_content, score=float(score), metadata=ChunkMetadata(**doc.metadata))
        for doc, score in hits
    ]
    if rerank:
        return rerank_chunks(query, chunks, final_k, allow_fallback=not rerank_strict)
    return chunks


def fetch_all_chunks(filters=None, collection_name: str | None = None) -> list[RetrievedChunk]:
    """Lấy toàn bộ chunk thỏa filter (scroll), sắp theo file/trang/thứ tự chunk."""
    name = collection_name or settings.qdrant_collection
    results: list[RetrievedChunk] = []

    for page in scroll_all(name, scroll_filter=filters_to_qdrant(filters)):
        for point in page:
            payload = point.payload or {}
            meta, text = payload.get("metadata") or {}, payload.get("page_content") or ""
            if meta and text:
                results.append(RetrievedChunk(text=text, score=0.0, metadata=ChunkMetadata(**meta)))

    return sorted(
        results,
        key=lambda r: (r.metadata.filename, r.metadata.page, int(r.metadata.chunk_id.rsplit(":", 1)[-1])),
    )


@lru_cache(maxsize=1)
def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt(template_name: str, **context) -> str:
    return _jinja_env().get_template(template_name).render(**context)


def format_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            source_index=i,
            source_marker=f"S{i}",
            filename=c.metadata.filename,
            page=c.metadata.page,
            section=c.metadata.section,
            chunk_id=c.metadata.chunk_id,
        )
        for i, c in enumerate(chunks, start=1)
    ]


def answer(
    question: str,
    k: int | None = None,
    filters=None,
    collection_name: str | None = None,
    rerank: bool = False,
    rerank_strict: bool = False,
) -> RagAnswer:
    chunks = retrieve(
        question,
        k=k,
        filters=filters,
        collection_name=collection_name,
        rerank=rerank,
        rerank_strict=rerank_strict,
    )
    if not chunks:
        return RagAnswer(question=question, answer=NO_CONTEXT_MESSAGE)

    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=chunks)
    text = invoke_llm(prompt)

    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(chunks),
        chunks=chunks,
    )
