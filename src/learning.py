"""Chức năng học tập: tóm tắt (single / map-reduce), quiz, flashcards."""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .config import settings
from .filters import filters_to_dict
from .llm import invoke_llm
from .rag import (
    FLASHCARDS_TEMPLATE,
    QUIZ_TEMPLATE,
    SUMMARY_MAP_TEMPLATE,
    SUMMARY_REDUCE_TEMPLATE,
    SUMMARY_SINGLE_TEMPLATE,
    fetch_all_chunks,
    format_citations,
    render_prompt,
    retrieve,
)
from .schemas import Flashcard, FlashcardSet, QuizItem, QuizSet, RetrievedChunk, Summary


class NoContentError(ValueError):
    """Không tìm thấy nội dung nào trong phạm vi được chọn."""


class LLMOutputError(RuntimeError):
    """LLM trả về dữ liệu không hợp lệ sau khi đã thử lại."""


# ----------------------------------------------------------------- helpers
def _resolve_target(document, query, filters, k, retrieval_k, collection_name=None):
    effective = dict(filters_to_dict(filters) or {})
    if document:
        effective.pop("filenames", None)
        effective["filename"] = document
    effective = filters_to_dict(effective) or {}

    if query:
        chunks = retrieve(query, k=k or retrieval_k, filters=effective or None, collection_name=collection_name)
        return chunks, "query", query

    if effective:
        chunks = fetch_all_chunks(filters=effective, collection_name=collection_name)
        scope = "document" if document else "filter"
        target = ", ".join(f"{key}={value}" for key, value in effective.items())
        return chunks, scope, target

    return fetch_all_chunks(filters=None, collection_name=collection_name), "corpus", None


def _sample_evenly(items: list, limit: int) -> list:
    """Chọn `limit` phần tử phân bố đều, giữ nguyên thứ tự."""
    if limit <= 0 or len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def _parse_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if not starts or end == -1:
            raise
        obj = json.loads(cleaned[min(starts) : end + 1])

    if not isinstance(obj, (dict, list)):
        raise ValueError("Expected JSON object or array.")
    return obj


def _call_json(prompt: str, attempts: int = 2) -> Any:
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return _parse_json(invoke_llm(prompt))
        except ValueError as exc:  # JSONDecodeError là con của ValueError
            last = exc
    raise LLMOutputError(f"LLM không trả về JSON hợp lệ: {last}")


def _validate_summary_payload(payload: Any) -> tuple[str, list[str]]:
    if not isinstance(payload, dict):
        raise LLMOutputError("Kết quả tóm tắt phải là một đối tượng JSON.")
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise LLMOutputError("Kết quả tóm tắt thiếu trường 'summary'.")
    raw_points = payload.get("key_points") or []
    points = [str(p).strip() for p in raw_points if str(p).strip()] if isinstance(raw_points, list) else []
    return summary.strip(), points


def _validate_items(payload, key, model_class, dedup_field, label, valid_markers):
    if isinstance(payload, list):
        payload = {key: payload}
    raw_items = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(raw_items, list):
        raise LLMOutputError(f"Kết quả thiếu danh sách '{key}'.")

    items, seen = [], set()
    for raw in raw_items:
        try:
            item = model_class.model_validate(raw)
        except ValidationError:
            continue

        norm = str(getattr(item, dedup_field, "")).strip().lower()
        if not norm or norm in seen:
            continue

        seen.add(norm)
        markers = [m for m in item.source_markers if m in valid_markers]
        items.append(item.model_copy(update={"source_markers": markers}))

    if not items:
        raise LLMOutputError(f"No valid {label} produced.")
    return items


# ----------------------------------------------------------------- public API
def summarize(document=None, query=None, filters=None, k=None, collection_name=None) -> Summary:
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.summarize_retrieval_k, collection_name
    )
    if not chunks:
        raise NoContentError("Không tìm thấy nội dung trong phạm vi đã chọn.")
    chunks = _sample_evenly(chunks, settings.summarize_max_chunks)
    size = settings.summarize_batch_size

    if len(chunks) <= size:
        payload = _call_json(render_prompt(SUMMARY_SINGLE_TEMPLATE, chunks=chunks))
        summary_text, key_points = _validate_summary_payload(payload)
    else:
        partials = []
        for start in range(0, len(chunks), size):
            batch = chunks[start : start + size]
            payload = _call_json(render_prompt(SUMMARY_MAP_TEMPLATE, chunks=batch, offset=start))
            text, points = _validate_summary_payload(payload)
            partials.append({"summary": text, "key_points": points})

        payload = _call_json(render_prompt(SUMMARY_REDUCE_TEMPLATE, partials=partials))
        summary_text, key_points = _validate_summary_payload(payload)

    return Summary(
        scope=scope,
        target=target,
        summary=summary_text,
        key_points=key_points,
        citations=format_citations(chunks),
        chunks=chunks,
    )


def _prepare_generation(document, query, filters, k, collection_name) -> tuple[list[RetrievedChunk], str, str | None, set[str]]:
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.generation_retrieval_k, collection_name
    )
    if not chunks:
        raise NoContentError("Không tìm thấy nội dung trong phạm vi đã chọn.")
    chunks = _sample_evenly(chunks, settings.max_generation_chunks)
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    return chunks, scope, target, valid_markers


def generate_quiz(document=None, query=None, filters=None, count=None, k=None, collection_name=None) -> QuizSet:
    chunks, scope, target, valid_markers = _prepare_generation(document, query, filters, k, collection_name)
    n = count or settings.quiz_default_count
    payload = _call_json(render_prompt(QUIZ_TEMPLATE, chunks=chunks, count=n))
    items = _validate_items(payload, "items", QuizItem, "question", "quiz items", valid_markers)[:n]
    return QuizSet(scope=scope, target=target, items=items, chunks=chunks, citations=format_citations(chunks))


def generate_flashcards(document=None, query=None, filters=None, count=None, k=None, collection_name=None) -> FlashcardSet:
    chunks, scope, target, valid_markers = _prepare_generation(document, query, filters, k, collection_name)
    n = count or settings.flashcards_default_count
    payload = _call_json(render_prompt(FLASHCARDS_TEMPLATE, chunks=chunks, count=n))
    cards = _validate_items(payload, "cards", Flashcard, "front", "flashcards", valid_markers)[:n]
    return FlashcardSet(scope=scope, target=target, cards=cards, chunks=chunks, citations=format_citations(chunks))
