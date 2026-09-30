"""Rerank bằng Cross-Encoder (BAAI/bge-reranker-v2-m3).

Khác patch cũ: mặc định KHÔNG âm thầm fallback. Nếu Cross-Encoder không tải được thì báo lỗi,
để kết quả đánh giá "có reranker" không bị đo nhầm bằng lexical. Chế độ app có thể bật
`allow_fallback=True` (sẽ log cảnh báo rõ ràng).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import settings
from .schemas import RetrievedChunk

log = logging.getLogger(__name__)


class RerankerUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=2)
def get_reranker(model_name: str | None = None):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name or settings.reranker_model)


def _lexical_scores(query: str, chunks: list[RetrievedChunk]) -> list[float]:
    q = set(query.lower().split())
    return [len(q & set(c.text.lower().split())) / max(1, len(q)) for c in chunks]


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
    *,
    model=None,
    allow_fallback: bool = False,
) -> list[RetrievedChunk]:
    """Chấm lại điểm (query, chunk) và trả về top_k chunk tốt nhất (bản sao, có rerank_score)."""
    if not chunks:
        return []
    top_k = top_k or settings.rerank_top_k

    try:
        model = model or get_reranker()
        scores = model.predict([(query, c.text) for c in chunks], show_progress_bar=False)
        scores = [float(s) for s in scores]
    except Exception as exc:  # noqa: BLE001
        if not allow_fallback:
            raise RerankerUnavailable(f"{type(exc).__name__}: {exc}") from exc
        log.warning(
            "Cross-Encoder không dùng được (%s: %s) -> tạm dùng lexical overlap.",
            type(exc).__name__,
            exc,
        )
        scores = _lexical_scores(query, chunks)

    scored = [c.model_copy(update={"rerank_score": s}) for c, s in zip(chunks, scores)]
    return sorted(scored, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_k]
