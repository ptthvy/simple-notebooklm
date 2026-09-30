"""Xuất kết quả (Pydantic model) ra text/Markdown/JSON."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .schemas import FlashcardSet, QuizSet, RagAnswer, Summary

ExportFormat = Literal["text", "md", "json"]
_LETTERS = "ABCD"


def _sources_md(citations) -> list[str]:
    if not citations:
        return []
    lines = ["", "## Nguồn", ""]
    for c in citations:
        lines.append(f"- [{c.source_marker}] {c.filename}, trang {c.page}")
    return lines


def _to_markdown(model: BaseModel) -> str:
    lines: list[str] = []

    if isinstance(model, RagAnswer):
        lines += ["# Hỏi đáp", "", f"**Câu hỏi:** {model.question}", "", model.answer]
        lines += _sources_md(model.citations)

    elif isinstance(model, Summary):
        lines += ["# Tóm tắt"]
        if model.target:
            lines += ["", f"*Phạm vi: {model.scope} — {model.target}*"]
        lines += ["", model.summary]
        if model.key_points:
            lines += ["", "## Ý chính", ""] + [f"- {p}" for p in model.key_points]
        lines += _sources_md(model.citations)

    elif isinstance(model, QuizSet):
        lines += ["# Câu hỏi trắc nghiệm"]
        for i, item in enumerate(model.items, start=1):
            lines += ["", f"### Câu {i}. {item.question}", ""]
            lines += [f"- {_LETTERS[j]}. {opt}" for j, opt in enumerate(item.options)]
            lines += ["", f"**Đáp án:** {_LETTERS[item.correct_index]}. {item.options[item.correct_index]}"]
            lines += [f"**Giải thích:** {item.explanation}"]
            if item.source_markers:
                lines += [f"**Nguồn:** {', '.join(item.source_markers)}"]
        lines += _sources_md(model.citations)

    elif isinstance(model, FlashcardSet):
        lines += ["# Flashcards"]
        for i, card in enumerate(model.cards, start=1):
            lines += ["", f"### Thẻ {i}", "", f"**Mặt trước:** {card.front}", f"**Mặt sau:** {card.back}"]
            if card.hint:
                lines += [f"**Gợi ý:** {card.hint}"]
            if card.source_markers:
                lines += [f"**Nguồn:** {', '.join(card.source_markers)}"]
        lines += _sources_md(model.citations)

    else:
        raise TypeError(f"Không hỗ trợ xuất kiểu {type(model).__name__}")

    return "\n".join(lines).rstrip() + "\n"


def export(model: BaseModel, *, fmt: ExportFormat = "text", output: Path | None = None):
    if fmt == "json":
        text = model.model_dump_json(indent=2) + "\n"
    elif fmt in {"text", "md"}:
        text = _to_markdown(model)
    else:
        raise ValueError(f"Unknown fmt '{fmt}'. Expected 'text' | 'md' | 'json'.")

    if output is None:
        return text

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output
