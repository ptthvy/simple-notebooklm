"""CLI bằng Typer. Chạy: python -m src.interfaces.cli --help"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ..export import export
from ..indexing import ingest as ingest_data_dir
from ..learning import generate_flashcards as generate_flashcards_learning
from ..learning import generate_quiz as generate_quiz_learning
from ..learning import summarize as summarize_learning
from ..rag import answer, retrieve
from ..store import list_documents

app = typer.Typer(help="RAG Learning System (Simple NotebookLM)", no_args_is_help=True, add_completion=False)

FiltersOpt = typer.Option(None, "--filters", "-f", help='JSON, vd: \'{"filename":"a.pdf","page":3}\'')
OutputOpt = typer.Option(None, "--output", "-o", help="Ghi kết quả ra file")
FmtOpt = typer.Option("text", "--fmt", help="text | md | json")


def _parse_filters(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"filters phải là JSON hợp lệ: {exc}") from exc
    if not isinstance(obj, dict):
        raise typer.BadParameter("filters phải là một JSON object")
    return obj


def _emit(result, output: Optional[Path], fmt: str) -> None:
    if output is None:
        typer.echo(export(result, fmt=fmt))
    else:
        path = export(result, fmt=fmt, output=output)
        typer.echo(f"Đã ghi: {path}")


def _print_sources(chunks) -> None:
    typer.echo("\nNguồn:")
    for i, c in enumerate(chunks, start=1):
        extra = f", rerank={c.rerank_score:.3f}" if c.rerank_score is not None else ""
        typer.echo(f"  [S{i}] {c.metadata.filename}, trang {c.metadata.page} (score={c.score:.3f}{extra})")


@app.command()
def ingest(recreate: bool = typer.Option(False, "--recreate", help="Xóa collection cũ rồi index lại")):
    """Index toàn bộ PDF trong data/."""
    count = ingest_data_dir(recreate=recreate)
    typer.echo(f"Done. {count} chunks indexed.")


@app.command()
def documents():
    """Liệt kê tài liệu đã index."""
    docs = list_documents()
    if not docs:
        typer.echo("Chưa có tài liệu nào. Hãy chạy: ingest")
    for d in docs:
        typer.echo(f"{d.filename}\tid={d.document_id}\tpages={d.pages}\tchunks={d.chunks}")


@app.command()
def ask(
    question: str,
    k: Optional[int] = typer.Option(None, "--k"),
    filters: Optional[str] = FiltersOpt,
    rerank: bool = typer.Option(False, "--rerank", help="Bật Cross-Encoder reranking"),
):
    """Hỏi đáp có trích dẫn nguồn."""
    result = answer(question, k=k, filters=_parse_filters(filters), rerank=rerank)
    typer.echo(result.answer)
    _print_sources(result.chunks)


@app.command("debug-retrieval")
def debug_retrieval(
    question: str,
    k: Optional[int] = typer.Option(None, "--k"),
    filters: Optional[str] = FiltersOpt,
    rerank: bool = typer.Option(False, "--rerank"),
):
    """Xem các chunk được truy xuất (không gọi LLM)."""
    chunks = retrieve(question, k=k, filters=_parse_filters(filters), rerank=rerank)
    typer.echo(json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2))


@app.command("summarize")
def summarize_cmd(
    document: Optional[str] = typer.Option(None, "--document", "-d"),
    query: Optional[str] = typer.Option(None, "--query", "-q"),
    filters: Optional[str] = FiltersOpt,
    k: Optional[int] = typer.Option(None, "--k"),
    output: Optional[Path] = OutputOpt,
    fmt: str = FmtOpt,
):
    """Tóm tắt theo tài liệu / truy vấn / bộ lọc."""
    result = summarize_learning(document=document, query=query, filters=_parse_filters(filters), k=k)
    _emit(result, output, fmt)


@app.command("quiz")
def quiz_cmd(
    document: Optional[str] = typer.Option(None, "--document", "-d"),
    query: Optional[str] = typer.Option(None, "--query", "-q"),
    filters: Optional[str] = FiltersOpt,
    count: Optional[int] = typer.Option(None, "--count", "-n"),
    k: Optional[int] = typer.Option(None, "--k"),
    output: Optional[Path] = OutputOpt,
    fmt: str = FmtOpt,
):
    """Tạo câu hỏi trắc nghiệm."""
    result = generate_quiz_learning(
        document=document, query=query, filters=_parse_filters(filters), count=count, k=k
    )
    _emit(result, output, fmt)


@app.command("flashcards")
def flashcards_cmd(
    document: Optional[str] = typer.Option(None, "--document", "-d"),
    query: Optional[str] = typer.Option(None, "--query", "-q"),
    filters: Optional[str] = FiltersOpt,
    count: Optional[int] = typer.Option(None, "--count", "-n"),
    k: Optional[int] = typer.Option(None, "--k"),
    output: Optional[Path] = OutputOpt,
    fmt: str = FmtOpt,
):
    """Tạo flashcards."""
    result = generate_flashcards_learning(
        document=document, query=query, filters=_parse_filters(filters), count=count, k=k
    )
    _emit(result, output, fmt)


if __name__ == "__main__":
    app()
