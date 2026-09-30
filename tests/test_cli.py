import json

from typer.testing import CliRunner

from src.interfaces.cli import app

runner = CliRunner()


def test_documents_and_ask(indexed, fake_llm):
    r = runner.invoke(app, ["documents"])
    assert r.exit_code == 0 and "sample.pdf" in r.output and "pages=3" in r.output

    r = runner.invoke(app, ["ask", "What is RAG?", "--filters", '{"filename": "sample.pdf"}'])
    assert r.exit_code == 0 and "[S1]" in r.output and "sample.pdf, trang" in r.output


def test_debug_retrieval_outputs_json_without_llm(indexed, fake_llm):
    r = runner.invoke(app, ["debug-retrieval", "low rank adaptation", "--k", "2"])
    assert r.exit_code == 0 and fake_llm == []
    data = json.loads(r.output)
    assert len(data) == 2 and data[0]["metadata"]["page"] == 2


def test_learning_commands_write_files(indexed, fake_llm, tmp_path):
    out = tmp_path / "quiz.json"
    r = runner.invoke(app, ["quiz", "-d", "sample.pdf", "-n", "3", "--fmt", "json", "-o", str(out)])
    assert r.exit_code == 0 and len(json.loads(out.read_text(encoding="utf-8"))["items"]) == 2
    r = runner.invoke(app, ["flashcards", "--fmt", "md"])
    assert r.exit_code == 0 and "# Flashcards" in r.output
    r = runner.invoke(app, ["summarize", "-d", "sample.pdf", "--fmt", "md"])
    assert r.exit_code == 0 and "# Tóm tắt" in r.output


def test_bad_filters_json(indexed):
    r = runner.invoke(app, ["ask", "q", "--filters", "{oops"])
    assert r.exit_code != 0


def test_ingest_command(sample_pdf):
    r = runner.invoke(app, ["ingest", "--recreate"])
    assert r.exit_code == 0 and "chunks indexed" in r.output
