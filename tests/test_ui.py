"""Chạy UI Streamlit trong-process, nối `httpx.request` vào FastAPI TestClient (không cần server thật)."""
from pathlib import Path

import httpx
import pytest

pytest.importorskip("streamlit")
from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.interfaces.api import app  # noqa: E402

UI = str(Path(__file__).resolve().parents[1] / "src" / "interfaces" / "ui.py")


@pytest.fixture
def ui(indexed, fake_llm, monkeypatch):
    client = TestClient(app)

    def fake_request(method, url, **kwargs):
        kwargs.pop("timeout", None)
        return client.request(method, url.replace("http://localhost:8000", ""), **kwargs)

    monkeypatch.setattr(httpx, "request", fake_request)
    at = AppTest.from_file(UI, default_timeout=60).run()
    assert not at.exception
    return at


def test_sidebar_lists_documents_and_tabs_render(ui):
    assert any("sample.pdf" in c.value for c in ui.sidebar.caption)
    assert len(ui.tabs) == 4


def test_chat_flow(ui):
    ui.chat_input[0].set_value("What is RAG?").run()
    assert not ui.exception
    assert any("[S1]" in m.value for m in ui.markdown) or any("[S1]" in t.value for t in ui.text)


def test_summary_quiz_flashcards_buttons(ui):
    ui.button(key="sum-btn").click().run()
    assert not ui.exception and not ui.error
    ui.button(key="quiz-btn").click().run()
    assert not ui.exception and not ui.error and len(ui.radio) == 2
    ui.button(key="fc-btn").click().run()
    assert not ui.exception and not ui.error
    assert len(ui.expander) >= 2


def test_api_down_shows_friendly_error(monkeypatch):
    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "request", down)
    at = AppTest.from_file(UI, default_timeout=60).run()
    assert not at.exception and any("Không kết nối được API" in e.value for e in at.error)
