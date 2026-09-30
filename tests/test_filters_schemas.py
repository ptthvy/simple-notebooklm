import pytest
from pydantic import ValidationError

from src.filters import MetadataFilter, filters_to_dict, filters_to_qdrant
from src.schemas import QuizItem


def test_single_filename_list_collapses_to_filename():
    f = MetadataFilter(filenames=["  a.pdf "])
    assert f.filename == "a.pdf" and f.filenames is None


def test_multiple_files_drop_page_and_filename():
    f = MetadataFilter(filename="x.pdf", filenames=["a.pdf", "b.pdf"], page=3)
    assert f.filename is None and f.page is None and f.filenames == ["a.pdf", "b.pdf"]


def test_blank_strings_become_none():
    assert filters_to_dict({"filename": "   ", "section": " "}) is None


def test_to_qdrant_conditions():
    q = filters_to_qdrant({"filenames": ["a.pdf", "b.pdf"]})
    assert q.must[0].key == "metadata.filename" and q.must[0].match.any == ["a.pdf", "b.pdf"]
    q = filters_to_qdrant({"filename": "a.pdf", "page": 2})
    assert {c.key for c in q.must} == {"metadata.filename", "metadata.page"}
    assert filters_to_qdrant(None) is None


def test_quiz_item_requires_four_options_and_valid_index():
    ok = dict(question="q", options=list("abcd"), correct_index=3, explanation="e")
    assert QuizItem(**ok).correct_index == 3
    with pytest.raises(ValidationError):
        QuizItem(**{**ok, "options": list("abc")})
    with pytest.raises(ValidationError):
        QuizItem(**{**ok, "correct_index": 4})
