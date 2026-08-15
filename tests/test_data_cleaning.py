"""Unit tests for clean_text function in prepare_imdb.py."""

from src.data.prepare_imdb import clean_text


def test_clean_text_strips_html_breaks():
    sample = "Great movie!<br /><br />Loved the ending.<br>Must watch!"
    expected = "Great movie! Loved the ending. Must watch!"
    assert clean_text(sample) == expected


def test_clean_text_collapses_whitespace():
    sample = "Too    many     spaces\n\nand   tabs\t\there."
    expected = "Too many spaces and tabs here."
    assert clean_text(sample) == expected


def test_clean_text_handles_non_string():
    assert clean_text(None) is None
