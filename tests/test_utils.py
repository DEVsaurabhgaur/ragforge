"""
tests/test_utils.py — Unit tests for RAGForge utility helpers.
"""
import pytest
from utils import (
    clean_text,
    truncate_text,
    format_source_display,
    count_tokens,
    estimate_cost,
    highlight_keywords,
)


def test_clean_text_collapses_spaces():
    """Verify clean_text collapses multiple consecutive spaces."""
    assert "  " not in clean_text("hello   world")


def test_clean_text_removes_extra_newlines():
    """Verify clean_text removes excessive blank lines."""
    assert "\n\n\n" not in clean_text("line1\n\n\nline2")


def test_truncate_at_word_boundary():
    """Verify truncate_text truncates at a clean word boundary."""
    result = truncate_text("one two three four", 10)
    assert result.endswith("...")
    assert "four" not in result


def test_truncate_returns_full_when_short():
    """Verify truncate_text returns full text if below max length."""
    assert truncate_text("hi", 100) == "hi"


def test_format_source_display():
    """Verify source reference formatting includes name and page."""
    result = format_source_display("report.pdf", 3)
    assert "report.pdf" in result
    assert "3" in result


def test_count_tokens_positive():
    """Verify count_tokens returns positive count for simple string."""
    assert count_tokens("Hello world") > 0


def test_estimate_cost_openai():
    """Verify estimate_cost returns non-zero value for OpenAI models."""
    cost = estimate_cost(10000, 5000, provider="openai")
    assert cost > 0


def test_estimate_cost_gemini():
    cost = estimate_cost(10000, 5000, provider="gemini")
    assert cost > 0
    assert cost < estimate_cost(10000, 5000, provider="openai")


def test_highlight_keywords_marks_word():
    result = highlight_keywords("The quick brown fox", "quick fox")
    assert "<mark" in result
    assert "quick" in result


def test_highlight_keywords_safe_html():
    result = highlight_keywords("<script>alert(1)</script>", "script")
    assert "<script>" not in result
