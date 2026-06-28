"""
tests/test_new_utils.py — Unit tests for new RAGForge utility helpers.
"""
import os
import tempfile
import pytest
from utils import (
    word_count,
    sanitize_filename,
    get_file_size_mb,
    clean_text,
    truncate_text,
)


def test_word_count_basic():
    """Verify word_count counts standard space-separated words."""
    assert word_count("hello world foo") == 3


def test_word_count_empty():
    """Verify word_count returns 0 for an empty string."""
    assert word_count("") == 0


def test_word_count_single_word():
    """Verify word_count returns 1 for a single word."""
    assert word_count("RAGForge") == 1


def test_word_count_with_punctuation():
    """Verify word_count ignores standard punctuation marks."""
    # punctuation is not counted as words
    result = word_count("Hello, world! How are you?")
    assert result == 5


def test_sanitize_filename_removes_special_chars():
    """Verify sanitize_filename strips special OS characters."""
    result = sanitize_filename("My File: <name> *#2!")
    assert "<" not in result
    assert ">" not in result
    assert "*" not in result
    assert "#" not in result


def test_sanitize_filename_replaces_spaces_with_underscores():
    """Verify sanitize_filename replaces spaces with underscores."""
    result = sanitize_filename("my file name")
    assert " " not in result
    assert "_" in result


def test_sanitize_filename_truncates_long_names():
    """Verify sanitize_filename limits length to 100 characters."""
    long_name = "a" * 200
    result = sanitize_filename(long_name)
    assert len(result) <= 100


def test_sanitize_filename_empty():
    """Verify sanitize_filename handles empty string cleanly."""
    assert sanitize_filename("") == ""


def test_get_file_size_mb_returns_float():
    """Verify get_file_size_mb returns a valid float value."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"x" * 1024)  # 1 KB
        tmp_path = tmp.name
    try:
        result = get_file_size_mb(tmp_path)
        assert isinstance(result, float)
        assert result >= 0.0
    finally:
        os.unlink(tmp_path)


def test_get_file_size_mb_empty_file():
    """Verify get_file_size_mb returns 0.0 for empty files."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp_path = tmp.name
    try:
        result = get_file_size_mb(tmp_path)
        assert result == 0.0
    finally:
        os.unlink(tmp_path)


def test_clean_text_strips_whitespace():
    result = clean_text("   hello   ")
    assert result == "hello"


def test_truncate_text_exact_boundary():
    text = "one two"
    result = truncate_text(text, 7)
    assert result == "one two"


# ── remove_stopwords tests ────────────────────────────────────────────────
from utils import remove_stopwords


def test_remove_stopwords_basic():
    result = remove_stopwords("the quick brown fox")
    assert "the" not in result.split()
    assert "quick" in result
    assert "fox" in result


def test_remove_stopwords_all_stopwords():
    result = remove_stopwords("a an the and or")
    assert result.strip() == ""


def test_remove_stopwords_empty():
    assert remove_stopwords("") == ""


def test_remove_stopwords_preserves_non_stopwords():
    result = remove_stopwords("machine learning algorithms")
    assert "machine" in result
    assert "learning" in result
    assert "algorithms" in result


# ── is_valid_collection_name tests ──────────────────────────────────────
from utils import is_valid_collection_name


def test_valid_collection_name():
    assert is_valid_collection_name("ragforge_docs") is True


def test_valid_collection_name_with_hyphens():
    assert is_valid_collection_name("my-collection-1") is True


def test_invalid_collection_name_too_short():
    assert is_valid_collection_name("ab") is False


def test_invalid_collection_name_too_long():
    assert is_valid_collection_name("a" * 64) is False


def test_invalid_collection_name_special_chars():
    assert is_valid_collection_name("my collection!") is False


def test_invalid_collection_name_starts_with_underscore():
    assert is_valid_collection_name("_mycoll") is False


def test_valid_collection_name_with_dots():
    assert is_valid_collection_name("my.collection.name") is True


def test_invalid_collection_name_consecutive_dots():
    assert is_valid_collection_name("my..collection") is False


def test_invalid_collection_name_ipv4():
    assert is_valid_collection_name("127.0.0.1") is False


def test_estimate_cost_specific_models():
    from utils import estimate_cost
    # gpt-4o should be more expensive than gpt-4o-mini
    cost_mini = estimate_cost(1000, 500, provider="openai", model="gpt-4o-mini")
    cost_pro = estimate_cost(1000, 500, provider="openai", model="gpt-4o")
    assert cost_pro > cost_mini

    # gemini-2.5-pro should be more expensive than gemini-2.5-flash
    cost_flash = estimate_cost(1000, 500, provider="gemini", model="gemini-2.5-flash")
    cost_gem_pro = estimate_cost(1000, 500, provider="gemini", model="gemini-2.5-pro")
    assert cost_gem_pro > cost_flash


def test_format_source_display_edge_cases():
    from utils import format_source_display
    assert format_source_display("doc.txt", None) == "doc.txt — Page N/A"
    assert format_source_display("doc.txt", "") == "doc.txt — Page N/A"
    assert format_source_display("doc.txt", "   ") == "doc.txt — Page N/A"


def test_clean_text_edge_cases():
    from utils import clean_text
    assert clean_text("") == ""
    # Only non-printable chars should yield empty string
    assert clean_text("\x01\x02\x03") == ""
    # Collapse multiple consecutive blank lines
    assert clean_text("line1\n\n\n\nline2") == "line1\n\nline2"
