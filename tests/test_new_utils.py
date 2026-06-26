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
    assert word_count("hello world foo") == 3


def test_word_count_empty():
    assert word_count("") == 0


def test_word_count_single_word():
    assert word_count("RAGForge") == 1


def test_word_count_with_punctuation():
    # punctuation is not counted as words
    result = word_count("Hello, world! How are you?")
    assert result == 5


def test_sanitize_filename_removes_special_chars():
    result = sanitize_filename("My File: <name> *#2!")
    assert "<" not in result
    assert ">" not in result
    assert "*" not in result
    assert "#" not in result


def test_sanitize_filename_replaces_spaces_with_underscores():
    result = sanitize_filename("my file name")
    assert " " not in result
    assert "_" in result


def test_sanitize_filename_truncates_long_names():
    long_name = "a" * 200
    result = sanitize_filename(long_name)
    assert len(result) <= 100


def test_sanitize_filename_empty():
    assert sanitize_filename("") == ""


def test_get_file_size_mb_returns_float():
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
