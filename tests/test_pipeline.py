"""
tests/test_pipeline.py — Unit tests for RAGForge document loading and chunking logic.
"""
import os
import tempfile
import pytest
from pathlib import Path
from langchain_core.documents import Document
from rag_pipeline import load_and_split_document


def write_temp_file(content: str, suffix: str) -> str:
    """Write content to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='w', encoding='utf-8')
    tmp.write(content)
    tmp.close()
    return tmp.name


# ── load_and_split_document ─────────────────────────────────────────────────

def test_load_txt_returns_chunks():
    """Verify standard text file loader successfully splits content into chunks."""
    path = write_temp_file("Hello world.\nThis is RAGForge testing.\n" * 50, ".txt")
    try:
        chunks = load_and_split_document(path)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)
    finally:
        os.unlink(path)


def test_load_md_returns_chunks():
    """Verify markdown loader splits markdown structure into chunks."""
    path = write_temp_file(
        "# Title\n\nThis is a markdown document.\n\n" * 40,
        ".md"
    )
    try:
        chunks = load_and_split_document(path)
        assert len(chunks) > 0
    finally:
        os.unlink(path)


def test_load_sets_source_file_metadata():
    """Verify loader metadata tracks source filename correctly."""
    path = write_temp_file("Content for testing metadata.\n" * 30, ".txt")
    filename = Path(path).name
    try:
        chunks = load_and_split_document(path)
        for chunk in chunks:
            assert chunk.metadata.get("source_file") == filename
    finally:
        os.unlink(path)


def test_load_sets_page_metadata():
    """Verify loader metadata defaults to page metadata index tracking."""
    path = write_temp_file("Some content.\n" * 30, ".txt")
    try:
        chunks = load_and_split_document(path)
        for chunk in chunks:
            assert "page" in chunk.metadata
    finally:
        os.unlink(path)


def test_load_filters_tiny_chunks():
    """Chunks with fewer than 20 chars after stripping should be removed."""
    path = write_temp_file(
        "Short.\n\n" + "This is a real meaningful chunk with enough content to pass the minimum size filter.\n" * 20,
        ".txt"
    )
    try:
        chunks = load_and_split_document(path)
        for chunk in chunks:
            assert len(chunk.page_content.strip()) > 20
    finally:
        os.unlink(path)


def test_load_custom_chunk_size():
    path = write_temp_file("Word " * 500, ".txt")
    try:
        chunks_small = load_and_split_document(path, chunk_size=200, chunk_overlap=50)
        chunks_large = load_and_split_document(path, chunk_size=800, chunk_overlap=100)
        # Smaller chunks should produce more documents
        assert len(chunks_small) > len(chunks_large)
    finally:
        os.unlink(path)


def test_load_unsupported_extension_returns_empty_or_attempts_fallback():
    """Unsupported but valid text files should either load or return empty gracefully."""
    path = write_temp_file("Some content\n" * 30, ".xyz")
    try:
        result = load_and_split_document(path)
        # Should be list (either empty or with fallback-loaded chunks)
        assert isinstance(result, list)
    finally:
        os.unlink(path)


def test_load_empty_file_returns_no_useful_chunks():
    path = write_temp_file("", ".txt")
    try:
        chunks = load_and_split_document(path)
        # Empty file should produce 0 chunks (all filtered by min size)
        for chunk in chunks:
            assert len(chunk.page_content.strip()) > 0
    finally:
        os.unlink(path)
