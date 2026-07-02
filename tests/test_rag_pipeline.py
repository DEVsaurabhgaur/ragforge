"""
tests/test_rag_pipeline.py — Unit tests for RAGForge RAG pipeline logic.
"""
import re
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from rag_pipeline import (
    rerank_documents,
    expand_query,
    reformulate_question,
    validate_context_constraints,
    clear_vectorstore_cache,
)


def make_doc(content: str, source: str = "test.pdf", page: int = 0) -> Document:
    """Helper to create a Document with standard metadata."""
    return Document(page_content=content, metadata={"source_file": source, "page": page})


# ── rerank_documents ──────────────────────────────────────────────────────────

def test_rerank_returns_all_docs():
    """Verify rerank_documents returns the complete set of input documents."""
    docs = [make_doc("apple banana cherry"), make_doc("delta echo foxtrot")]
    result = rerank_documents(docs, "apple")
    assert len(result) == 2


def test_rerank_puts_best_match_first():
    """Verify rerank_documents places highest scoring match at the first index."""
    docs = [
        make_doc("completely unrelated content here"),
        make_doc("machine learning algorithms and neural networks"),
    ]
    result = rerank_documents(docs, "machine learning neural networks")
    assert "machine learning" in result[0].page_content.lower()


def test_rerank_with_empty_query():
    docs = [make_doc("hello world"), make_doc("foo bar")]
    result = rerank_documents(docs, "")
    assert len(result) == 2


def test_rerank_empty_docs():
    result = rerank_documents([], "some query")
    assert result == []


def test_rerank_phrase_match_boosts_score():
    docs = [
        make_doc("this has the exact phrase: machine learning overview"),
        make_doc("machine overview learning random words"),
    ]
    result = rerank_documents(docs, "machine learning overview")
    # The doc with exact phrase match should rank higher
    assert "machine learning overview" in result[0].page_content.lower()


# ── validate_context_constraints ─────────────────────────────────────────────

def test_validate_valid_answer():
    assert validate_context_constraints("The capital of France is Paris.") is True


def test_validate_refusal_phrase():
    assert validate_context_constraints("I could not find this in the documents.") is False


def test_validate_refusal_not_in_context():
    assert validate_context_constraints("This is not in the provided context.") is False


def test_validate_insufficient_info():
    assert validate_context_constraints("There is insufficient information to answer.") is False


def test_validate_new_refusal_outside_scope():
    assert validate_context_constraints("This is outside the scope of the documents.") is False


def test_validate_new_refusal_cannot_find():
    assert validate_context_constraints("I cannot find this information in the context.") is False


def test_validate_case_insensitive():
    assert validate_context_constraints("NO RELEVANT INFORMATION was found.") is False


# ── clear_vectorstore_cache ───────────────────────────────────────────────────

def test_clear_vectorstore_cache_runs():
    """Smoke test: clear_vectorstore_cache should not raise."""
    clear_vectorstore_cache()


# ── reformulate_question ──────────────────────────────────────────────────────

def test_reformulate_no_history_returns_original():
    mock_llm = MagicMock()
    result = reformulate_question("What is the summary?", [], mock_llm)
    assert result == "What is the summary?"
    mock_llm.invoke.assert_not_called()
