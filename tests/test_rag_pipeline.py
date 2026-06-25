"""
tests/test_rag_pipeline.py — Unit tests for rag_pipeline module functions.
"""
import pytest
from langchain_core.documents import Document
from rag_pipeline import (
    rerank_documents,
    validate_context_constraints,
    expand_query,
    reformulate_question,
)


class MockLLM:
    def __init__(self, resp):
        self.content = resp
    def invoke(self, _):
        return self


def test_rerank_exact_phrase_scores_highest():
    docs = [
        Document(page_content="Python is a programming language", metadata={}),
        Document(page_content="Machine learning uses neural networks", metadata={}),
    ]
    ranked = rerank_documents(docs, "Python programming language")
    assert "Python" in ranked[0].page_content


def test_rerank_preserves_all_docs():
    docs = [Document(page_content=f"doc {i}", metadata={}) for i in range(5)]
    assert len(rerank_documents(docs, "query")) == 5


def test_validate_constraints_pass():
    assert validate_context_constraints("Here is the answer from the document.") is True


def test_validate_constraints_fail_not_found():
    assert validate_context_constraints("I could not find this in the context.") is False


def test_validate_constraints_fail_not_mentioned():
    assert validate_context_constraints("This is not mentioned in the context.") is False


def test_expand_query_includes_original():
    llm = MockLLM("alt query one\nalt query two")
    result = expand_query("test question", llm)
    assert result[0] == "test question"


def test_expand_query_returns_list():
    llm = MockLLM("variation one\nvariation two")
    result = expand_query("something", llm)
    assert isinstance(result, list)
    assert len(result) >= 1


def test_reformulate_no_history():
    llm = MockLLM("standalone")
    result = reformulate_question("What is it?", [], llm)
    assert result == "What is it?"


def test_reformulate_with_history():
    llm = MockLLM("What is the capital of France?")
    history = [{"role": "user", "content": "Tell me about France"}]
    result = reformulate_question("What is its capital?", history, llm)
    assert result == "What is the capital of France?"


def test_rerank_empty_docs():
    assert rerank_documents([], "query") == []
