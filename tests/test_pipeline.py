import os
import pytest
from pathlib import Path
from langchain_core.documents import Document

from utils import clean_text, truncate_text, count_tokens, estimate_cost, highlight_keywords
from rag_pipeline import (
    load_and_split_document,
    rerank_documents,
    reformulate_question,
    expand_query,
    validate_context_constraints
)


# Mock LLM for local fast testing without network calls
class MockLLM:
    def __init__(self, response_content: str):
        self.content = response_content

    def invoke(self, prompt: str):
        return self


def test_clean_text():
    raw_text = "Hello    World!   \n\n\nNew Line\t\tTab"
    cleaned = clean_text(raw_text)
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Hello World!")


def test_truncate_text():
    text = "This is a simple text that needs truncation"
    truncated = truncate_text(text, 15)
    assert truncated.endswith("...")
    assert len(truncated) <= 15


def test_count_tokens():
    text = "Hello world! This is a test."
    tokens = count_tokens(text, model_name="gpt-4o-mini")
    # tiktoken counts or fallback length
    assert tokens > 0


def test_estimate_cost():
    # OpenAI Mini pricing: $0.15 / 1M input, $0.60 / 1M output
    cost = estimate_cost(1000, 2000, provider="openai")
    expected = (1000 * 0.15 + 2000 * 0.60) / 1_000_000
    assert pytest.approx(cost) == expected


def test_highlight_keywords():
    text = "The quick brown fox jumps over the lazy dog"
    query = "brown dog"
    highlighted = highlight_keywords(text, query)
    
    assert '<mark style=' in highlighted
    assert "brown" in highlighted
    assert "dog" in highlighted


def test_rerank_documents():
    docs = [
        Document(page_content="Cats are great pets and love to sleep.", metadata={}),
        Document(page_content="Dogs are friendly and need to be walked daily.", metadata={}),
    ]
    query = "friendly dogs walk"
    reranked = rerank_documents(docs, query)
    
    # The second document matches "friendly" and "dogs" and "walked", should rank first
    assert "friendly" in reranked[0].page_content
    assert "Dogs" in reranked[0].page_content


def test_reformulate_question():
    mock_llm = MockLLM("What is the capital of France?")
    chat_history = [
        {"role": "user", "content": "I want to visit Europe."},
        {"role": "assistant", "content": "Great! Which country are you going to?"}
    ]
    question = "What is its capital?"
    standalone = reformulate_question(question, chat_history, mock_llm)
    assert standalone == "What is the capital of France?"


def test_expand_query():
    mock_llm = MockLLM("weather in London\nLondon weather forecast")
    variations = expand_query("London weather", mock_llm)
    assert len(variations) == 3
    assert variations[0] == "London weather"
    assert "weather in London" in variations


def test_validate_context_constraints():
    # True means it is valid (no refusal)
    assert validate_context_constraints("This document describes the process.") is True
    # False means refusal detected
    assert validate_context_constraints("I could not find this in the context.") is False


def test_load_and_split_document_txt(tmp_path):
    # Create temp text file
    temp_txt = tmp_path / "test.txt"
    temp_txt.write_text("Hello line 1.\n\nHello line 2.\nThis is a dummy context with sufficient length to pass the 20 character filter.", encoding="utf-8")
    
    chunks = load_and_split_document(str(temp_txt), chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["source_file"] == "test.txt"
        assert chunk.metadata["page"] == 0
