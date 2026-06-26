"""
conftest.py — Shared pytest fixtures and path configuration.
"""
import sys
import os
import pytest

# Ensure project root is on sys.path for all test modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def sample_text():
    """Reusable sample text fixture for text processing tests."""
    return "This is a sample document with   extra  spaces and\n\n\n triple newlines."


@pytest.fixture
def sample_query():
    """Reusable query fixture for retrieval tests."""
    return "What is the main topic discussed?"


@pytest.fixture(autouse=True)
def mock_external_services():
    """Globally mock database connections and embedding model initialization for all tests."""
    from unittest.mock import patch
    with patch("langchain_community.vectorstores.Chroma") as mock_chroma, \
         patch("langchain_community.embeddings.HuggingFaceEmbeddings") as mock_hf, \
         patch("langchain_openai.OpenAIEmbeddings") as mock_openai_emb:
        yield mock_chroma, mock_hf, mock_openai_emb
