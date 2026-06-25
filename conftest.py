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
