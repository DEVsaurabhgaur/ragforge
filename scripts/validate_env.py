"""
scripts/validate_env.py — Check that all required environment variables are set.
Usage: python scripts/validate_env.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import config

errors = []

if config.LLM_PROVIDER == "openai" and not config.OPENAI_API_KEY:
    errors.append("OPENAI_API_KEY is not set but LLM_PROVIDER=openai")

if config.LLM_PROVIDER == "gemini" and not config.GEMINI_API_KEY:
    errors.append("GEMINI_API_KEY is not set but LLM_PROVIDER=gemini")

if config.EMBEDDING_PROVIDER == "openai" and not config.OPENAI_API_KEY:
    errors.append("OPENAI_API_KEY is not set but EMBEDDING_PROVIDER=openai")

if errors:
    print("Environment validation FAILED:")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("Environment validation PASSED ✓")
    print(f"  LLM Provider     : {config.LLM_PROVIDER}")
    print(f"  Embedding Mode   : {config.EMBEDDING_PROVIDER}")
    print(f"  Retrieval Mode   : {config.RETRIEVAL_MODE}")
    print(f"  Chunk Size       : {config.CHUNK_SIZE}")
    print(f"  Top K Results    : {config.TOP_K_RESULTS}")
