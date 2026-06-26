"""
scripts/validate_env.py — Check that all required environment variables are set.
Usage: python scripts/validate_env.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import config

errors = []
warnings = []

# ── API Key checks ─────────────────────────────────────────────────────────
if config.LLM_PROVIDER == "openai" and not config.OPENAI_API_KEY:
    errors.append("OPENAI_API_KEY is not set but LLM_PROVIDER=openai")

if config.LLM_PROVIDER == "gemini" and not config.GEMINI_API_KEY:
    errors.append("GEMINI_API_KEY is not set but LLM_PROVIDER=gemini")

if config.EMBEDDING_PROVIDER == "openai" and not config.OPENAI_API_KEY:
    errors.append("OPENAI_API_KEY is not set but EMBEDDING_PROVIDER=openai")

# ── Soft warnings ─────────────────────────────────────────────────────────
if config.CHUNK_SIZE < 200:
    warnings.append(f"CHUNK_SIZE={config.CHUNK_SIZE} is unusually small (< 200 chars)")

if config.CHUNK_OVERLAP >= config.CHUNK_SIZE:
    warnings.append(
        f"CHUNK_OVERLAP={config.CHUNK_OVERLAP} >= CHUNK_SIZE={config.CHUNK_SIZE} "
        f"— overlap should be less than chunk size"
    )

if config.TOP_K_RESULTS < 1:
    errors.append(f"TOP_K_RESULTS must be >= 1, got {config.TOP_K_RESULTS}")

if not (0.0 <= config.DEFAULT_TEMPERATURE <= 1.0):
    warnings.append(
        f"DEFAULT_TEMPERATURE={config.DEFAULT_TEMPERATURE} is outside [0.0, 1.0] range"
    )

# ── Directory checks ──────────────────────────────────────────────────────
for path_name, path_val in [
    ("UPLOAD_DIR", config.UPLOAD_DIR),
    ("SESSION_DIR", config.SESSION_DIR),
    ("CHROMA_DB_DIR", config.CHROMA_DB_DIR),
]:
    if not os.path.exists(path_val):
        warnings.append(f"{path_name}={path_val!r} does not exist yet (will be created on startup)")

# ── Report ────────────────────────────────────────────────────────────────
if warnings:
    print("⚠️  Warnings:")
    for w in warnings:
        print(f"  ! {w}")

if errors:
    print("\n❌  Environment validation FAILED:")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("\n✅  Environment validation PASSED")
    print(f"  App Version      : {config.APP_VERSION}")
    print(f"  LLM Provider     : {config.LLM_PROVIDER}")
    print(f"  LLM Model        : {config.LLM_MODEL_GEMINI if config.LLM_PROVIDER == 'gemini' else config.LLM_MODEL_OPENAI}")
    print(f"  Embedding Mode   : {config.EMBEDDING_PROVIDER}")
    print(f"  Retrieval Mode   : {config.RETRIEVAL_MODE}")
    print(f"  Chunk Size       : {config.CHUNK_SIZE}")
    print(f"  Chunk Overlap    : {config.CHUNK_OVERLAP}")
    print(f"  Top K Results    : {config.TOP_K_RESULTS}")
    print(f"  Temperature      : {config.DEFAULT_TEMPERATURE}")
    print(f"  Max File Size    : {config.MAX_FILE_SIZE_MB} MB")
    print(f"  Max Documents    : {config.MAX_DOCUMENTS}")
