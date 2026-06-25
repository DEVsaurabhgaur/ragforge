"""
scripts/check_deps.py — Verify that all required Python packages are installed.
Usage: python scripts/check_deps.py
"""
REQUIRED = [
    "streamlit", "langchain", "langchain_community", "langchain_classic",
    "langchain_google_genai", "langchain_openai", "chromadb",
    "pypdf", "sentence_transformers", "openai", "google.generativeai",
    "dotenv", "tiktoken", "rank_bm25",
]

missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"Missing packages ({len(missing)}): {', '.join(missing)}")
    print("Run: pip install -r requirements-dev.txt")
else:
    print(f"All {len(REQUIRED)} required packages are installed ✓")
