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

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

if missing:
    print(f"{RED}✗ Missing packages ({len(missing)}): {', '.join(missing)}{RESET}")
    print(f"Run: {RED}pip install -r requirements-dev.txt{RESET}")
else:
    print(f"{GREEN}✓ All {len(REQUIRED)} required packages are installed{RESET}")
