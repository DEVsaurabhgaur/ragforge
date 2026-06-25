"""
scripts/ingest.py — CLI document ingestion without the Streamlit UI.
Usage: python scripts/ingest.py path/to/doc1.pdf path/to/doc2.txt
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_pipeline import build_vectorstore


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python scripts/ingest.py <file1> [file2 ...]")
        sys.exit(1)

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"Files not found: {missing}")
        sys.exit(1)

    print(f"Ingesting {len(paths)} document(s)...")
    vs = build_vectorstore(paths)
    count = len(vs.get()["ids"])
    print(f"Done! {count} chunks stored in vectorstore.")


if __name__ == "__main__":
    main()
