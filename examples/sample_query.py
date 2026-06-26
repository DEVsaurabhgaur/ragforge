"""
examples/sample_query.py — Demonstrates how to use the RAG pipeline programmatically.
"""
import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import config
from rag_pipeline import build_vectorstore, query_rag
from utils import word_count, get_file_size_mb


def main():
    """
    End-to-end example:
    1. Ingest a sample text document
    2. Query the vectorstore with a question
    3. Print the answer, sources, and token metrics
    """
    # ── Step 1: Create a sample document ──
    sample_dir = Path(config.UPLOAD_DIR)
    sample_dir.mkdir(exist_ok=True)

    sample_file = sample_dir / "example_doc.txt"
    sample_file.write_text(
        "RAGForge is an open-source Retrieval-Augmented Generation system.\n"
        "It supports PDF, TXT, and Markdown documents.\n"
        "The system uses ChromaDB for vector storage and LangChain for orchestration.\n"
        "Hybrid search combines BM25 keyword scoring with semantic vector similarity.\n"
        "The hallucination guard validates responses against the source context.\n",
        encoding="utf-8"
    )

    file_size = get_file_size_mb(str(sample_file))
    print(f"\n\u2705 Sample file created: {sample_file.name} ({file_size} MB)")

    # ── Step 2: Build vectorstore ──
    print("\u23f3 Building vectorstore from sample document...")
    try:
        vs = build_vectorstore([str(sample_file)])
        print(f"\u2705 Vectorstore built. Total chunks: {len(vs.get()['ids'])}")
    except Exception as e:
        print(f"\u274c Failed to build vectorstore: {e}")
        print("   Make sure your GEMINI_API_KEY or OPENAI_API_KEY is set in .env")
        return

    # ── Step 3: Query ──
    question = "What is RAGForge and what documents does it support?"
    wc = word_count(question)
    print(f"\n\ud83d\udcac Query ({wc} words): '{question}'")

    try:
        result = query_rag(
            question=question,
            vectorstore=vs,
            chat_history=[],
            retrieval_mode="hybrid",
            k_results=3,
            temperature=0.2,
        )

        print(f"\n\ud83e\udd16 Answer:\n{result['answer']}")

        if result["sources"]:
            print(f"\n\ud83d\udcce Sources ({len(result['sources'])} retrieved):")
            for i, src in enumerate(result["sources"]):
                meta = src.metadata
                print(f"  {i+1}. {meta.get('source_file', 'unknown')} — Page {meta.get('page', '?')}")

        m = result["metrics"]
        print(
            f"\n\ud83d\udcb0 Metrics: {m['input_tokens']} in | {m['output_tokens']} out | "
            f"${m['cost']:.6f} USD"
        )

    except Exception as e:
        print(f"\u274c Query failed: {e}")

    print()


if __name__ == "__main__":
    main()
