"""
scripts/db_stats.py — Inspect the ChromaDB collection contents, showing all ingested files and chunk counts.
Usage: python scripts/db_stats.py [--json]
"""
import sys
import os
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from rag_pipeline import load_existing_vectorstore, vectorstore_exists


def main():
    use_json = "--json" in sys.argv

    if not vectorstore_exists():
        if use_json:
            print(json.dumps({"exists": False, "total_chunks": 0, "documents": []}))
        else:
            print("ChromaDB vector store does not exist on disk yet.")
        sys.exit(0)

    try:
        vs = load_existing_vectorstore()
        data = vs.get()
    except Exception as e:
        if use_json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error connecting to ChromaDB: {e}")
        sys.exit(1)

    ids = data.get("ids", [])
    metadatas = data.get("metadatas", []) or []

    doc_counter = Counter()
    for meta in metadatas:
        if meta:
            source = meta.get("source_file", "unknown")
            doc_counter[source] += 1

    total_chunks = len(ids)

    if use_json:
        doc_list = [{"document": doc, "chunks": count} for doc, count in doc_counter.items()]
        out = {
            "exists": True,
            "total_chunks": total_chunks,
            "collection_name": config.COLLECTION_NAME,
            "persist_directory": config.CHROMA_DB_DIR,
            "documents": doc_list
        }
        print(json.dumps(out, indent=2))
    else:
        print("==========================================")
        print("          RAGForge ChromaDB Stats         ")
        print("==========================================")
        print(f"Collection Name:   {config.COLLECTION_NAME}")
        print(f"Persist Directory: {config.CHROMA_DB_DIR}")
        print(f"Total Chunks:      {total_chunks}")
        print("------------------------------------------")
        if doc_counter:
            print("Ingested Documents:")
            for doc, count in doc_counter.items():
                print(f" - {doc}: {count} chunks")
        else:
            print("No documents found in the collection metadata.")
        print("==========================================")


if __name__ == "__main__":
    main()
