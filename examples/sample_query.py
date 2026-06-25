"""
examples/sample_query.py — Minimal programmatic RAGForge usage example.
"""
import sys
sys.path.insert(0, '..')

from rag_pipeline import load_existing_vectorstore, query_rag, vectorstore_exists

def main():
    if not vectorstore_exists():
        print("No vectorstore found. Please upload and process documents via the UI first.")
        return

    vs = load_existing_vectorstore()
    result = query_rag(
        question="What is the main topic of the document?",
        vectorstore=vs,
        retrieval_mode="hybrid",
        k_results=4,
        temperature=0.3,
    )
    print("Answer:", result["answer"])
    print("Sources:", [s.metadata.get("source_file") for s in result["sources"]])
    print("Tokens — In:", result["metrics"]["input_tokens"],
          "Out:", result["metrics"]["output_tokens"],
          "Cost: $", round(result["metrics"]["cost"], 6))

if __name__ == "__main__":
    main()
