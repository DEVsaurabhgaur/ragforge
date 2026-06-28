# Frequently Asked Questions

### Q: Can I use RAGForge without an API key?
Yes! Set `LLM_PROVIDER=gemini` and `EMBEDDING_PROVIDER=local` in `.env`.
Get a free Gemini key at https://aistudio.google.com/app/apikey

### Q: What file formats are supported?
PDF (`.pdf`), Plain Text (`.txt`), and Markdown (`.md`).
DOCX support (`.docx`) is tracked in [config.SUPPORTED_EXTENSIONS](../config.py) and will be
added in a future release with a dedicated `python-docx` loader.

### Q: How do I reset the vectorstore?
Run `python scripts/reset_db.py` or click **Reset Application** in the sidebar.

### Q: How does Hybrid Search work?
RAGForge combines BM25 keyword frequency scoring with ChromaDB vector similarity
using LangChain's `EnsembleRetriever` (50/50 weight blend). This catches both
exact keyword hits and semantically similar passages.

### Q: Can I save and resume conversations?
Yes — conversations are auto-saved to `.sessions/` as JSON and can be reloaded
from the sidebar session picker.

### Q: How is token cost calculated?
Using `tiktoken` for precise counts and published pricing for GPT-4o-mini / Gemini 2.5 Flash.

### Q: What are System Prompt Presets?
RAGForge ships with 5 built-in presets you can switch between in the sidebar:
- **Strict Q&A** — Only answers from context, refuses outside knowledge
- **Detailed Explainer** — Step-by-step structured answers with citations
- **Bullet Summary** — Short bullet-point format
- **Technical Analyst** — Precise, structured answers with figures/code references
- **ELI5 Explainer** — Simple, beginner-friendly explanations using analogies

### Q: How do I increase the number of retrieved chunks?
Adjust **Retrieve Chunks (Top K)** in the sidebar's ⚙️ RAG Hyperparameters panel.

### Q: What is Query Expansion?
When enabled, the LLM generates 2 alternative phrasings of your query to improve
recall — useful when your question may be phrased differently from the document text.

### Q: Is my data sent to the cloud?
Only if you use `LLM_PROVIDER=openai` or `LLM_PROVIDER=gemini`. With `EMBEDDING_PROVIDER=local`,
embeddings are computed entirely on your machine using HuggingFace `all-MiniLM-L6-v2`.

### Q: How does the Hallucination Guard work?
After the LLM responds, RAGForge checks the answer for known refusal phrases
(e.g., "not in the provided context", "outside the scope of the documents").
If detected, it replaces the response with a standard safe fallback message.

## 💡 Developer Tips & Troubleshooting

1. **Windows File Locking**: ChromaDB files might get locked by active streamlit processes on Windows. Restart streamlit to unlock.

2. **API Keys**: Ensure your Gemini key is updated; expired keys return 400 Bad Request error.

3. **Local Embeddings RAM**: The `all-MiniLM-L6-v2` model is cached locally in `~/.cache/torch` or similar directory.

4. **Custom Storage Directories**: You can override the ChromaDB and upload directories by editing `config.py` or `.env` variables.

5. **Session File Cleanup**: Old json sessions in `.sessions/` can be deleted manually to free up disk space.

6. **Tiktoken Encoding**: For non-OpenAI models, a fallback char-length heuristic is used to avoid Tiktoken load overhead.

7. **Streamlit Port**: Streamlit defaults to port 8501, but will auto-increment to 8502, 8503 if 8501 is busy.

8. **Python version**: RAGForge is optimized for Python 3.10+; running on 3.8 or 3.9 may require manual dependency adjustments.

9. **Docker Resource Limits**: Assign at least 2GB of memory to Docker containers to prevent PyTorch crash during model loading.

10. **Custom Chunk Size**: Ensure chunk overlap is always strictly smaller than chunk size to prevent ingestion looping.

11. **Keyword Overlap Reranking**: Case sensitivity is stripped out during keyword matching to optimize overlap retrieval.

12. **Hallucination Guard Phrase**: Standard refusal phrases can be expanded in `rag_pipeline.py` if custom LLMs are used.

13. **Diagnostics Logs**: Access the diagnostics foldout in Streamlit UI for API call latencies and vectorstore status.

14. **Custom PDF parsing**: For scanned PDFs, verify that PyPDF extracts characters; scanned-image PDFs require OCR.

15. **HuggingFace Embedding Config**: Check your offline network settings if local HuggingFace downloads fail to initiate.
