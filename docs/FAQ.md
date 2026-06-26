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
