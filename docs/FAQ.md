# Frequently Asked Questions

### Q: Can I use RAGForge without an API key?
Yes! Set `LLM_PROVIDER=gemini` and `EMBEDDING_PROVIDER=local` in `.env`.
Get a free Gemini key at https://aistudio.google.com/app/apikey

### Q: What file formats are supported?
PDF (`.pdf`), Plain Text (`.txt`), and Markdown (`.md`).

### Q: How do I reset the vectorstore?
Run `python scripts/reset_db.py` or click **Reset Application** in the sidebar.

### Q: How does Hybrid Search work?
RAGForge combines BM25 keyword frequency scoring with ChromaDB vector similarity
using LangChain's EnsembleRetriever (50/50 weight blend).

### Q: Can I save and resume conversations?
Yes — conversations are auto-saved to `.sessions/` as JSON and can be reloaded
from the sidebar session picker.

### Q: How is token cost calculated?
Using tiktoken for precise counts and published pricing for GPT-4o-mini / Gemini 2.5 Flash.
