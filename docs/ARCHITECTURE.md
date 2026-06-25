# RAGForge Architecture

```
┌─────────────────────────────────────────────────┐
│                   Streamlit UI                  │
│  Sidebar: Upload | Settings | Sessions | Stats  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Ingestion Pipeline                 │
│  PDF/TXT/MD → Chunk → Clean → Embed → ChromaDB │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Query Pipeline                     │
│  Question → Reformulate (history)               │
│           → Retrieve (BM25 + Chroma)            │
│           → Rerank (keyword overlap)            │
│           → LLM Synthesis                       │
│           → Token tracking + Source display     │
└─────────────────────────────────────────────────┘
```
