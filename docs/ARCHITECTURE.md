# RAGForge — Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)                      │
│  ┌──────────┐  ┌────────────────┐  ┌──────────────────────────┐ │
│  │ Sidebar  │  │  Hero Header   │  │      Chat Interface      │ │
│  │  Config  │  │  Status Badge  │  │  Messages + Citations    │ │
│  │ Upload   │  │                │  │  Token Metrics           │ │
│  │ Session  │  └────────────────┘  └──────────────────────────┘ │
│  │ Stats    │                                                     │
│  └──────────┘                                                     │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RAG Pipeline (rag_pipeline.py)                 │
│                                                                   │
│  1. Question Reformulation (chat history → standalone query)     │
│  2. Retrieval                                                    │
│     ├── Hybrid: BM25 (rank_bm25) + ChromaDB (EnsembleRetriever) │
│     └── Semantic-Only: ChromaDB similarity search               │
│  3. Query Expansion (optional LLM sub-query generation)          │
│  4. Reranking (keyword-overlap scorer, phrase match boost)       │
│  5. Context Assembly (source-tagged numbered chunks)             │
│  6. LLM Generation (Gemini 2.5 Flash / GPT-4o-mini)             │
│  7. Hallucination Guard (refusal-phrase validator)               │
│  8. Token Counting + Cost Estimation (tiktoken)                  │
└─────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌────────────────────┐
│  ChromaDB        │          │   LLM Providers    │
│  (local embed)   │          │  ┌─────────────┐  │
│  ./chroma_db/    │          │  │Google Gemini│  │
│                  │          │  │  2.5 Flash  │  │
│  Embeddings:     │          │  └─────────────┘  │
│  HuggingFace     │          │  ┌─────────────┐  │
│  all-MiniLM-L6v2 │          │  │  GPT-4o-mini│  │
│  OR OpenAI Ada   │          │  └─────────────┘  │
└─────────────────┘          └────────────────────┘
```

## Component Descriptions

### `app.py` — Streamlit UI
- **Hero Header**: Gradient animated header with live status badge
- **Sidebar**: Config panel (LLM, retrieval mode, temperature, top-K, chunking)
- **Document Uploader**: Multi-file with progress bar, size validation
- **Chat Interface**: Multi-turn Q&A with source citations and token metrics
- **Session Manager**: Save/load/export chat sessions as JSON
- **Diagnostics Console**: Live event log panel

### `rag_pipeline.py` — Core RAG Logic
- `get_embeddings()` — Selects embedding model (local HuggingFace or OpenAI)
- `get_llm()` — Selects generative model (Gemini or OpenAI)
- `load_and_split_document()` — Loads PDF/TXT/MD and chunks with enriched metadata
- `build_vectorstore()` — Embeds documents into ChromaDB
- `get_hybrid_retriever()` — EnsembleRetriever (BM25 + Chroma, 50/50 weights)
- `rerank_documents()` — Keyword-overlap + phrase-match scoring
- `expand_query()` — LLM-powered sub-query generation
- `reformulate_question()` — Conversational context reformulation
- `validate_context_constraints()` — Hallucination guard (refusal phrase detection)
- `query_rag()` — Full pipeline orchestrator

### `utils.py` — Utility Helpers
- `clean_text()` — PDF artifact removal, whitespace normalization
- `truncate_text()` — Word-boundary-aware truncation
- `highlight_keywords()` — Safe HTML keyword highlighting (XSS-safe)
- `count_tokens()` — tiktoken-based token counting with fallback
- `estimate_cost()` — USD cost estimation (Gemini + OpenAI pricing)
- `word_count()` — Word count helper
- `sanitize_filename()` — Safe filename generator
- `get_file_size_mb()` — File size in MB

### `config.py` — Configuration
- LLM/embedding provider selection via environment variables
- Chunking parameters (chunk size, overlap)
- System prompt presets (5 built-in: Strict Q&A, Explainer, Bullet, Technical, ELI5)
- Storage paths and application metadata constants

## Data Flow

```
User Question
     │
     ▼
reformulate_question()  ← chat_history
     │
     ▼
get_hybrid_retriever() or semantic retriever
     │
     ▼ (optional)
expand_query() → multiple sub-queries
     │
     ▼
ChromaDB + BM25 retrieval
     │
     ▼
rerank_documents()
     │
     ▼
Context assembly (with source tags)
     │
     ▼
LLM(system_prompt + context + question)
     │
     ▼
validate_context_constraints()
     │
     ▼
Response + sources + metrics → Streamlit UI
```

## Storage Layout

```
ragforge/
├── app.py                  # Streamlit UI
├── rag_pipeline.py         # RAG engine
├── config.py               # Configuration
├── utils.py                # Helpers
├── chroma_db/              # ChromaDB vector store (auto-created)
├── uploaded_docs/          # Temporary document storage
├── .sessions/              # Chat session JSON files
├── tests/                  # pytest unit tests
│   ├── test_utils.py
│   ├── test_new_utils.py
│   ├── test_config.py
│   ├── test_pipeline.py
│   └── test_rag_pipeline.py
├── scripts/                # Utility CLI scripts
├── docs/                   # Documentation
└── .github/workflows/      # CI/CD
```

## 📐 Module Relationship and Design Notes

1. **Loose Coupling**: The pipeline logic in `rag_pipeline.py` is independent of the Streamlit UI in `app.py`.

2. **State Management**: Streamlit `session_state` is used to persist chat history, loaded files, and configurations.

3. **ChromaDB Backend**: ChromaDB stores document chunks in local SQLite format, avoiding overhead of external databases.

4. **Ensemble Retrieval**: BM25 keyword matching runs in parallel with semantic search to guarantee precision.

5. **Reranker Pipeline**: The custom reranker ranks high-recall documents based on overlapping keywords and phrases.

6. **Hallucination Validator**: Rejects response structures that match pre-defined rejection expressions to ensure safety.

7. **Re-entrancy in config**: `config.py` validates values at startup, throwing errors early if variables are invalid.

8. **Resource Isolation**: Uploaded files are moved to `uploaded_docs` for indexing, isolating system folders.
