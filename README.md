---
title: RAGForge
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: true
---

# 🔍 RAGForge — Intelligent Multi-Document Q&A

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![CI](https://github.com/DEVsaurabhgaur/ragforge/actions/workflows/ci.yml/badge.svg)](https://github.com/DEVsaurabhgaur/ragforge/actions)

Upload PDF, Plain Text, or Markdown documents and ask questions in natural language. RAGForge processes, chunks, embeds, retrieves, and reformulates conversational queries using advanced hybrid keyword-semantic search, returning precise answers with source citations.

---

## 🌟 Advanced Features

RAGForge has been enhanced with 30 production-grade updates, including:

1.  **🔀 Hybrid Search Retrieval**: Combines BM25 keyword frequency and Chroma vector similarity searches via LangChain's `EnsembleRetriever` for highly accurate keyword + conceptual queries.
2.  **📋 Dynamic Settings Sidebar**: Real-time tuning of Temperature, Top K, Chunk Size, Chunk Overlap, Retrieval Mode, and Custom System Prompts.
3.  **🧠 Conversational Context Reformulation**: Contextual query reformulation rewriting follow-up queries based on chat history.
4.  **📊 Statistics Dashboard**: Shows ingestion details such as total files, total chunks, and average character chunk sizes.
5.  **💾 Chat Session Saver**: Save conversations to disk (`.sessions/*.json`) and restore previous conversations directly from the sidebar.
6.  **📝 Export History**: Download chat sessions directly in Markdown or JSON format.
7.  **💰 Cost & Token Tracker**: Calculates input/output tokens using `tiktoken` and calculates estimated costs in USD for every response.
8.  **⚡ Ingestion Progress**: Live page-by-page progress bar indicator during file uploads.
9.  **🖍️ Source Keyword Highlight**: Highlights relevant query terms inside the matching retrieved snippets using safe HTML tags.
10. **🛠️ Diagnostics Log Console**: View pipeline events and execution steps directly inside a collapsible dashboard terminal.
11. **🧪 Pytest Suite**: Integrated unit testing to verify text loaders, splitters, query expansions, and local embedding retrieval.
12. **🐳 Containerized Deployments**: Easy container packaging using multi-stage `Dockerfile`.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/DEVsaurabhgaur/ragforge.git
# Enter directory
cd ragforge

# Initialize virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies & test packages
pip install -r requirements.txt
pip install pytest flake8

# Configure your keys
copy .env.example .env       # Windows
# cp .env.example .env       # Mac/Linux
# Add your Gemini or OpenAI API keys inside .env

# Run unit tests
pytest tests/

# Launch app
streamlit run app.py
```

App opens at http://localhost:8501

---

## Architecture

```
Ingestion:
[PDF/TXT/MD] ──> Loader ──> Text Splitter (Size: 200-2000)
                              ├──> Embeddings ──> ChromaDB (Semantic)
                              └──> BM25 Matrix ──> Keyword index

Query:
[User Query] ──> Conversational Reformulator (using History)
                              │
                              ├──> [Expanded Queries (Optional)]
                              │
                              ├──> Retrieval (Chroma Similarity + BM25 Ensemble)
                              │
                              ├──> Local Keyword Reranker
                              │
                              └──> LLM Synthesis ──> Answer + Token Tracking + Highlights
```

---

## Developer Commands (Makefile)

The workspace includes a `Makefile` for streamlined development tasks:
- `make install`: Installs required packages.
- `make run`: Starts the local Streamlit development server.
- `make test`: Runs `pytest` on tests suite.
- `make lint`: Performs code syntactical checks using `flake8`.
- `make clean`: Removes local cache directories, session logs, and DB files.

---

## Docker Execution

To build and run RAGForge inside Docker:

```bash
# Build docker image
docker build -t ragforge:latest .

# Run container
docker run -p 8501:8501 --env-file .env ragforge:latest
```

---

## License

MIT — Built by [Saurabh Gaur](https://github.com/DEVsaurabhgaur)

