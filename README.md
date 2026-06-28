<div align="center">

# 🔍 RAGForge

### Intelligent Multi-Document Q&A System powered by RAG

*Chat with your PDFs, docs, and notes using AI — locally, privately, and for free.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-F97316?style=for-the-badge)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

[![CI](https://github.com/DEVsaurabhgaur/ragforge/actions/workflows/ci.yml/badge.svg)](https://github.com/DEVsaurabhgaur/ragforge/actions)
[![GitHub Stars](https://img.shields.io/github/stars/DEVsaurabhgaur/ragforge?style=social)](https://github.com/DEVsaurabhgaur/ragforge/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/DEVsaurabhgaur/ragforge?style=social)](https://github.com/DEVsaurabhgaur/ragforge/network/members)

<br/>

> **Upload any document → Ask anything → Get precise answers with source citations.**
>
> Works free with Google Gemini API (no credit card required) and local HuggingFace embeddings.

</div>

---

## 📌 What is RAGForge?

**RAGForge** is a fully local, open-source **Retrieval-Augmented Generation (RAG)** application. It lets you upload multiple documents — PDFs, text files, or Markdown — and have a multi-turn AI conversation about their contents.

Unlike generic chatbots, RAGForge **strictly grounds every answer in your documents**, shows you the exact source pages it used, tracks token costs, and never makes things up.

Built with **LangChain**, **ChromaDB**, **Google Gemini / OpenAI**, and **Streamlit**.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🔀 **Hybrid Search** | BM25 keyword + ChromaDB vector similarity via `EnsembleRetriever` |
| 🧠 **Conversational AI** | Follow-up questions understood using full chat history context |
| 🔍 **Query Expansion** | LLM generates sub-queries to improve document recall |
| 📊 **Reranking** | Local keyword-overlap reranker prioritises best retrieved chunks |
| 🛡️ **Hallucination Guard** | Post-generation validator rejects out-of-context answers |
| 💰 **Cost & Token Tracker** | Live token count + USD cost estimate per response (tiktoken) |
| 💾 **Session Save/Load** | Persist and restore full chat sessions as JSON files |
| 📝 **Export Chat** | Download conversations in Markdown or JSON format |
| 📎 **Source Citations** | Every answer shows exact file name + page number |
| 🖍️ **Keyword Highlight** | Query terms highlighted in retrieved snippets |
| ⚡ **Live Ingestion Bar** | Real-time progress bar during document embedding |
| 🎛️ **Dynamic Sidebar** | Tune Temperature, Top-K, Chunk Size, Retrieval Mode live |
| 💬 **System Prompt Presets** | Choose between Strict Q&A, Explainer, and Bullet Summary modes |
| 🐳 **Docker Ready** | One-command container deployment with health checks |
| 🧪 **Full Test Suite** | Pytest unit tests for pipeline, utils, config, and retrieval |
| 🛠️ **Diagnostics Console** | Collapsible pipeline event log at the bottom of the UI |

---

## 🏗️ Architecture

```
┌─────────────────── INGESTION PIPELINE ───────────────────────┐
│                                                               │
│  [PDF / TXT / MD]                                             │
│       │                                                       │
│       ▼                                                       │
│  Document Loader  ──►  Text Splitter (200–2000 chars)        │
│                              │                                │
│                   ┌──────────┴──────────┐                     │
│                   ▼                     ▼                     │
│           HuggingFace / OpenAI     BM25 Matrix                │
│             Embeddings             (Keyword Index)            │
│                   │                                           │
│                   ▼                                           │
│              ChromaDB (persisted on disk)                     │
└───────────────────────────────────────────────────────────────┘

┌─────────────────── QUERY PIPELINE ────────────────────────────┐
│                                                               │
│  [User Question]                                              │
│       │                                                       │
│       ▼                                                       │
│  Conversational Reformulator  (uses Chat History)             │
│       │                                                       │
│       ▼                                                       │
│  [Optional] LLM Query Expansion  →  2 Sub-queries            │
│       │                                                       │
│       ▼                                                       │
│  EnsembleRetriever                                            │
│    ├── ChromaDB Semantic Search (vector similarity)           │
│    └── BM25 Keyword Search      (term frequency)             │
│       │                                                       │
│       ▼                                                       │
│  Local Keyword Reranker  (word-overlap scoring)              │
│       │                                                       │
│       ▼                                                       │
│  LLM Synthesis  (Gemini 2.5 Flash / GPT-4o-mini)             │
│       │                                                       │
│       ▼                                                       │
│  Hallucination Guard  →  Answer + Citations + Token Cost     │
└───────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A **free** [Google Gemini API key](https://aistudio.google.com/app/apikey) *(no credit card)*

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/DEVsaurabhgaur/ragforge.git
cd ragforge

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
copy .env.example .env       # Windows
# cp .env.example .env       # Mac / Linux
```

Edit `.env`:
```env
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=local        # FREE — uses local HuggingFace model
GEMINI_API_KEY=your_key_here    # Free from aistudio.google.com
```

```bash
# 5. Launch the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser. 🎉

---

## 🆓 Zero-Cost Setup

RAGForge can run **completely free** with no credit card:

| Component | Free Option |
|---|---|
| LLM | Google Gemini 2.5 Flash (free tier) |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace (local, no API) |
| Vector DB | ChromaDB (local, on-disk) |
| App | Streamlit (open-source) |

---

## 🐳 Docker Deployment

```bash
# Build and run
docker build -t ragforge:latest .
docker run -p 8501:8501 --env-file .env ragforge:latest

# Or with Docker Compose
docker-compose up
```

---

## 🧪 Testing

```bash
# Run full test suite
pytest tests/ -v

# Lint check
flake8 .

# Or use Makefile shortcuts
make test
make lint
make run
```

---

## 📁 Project Structure

```
ragforge/
├── app.py                  # Main Streamlit UI
├── rag_pipeline.py         # Core RAG logic (embed, retrieve, query)
├── config.py               # Centralised configuration
├── utils.py                # Helper functions
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── docker-compose.yml      # Compose file
├── Makefile                # Dev shortcuts
│
├── .streamlit/
│   └── config.toml         # Streamlit dark theme config
│
├── tests/                  # Pytest unit tests
│   ├── test_pipeline.py
│   ├── test_utils.py
│   ├── test_db_stats.py
│   └── test_config.py
│
├── scripts/                # Utility scripts
│   ├── ingest.py           # CLI document ingestion
│   ├── reset_db.py         # Wipe ChromaDB
│   ├── db_stats.py         # Inspect ChromaDB collection contents
│   ├── validate_env.py     # Check environment config
│   ├── export_sessions.py  # Export all chat sessions
│   └── check_deps.py       # Verify installed packages
│
└── docs/                   # Extended documentation
    ├── ARCHITECTURE.md
    ├── CONTRIBUTING.md
    ├── DEPLOYMENT.md
    └── FAQ.md
```

---

## ⚙️ Configuration Reference

All settings live in `.env` and can also be tuned live via the sidebar:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `EMBEDDING_PROVIDER` | `local` | `local` (free) or `openai` |
| `GEMINI_API_KEY` | — | Google AI Studio key |
| `OPENAI_API_KEY` | — | OpenAI platform key |

**Sidebar controls (live, no restart needed):**
- Temperature (0.0 → 1.0)
- Top-K retrieved chunks (1 → 10)
- Chunk size (200 → 2000 chars)
- Chunk overlap
- Retrieval Mode (`hybrid` / `semantic`)
- Query Expansion toggle
- System Prompt Preset

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **RAG Framework** | LangChain 0.3 |
| **Vector Database** | ChromaDB |
| **LLM** | Google Gemini 2.5 Flash / OpenAI GPT-4o-mini |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` / OpenAI `text-embedding-3-small` |
| **Keyword Search** | BM25 (rank_bm25) |
| **PDF Parsing** | PyPDF |
| **Token Counting** | tiktoken |
| **Testing** | Pytest |
| **Containerisation** | Docker |
| **CI/CD** | GitHub Actions |

---

## 🗺️ Roadmap

- [ ] Web URL ingestion (scrape and embed web pages)
- [ ] Multi-user session isolation
- [ ] Reranking with cross-encoder models (e.g., `ms-marco-MiniLM`)
- [ ] Streaming LLM responses
- [ ] Hugging Face Spaces deployment guide
- [ ] DOCX / Excel / CSV file support

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) first.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push and open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by [Saurabh Gaur](https://github.com/DEVsaurabhgaur)**

*If this project helped you, please ⭐ star the repo — it means a lot!*

[![GitHub](https://img.shields.io/badge/GitHub-DEVsaurabhgaur-181717?style=for-the-badge&logo=github)](https://github.com/DEVsaurabhgaur)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/saurabhgaur)

</div>

## ⚡ Quick Project Reference Notes

* **License**: RAGForge is licensed under the MIT License.

* **Python Requirements**: Supports Python versions 3.10 and newer.

* **API Key Free Tier**: Supports local HuggingFace embeddings out of the box.

* **Docker Execution**: Run `docker-compose up -d` for single-command start.

* **UI Configuration**: Tweak temperatures, retrieval models, and system prompts in real-time.
