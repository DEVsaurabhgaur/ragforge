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

Upload any PDF documents and ask questions in natural language.  
RAGForge retrieves the most relevant passages and generates grounded answers with source citations.

## Demo

![Demo GIF](assets/demo.gif)

## Features

- 📄 Multi-PDF upload and processing
- 🔎 Semantic similarity search (ChromaDB + embeddings)
- 💬 Conversational Q&A with chat history
- 📌 Source citations with file name + page number
- 🤖 OpenAI GPT-4o-mini OR Google Gemini 1.5 Flash (free)
- 💸 Fully free mode: Gemini LLM + local HuggingFace embeddings
- 💾 Persistent vectorstore — reload previous sessions

## Tech Stack

| Layer | Tool |
|---|---|
| RAG Framework | LangChain 0.3 |
| Vector Store | ChromaDB (persistent, local) |
| Embeddings | OpenAI text-embedding-3-small / HuggingFace MiniLM |
| LLM | GPT-4o-mini / Gemini 1.5 Flash |
| UI | Streamlit |
| PDF Loader | pypdf |

## Quick Start

```bash
git clone https://github.com/DEVsaurabhgaur/ragforge.git
cd ragforge

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

copy .env.example .env       # Windows
# cp .env.example .env       # Mac/Linux
# Add your Gemini or OpenAI key inside .env

streamlit run app.py
```

App opens at http://localhost:8501

## Free Setup (No Credit Card)

1. Get a free Gemini API key: https://aistudio.google.com/app/apikey
2. In `.env` set:
   ```
   LLM_PROVIDER=gemini
   EMBEDDING_PROVIDER=local
   GEMINI_API_KEY=your-key-here
   ```
3. Run — no OpenAI account needed!

## Architecture

```
PDF → PyPDF Loader → RecursiveCharacterTextSplitter (1000/200)
    → Embeddings → ChromaDB (persistent)

Query → Semantic Search → Top-4 Chunks → Prompt Template
      → LLM (Gemini / GPT) → Answer + Sources
```

## Resume Bullets

```
• Built RAGForge — a production-grade RAG system using LangChain, ChromaDB,
  and OpenAI/Gemini; deployed on HuggingFace Spaces

• Implemented end-to-end RAG pipeline: PDF ingestion → recursive chunking →
  vector embedding → semantic retrieval → LLM synthesis with source citation

• Designed multi-LLM support (GPT-4o-mini / Gemini 1.5 Flash) and dual embedding
  modes (OpenAI / local HuggingFace) for cost flexibility
```

## License

MIT — Built by [Saurabh Gaur](https://github.com/DEVsaurabhgaur)
