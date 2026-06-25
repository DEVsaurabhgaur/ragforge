# Changelog

## [2.0.1] - 2026-06-25
### Fixed
- CI pipeline: added `langchain-classic` and `rank_bm25` to dependencies
- CI pipeline: set `PYTHONPATH` so pytest can resolve local modules
- Added `conftest.py` with shared fixtures and path setup
- Added `setup.cfg` with pytest `pythonpath = .` configuration
- Removed redundant `global _VECTORSTORE_CACHE` declaration (flake8 F824)
- Fixed flake8 to exclude `venv/` directory

## [2.0.0] - 2026-06-25
### Added
- Hybrid BM25 + semantic search retrieval (EnsembleRetriever)
- Conversational context reformulation using chat history
- LLM query expansion for improved recall
- Local keyword reranker (word-overlap scoring)
- Chat session save/load as JSON (`.sessions/`)
- Chat export to Markdown and JSON
- Token counting and cost estimation (tiktoken + pricing tables)
- Statistics dashboard (files, chunks, avg chunk size)
- Progress bar during document ingestion
- Keyword highlighting in retrieved source snippets
- Diagnostics log console panel
- TXT and Markdown file format support
- Hallucination guard post-generation validator
- Dynamic sidebar: temperature, top_k, chunk size, retrieval mode
- System prompt presets (Strict Q&A, Explainer, Bullet Summary)
- Document download buttons in sidebar
- Dockerfile with HEALTHCHECK
- docker-compose.yml
- GitHub Actions CI workflow
- pytest unit test suites (test_pipeline, test_utils, test_config, test_rag_pipeline)
- setup.cfg and conftest.py for test configuration
- pyproject.toml with Black and isort settings
- .flake8 config file
- CONTRIBUTING.md, FAQ.md, ARCHITECTURE.md, DEPLOYMENT.md, SECURITY.md
- scripts/reset_db.py, ingest.py, export_sessions.py, validate_env.py, check_deps.py
- examples/sample_query.py

## [1.0.0] - 2026-06-24
### Added
- Initial RAGForge release: PDF upload, ChromaDB, Gemini/OpenAI Q&A
