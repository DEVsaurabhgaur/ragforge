# Changelog

## [2.1.0] - 2026-06-26
### Added
- Google Inter font via `@import` for premium typography across all UI elements
- Animated gradient hero header with radial glow and pulsing status badge
- CSS `fadeSlideIn` animation on all chat messages for smooth rendering
- Glassmorphism stats cards with `backdrop-filter: blur` and hover effects
- Hover lift effect on source citation cards with gradient background
- Custom styled scrollbar (thin indigo track) in logs box and main area
- Gradient progress bar (indigo → violet) during document ingestion
- Source card hover animation (`translateY(-2px)` + glow shadow)
- Chat input border glow on focus with indigo ring
- Sidebar branding overhaul: gradient logo, version chip, message counter
- `hero-badge` with animated pulsing dot for document status
- Upgraded button hover: cubic-bezier easing + stronger box-shadow
- `stats-label` uppercase micro-typography above stats values
- Two new system prompt presets: `Technical Analyst` and `ELI5 Explainer`
- `APP_VERSION`, `APP_NAME`, `MAX_FILE_SIZE_MB`, `MAX_DOCUMENTS` config constants
- `.docx` added to `SUPPORTED_EXTENSIONS` constant (UI support pending loader)
- Three new utility functions: `word_count`, `sanitize_filename`, `get_file_size_mb`
- Explicit `clear_vectorstore_cache()` function in `rag_pipeline.py`
- Expanded hallucination guard with 5 additional refusal phrase patterns
- `test_new_utils.py`: 12 new unit tests covering new utility helpers
- Extended `test_config.py` with tests for new constants and preset keys
- Streamlit config: added XSRF protection, CORS off, usage stats opt-out
- Improved `validate_context_constraints` docstring with return value semantics
- Message count display in sidebar footer (questions + total messages)

### Changed
- Stats cards now use semantic `stats-label` + `stats-val` structure for accessibility
- `_VECTORSTORE_CACHE` annotated with explicit `dict` type hint
- Sidebar caption updated from v2.0 → v2.1
- `logs-box` upgraded to monospace JetBrains Mono / Fira Code font stack
- `logs-box` max-height increased from 200px to 220px

### Fixed
- Source card CSS: removed conflicting duplicate `border-left` declaration

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
