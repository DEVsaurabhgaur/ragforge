# Contributing to RAGForge

Thank you for your interest in contributing to RAGForge! 🎉

## Prerequisites

- Python 3.10+
- Git

## Setup

```bash
git clone https://github.com/DEVsaurabhgaur/ragforge.git
cd ragforge
python -m venv venv && venv\Scripts\activate   # Windows
# OR
python -m venv venv && source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Running Tests

```bash
pytest tests/ -v
```

To run a specific test file:
```bash
pytest tests/test_utils.py -v
pytest tests/test_config.py -v
pytest tests/test_rag_pipeline.py -v
pytest tests/test_new_utils.py -v
```

## Linting

```bash
flake8 . --exclude=venv --select=E9,F63,F7,F82
```

## Code Formatting

RAGForge uses Black for formatting and isort for import sorting (configured in `pyproject.toml`):

```bash
black .
isort .
```

## Submitting a PR

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit changes with clear, descriptive messages following [Conventional Commits](https://www.conventionalcommits.org/)
   - `feat(ui): add dark mode toggle`
   - `fix(pipeline): handle empty PDF pages gracefully`
   - `docs(faq): add question about export formats`
   - `test(utils): add tests for word_count`
4. Ensure all tests pass: `pytest tests/ -v`
5. Push to your fork and open a Pull Request against `main`

## Project Structure

| File/Dir | Purpose |
|---|---|
| `app.py` | Streamlit UI — all page layout and interaction |
| `rag_pipeline.py` | Core RAG logic: load, embed, retrieve, generate |
| `config.py` | Centralised configuration and constants |
| `utils.py` | Shared utility helper functions |
| `tests/` | pytest unit test suite |
| `docs/` | Documentation (FAQ, Architecture, Deployment) |
| `scripts/` | CLI utility scripts |

## Adding a New System Prompt Preset

1. Open `config.py`
2. Add your preset to `SYSTEM_PRESETS` dict
3. Write a clear, instruction-focused prompt
4. Add a test in `tests/test_config.py` to verify it loads correctly

## Reporting Issues

Please open a GitHub issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS

## 🛠️ Code Conventions & Standards Details

1. **Type Annotations**: Always declare input and output types for all public utility and pipeline functions.

2. **Docstrings Style**: Use Google Style Python docstrings with Args, Returns, and Raises sections.

3. **Error Handling**: Use broad except blocks sparingly. Catch specific exceptions (e.g. FileNotFoundError, ValueError).

4. **Logging Levels**: Use `logging.info()` for pipeline progression and `logging.error()` for critical failures.

5. **RegEx Precompilation**: Precompile heavy regular expressions at the module level for performance optimization.

6. **Test Independence**: Every test should utilize temporary directories or mock setups to ensure zero side-effects.

7. **Black Formatting**: Run `black .` to automatically standardise formatting before staging PR commits.

8. **Import Ordering**: Maintain imports grouped by standard library, third-party libraries, and local modules.

9. **Constant Naming**: Define configuration constants in UPPER_CASE inside the centralized `config.py` file.

10. **Conventional Commit Types**: Use `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore` tags in commit titles.

11. **Streamlit UI Components**: Place all user-facing interactive elements inside sidebar or main container cleanly.
