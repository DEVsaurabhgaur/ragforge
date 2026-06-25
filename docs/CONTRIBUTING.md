# Contributing to RAGForge

Thank you for your interest in contributing!

## Setup

```bash
git clone https://github.com/DEVsaurabhgaur/ragforge.git
cd ragforge
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
pip install langchain-classic rank_bm25 pytest flake8
```

## Running Tests

```bash
pytest tests/ -v
```

## Linting

```bash
flake8 . --exclude=venv --select=E9,F63,F7,F82
```

## Submitting a PR

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit changes with clear messages
4. Push and open a Pull Request
