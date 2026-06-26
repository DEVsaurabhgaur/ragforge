.PHONY: install install-dev run clean format lint test test-cov stats validate docker-build docker-run docker-compose-up docker-compose-down

install:
	pip install -r requirements.txt
	pip install langchain-classic rank_bm25

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install langchain-classic rank_bm25

run:
	streamlit run app.py

clean:
	rm -rf chroma_db/ uploaded_docs/ .sessions/ __pycache__/ .pytest_cache/ .coverage tests/__pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

format:
	black . && isort .

lint:
	flake8 . --exclude=venv --count --select=E9,F63,F7,F82 --show-source --statistics

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

stats:
	python scripts/stats.py

validate:
	python scripts/validate_env.py

sessions:
	python scripts/list_sessions.py

docker-build:
	docker build -t ragforge:latest .

docker-run:
	docker run -p 8501:8501 --env-file .env ragforge:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down
