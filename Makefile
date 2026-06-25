.PHONY: install run clean format lint test docker-build docker-run

install:
\tpip install -r requirements.txt
\tpip install langchain-classic rank_bm25 pytest flake8

run:
\tstreamlit run app.py

clean:
\trm -rf chroma_db/ uploaded_docs/ .sessions/ __pycache__/ .pytest_cache/ .coverage tests/__pycache__

format:
\tblack . && isort .

lint:
\tflake8 .

test:
\tpython -m pytest tests/ -v

docker-build:
\tdocker build -t ragforge:latest .

docker-run:
\tdocker run -p 8501:8501 --env-file .env ragforge:latest
