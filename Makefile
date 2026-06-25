.PHONY: install run clean format lint test

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

clean:
	rm -rf chroma_db/ uploaded_docs/ .sessions/ __pycache__/ .pytest_cache/ .coverage tests/__pycache__

format:
	black . && isort .

lint:
	python -m pip install flake8 && flake8 . --exclude=venv --count --select=E9,F63,F7,F82 --show-source --statistics

test:
	python -m pytest tests/

