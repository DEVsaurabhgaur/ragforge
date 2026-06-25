.PHONY: install run clean format

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

clean:
	rm -rf chroma_db/ uploaded_docs/ __pycache__/ .pytest_cache/

format:
	black . && isort .
