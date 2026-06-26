# Use official lightweight Python image
FROM python:3.11-slim

# Metadata labels
LABEL maintainer="Saurabh Gaur <github.com/DEVsaurabhgaur>"
LABEL org.opencontainers.image.title="RAGForge"
LABEL org.opencontainers.image.description="Intelligent Multi-Document Q&A System powered by RAG"
LABEL org.opencontainers.image.version="2.1.0"
LABEL org.opencontainers.image.source="https://github.com/DEVsaurabhgaur/ragforge"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8501
# Disable HuggingFace telemetry inside the container
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_OFFLINE=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir langchain-classic rank_bm25

# Copy application source
COPY . .

# Create required directories to ensure they exist in the container
RUN mkdir -p chroma_db uploaded_docs .sessions

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
