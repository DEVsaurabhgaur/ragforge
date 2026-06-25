# Deployment Guide

## Local Development

```bash
streamlit run app.py
```

## Docker

```bash
docker build -t ragforge:latest .
docker run -p 8501:8501 --env-file .env ragforge:latest
```

## Docker Compose

```bash
docker-compose up -d
```

## HuggingFace Spaces

1. Fork or push the repository to HuggingFace.
2. Set `GEMINI_API_KEY` in Space Secrets.
3. The `README.md` frontmatter configures the Space automatically.

## Environment Variables

Set all variables from `.env.example` in your deployment platform's secrets manager.
