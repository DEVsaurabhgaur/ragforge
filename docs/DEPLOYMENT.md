# Deployment Guide

## Prerequisites

- Python 3.10+
- A Gemini API key (free) or OpenAI API key
- Docker (optional, for containerised deployment)

---

## Local Development

```bash
# 1. Clone and set up
git clone https://github.com/DEVsaurabhgaur/ragforge.git
cd ragforge
python -m venv venv && venv\Scripts\activate   # Windows
# OR: source venv/bin/activate                  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Launch
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## Docker

```bash
# Build the image
docker build -t ragforge:latest .

# Run with environment file
docker run -p 8501:8501 --env-file .env ragforge:latest
```

---

## Docker Compose

```bash
# Start (detached)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## HuggingFace Spaces

1. Fork or push the repository to HuggingFace.
2. Set `GEMINI_API_KEY` in **Space Secrets** (Settings → Repository secrets).
3. Set `LLM_PROVIDER=gemini` and `EMBEDDING_PROVIDER=local` in Space secrets.
4. The `README.md` frontmatter configures the Space automatically.

> **Note:** HuggingFace Spaces have a 16 GB RAM limit. The local HuggingFace embedding model
> (`all-MiniLM-L6-v2`) is well within this budget.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in required values:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | If using Gemini | — | Google AI Studio API key |
| `OPENAI_API_KEY` | If using OpenAI | — | OpenAI platform API key |
| `LLM_PROVIDER` | No | `gemini` | `gemini` or `openai` |
| `EMBEDDING_PROVIDER` | No | `local` | `local` or `openai` |
| `RETRIEVAL_MODE` | No | `hybrid` | `hybrid` or `semantic` |
| `CHUNK_SIZE` | No | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | No | `200` | Character overlap between chunks |

Run `python scripts/validate_env.py` to verify your configuration before starting.

---

## Health Check

The Docker image includes a built-in health check that pings `http://localhost:8501/_stcore/health`.
You can monitor container health with:

```bash
docker inspect --format='{{.State.Health.Status}}' <container_id>
```

## 🚀 Production Deployment Considerations

1. **Docker Compose Logs**: Run `docker-compose logs --tail=100 -f` to monitor active user query statuses.

2. **HuggingFace Spaces Cache**: Configure `HF_HOME` environment variable if cache write permissions are restricted on HuggingFace.

3. **Reverse Proxying**: When routing through Nginx or Apache, ensure websocket support is enabled for Streamlit.

4. **Secure API Key Handling**: Never hardcode API keys in `config.py`. Use environment variables or secret managers.

5. **Persistent Storage**: Mount the `chroma_db` folder as a Docker volume to persist index databases across container restarts.

6. **Multiple Users Concurrent Access**: RAGForge uses local locks to avoid multi-write collisions on the SQLite backend of Chroma.

7. **Memory Constraints**: Local embed models load in RAM; scale instances based on expected concurrent uploads.

8. **Health check endpoint**: Monitor `http://localhost:8501/_stcore/health` via Prometheus or other pinging daemons.

9. **Environment Validation**: Run `python scripts/validate_env.py` in your CI/CD pipelines to prevent startup failures.

10. **Disable Streamlit CORS**: Set `CORS = false` in Streamlit configuration if deployment is behind a secure VPN gateway.

11. **Tiktoken offline usage**: Tiktoken fetches files from public URLs during first run. Pre-download them if deploying offline.

12. **Gevent / Eventlet Compatibility**: Streamlit utilizes websockets; avoid event-loop monkeypatching during initialization.

13. **Temp directories cleanups**: Clean up `/tmp` or local `uploaded_docs` via a cron job to avoid disk space exhaustion.
