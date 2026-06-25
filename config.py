import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Settings ──────────────────────────────────────────────
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini')       # 'openai' | 'gemini'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
LLM_MODEL_OPENAI = 'gpt-4o-mini'
LLM_MODEL_GEMINI = 'gemini-2.5-flash'

# ── Embedding Settings ─────────────────────────────────────────
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'local')  # 'openai' | 'local'
EMBEDDING_MODEL_OPENAI = 'text-embedding-3-small'
EMBEDDING_MODEL_LOCAL = 'all-MiniLM-L6-v2'               # free, no API key needed

# ── Chunking Settings ──────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Retrieval Settings ─────────────────────────────────────────
TOP_K_RESULTS = 4
RETRIEVAL_MODE = 'hybrid'  # 'semantic' | 'hybrid'

# ── LLM Generation Settings ────────────────────────────────────
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers questions based ONLY on the provided document context.\nIf the answer is not in the context, say exactly: \"I could not find this in the uploaded documents.\"\nDo NOT use any outside knowledge or make things up."

# Preset System Prompts
SYSTEM_PRESETS = {
    "Strict Q&A": "You are a precise assistant. Answer the user's question using ONLY the provided context. If the context does not contain the answer, reply exactly: \"I could not find this in the uploaded documents.\" Do not synthesize outside facts.",
    "Detailed Explainer": "You are a thorough educational assistant. Answer using the context in a detailed, structured, step-by-step manner. Include citations. Do not make up facts outside the context.",
    "Bullet Summary": "You are a summarization bot. Answer the question using concise bullet points strictly derived from the context. Keep it short."
}

# ── Storage & Paths ────────────────────────────────────────────
CHROMA_DB_DIR = './chroma_db'
UPLOAD_DIR = './uploaded_docs'
SESSION_DIR = './.sessions'
COLLECTION_NAME = 'ragforge_docs'
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.md']

