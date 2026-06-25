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

# ── Storage ────────────────────────────────────────────────────
CHROMA_DB_DIR = './chroma_db'
UPLOAD_DIR = './uploaded_docs'
COLLECTION_NAME = 'ragforge_docs'
