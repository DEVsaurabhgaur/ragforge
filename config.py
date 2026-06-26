"""
config.py — Centralised configuration for RAGForge
"""
import os
from dotenv import load_dotenv

load_dotenv()

# â”€â”€ LLM Settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini')       # 'openai' | 'gemini'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
LLM_MODEL_OPENAI = 'gpt-4o-mini'
LLM_MODEL_GEMINI = 'gemini-2.5-flash'

# â”€â”€ Embedding Settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'local')  # 'openai' | 'local'
EMBEDDING_MODEL_OPENAI = 'text-embedding-3-small'
EMBEDDING_MODEL_LOCAL = 'all-MiniLM-L6-v2'               # free, no API key needed

# â”€â”€ Chunking Settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# â”€â”€ Retrieval Settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TOP_K_RESULTS = 4
RETRIEVAL_MODE = 'hybrid'  # 'semantic' | 'hybrid'

# â”€â”€ LLM Generation Settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers questions based ONLY on the provided document context.\nIf the answer is not in the context, say exactly: \"I could not find this in the uploaded documents.\"\nDo NOT use any outside knowledge or make things up."

# Preset System Prompts
SYSTEM_PRESETS = {
    "Strict Q&A": "You are a precise assistant. Answer the user's question using ONLY the provided context. If the context does not contain the answer, reply exactly: \"I could not find this in the uploaded documents.\" Do not synthesize outside facts.",
    "Detailed Explainer": "You are a thorough educational assistant. Answer using the context in a detailed, structured, step-by-step manner. Include citations. Do not make up facts outside the context.",
    "Bullet Summary": "You are a summarization bot. Answer the question using concise bullet points strictly derived from the context. Keep it short.",
    "Technical Analyst": "You are a technical document analyst. Answer with precision, referencing exact figures, tables, code snippets, or technical specifications found in the context. Use structured formatting with headers where appropriate.",
    "ELI5 Explainer": "You are a friendly teacher. Explain the answer in simple, easy-to-understand language as if teaching a beginner. Use analogies and plain English. Base your answer strictly on the provided context.",
}

# â”€â”€ Storage & Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHROMA_DB_DIR = './chroma_db'
UPLOAD_DIR = './uploaded_docs'
SESSION_DIR = './.sessions'
COLLECTION_NAME = 'ragforge_docs'
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.md', '.docx']

# ── Application Metadata ───────────────────────────────────────
APP_VERSION = '2.1.0'
APP_NAME = 'RAGForge'
MAX_FILE_SIZE_MB = 50  # Maximum allowed upload file size in megabytes
MAX_DOCUMENTS = 20    # Maximum number of documents per session
