"""
app.py — RAGForge Streamlit UI
Run with: streamlit run app.py
"""
import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
import streamlit as st

import config
from rag_pipeline import (
    build_vectorstore,
    load_existing_vectorstore,
    vectorstore_exists,
    query_rag,
)
from utils import (
    ensure_dirs,
    truncate_text,
    highlight_keywords,
    clear_upload_dir,
    sanitize_filename
)

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title='RAGForge — Intelligent Multi-Doc Q&A',
    page_icon='🔍',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Ensure required dirs exist
ensure_dirs()
log_diagnostic(f"Booting system v{config.APP_VERSION} in environment mode.")

# Initialize session state variables
if "logs" not in st.session_state:
    st.session_state["logs"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "doc_stats" not in st.session_state:
    st.session_state["doc_stats"] = {}

if "doc_names" not in st.session_state:
    st.session_state["doc_names"] = []

if "current_session_file" not in st.session_state:
    st.session_state["current_session_file"] = ""


def log_diagnostic(msg: str):
    """Add a diagnostic log entry with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["logs"].append(f"[{timestamp}] {msg}")
    logging.info(msg)


# ── Custom CSS for Rich Aesthetics ─────────────────────────────
st.markdown("""
<style>
    /* Import Google Fonts - Outfit for headers, Inter for body text */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800;900&display=swap');

    /* Global Typography overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #0b0c10 !important;
        color: #e2e8f0 !important;
    }

    h1, h2, h3, h4, h5, h6, .hero-title {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Core Page background mesh */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.05) 0%, transparent 40%),
                    #0a0b10 !important;
        color: #f1f5f9;
    }

    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background-color: rgba(14, 16, 26, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }

    /* ── Gradient Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 40%, rgba(139, 92, 246, 0.1) 0%, transparent 50%);
        animation: pulse-bg 8s ease-in-out infinite alternate;
        pointer-events: none;
    }
    @keyframes pulse-bg {
        0%   { opacity: 0.4; transform: scale(1); }
        100% { opacity: 1;   transform: scale(1.05); }
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a5b4fc, #e0e7ff, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 8px 0;
        letter-spacing: -0.8px;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 20px;
        margin-top: 16px;
        letter-spacing: 0.4px;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.1);
    }
    .hero-badge .dot {
        width: 8px;
        height: 8px;
        background: #818cf8;
        border-radius: 50%;
        box-shadow: 0 0 8px #818cf8;
        animation: ping 1.8s cubic-bezier(0,0,0.2,1) infinite;
    }
    @keyframes ping {
        0%   { transform: scale(1); opacity: 1; }
        75%, 100% { transform: scale(2.2); opacity: 0; }
    }

    /* ── Source cards with subtle borders ── */
    .source-card {
        background: linear-gradient(135deg, rgba(20, 24, 40, 0.6) 0%, rgba(26, 32, 53, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-left: 4px solid #818cf8 !important;
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 14px;
        font-size: 0.9rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    .source-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.3) !important;
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.12);
    }

    .badge {
        background: rgba(129, 140, 248, 0.12);
        color: #a5b4fc;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(129, 140, 248, 0.2);
        letter-spacing: 0.3px;
    }

    /* ── Glassmorphic Stats ── */
    .stats-card {
        background: rgba(22, 26, 45, 0.6);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .stats-card:hover {
        border-color: rgba(99, 102, 241, 0.2);
        background: rgba(22, 26, 45, 0.8);
    }
    .stats-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .stats-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #a5b4fc;
        line-height: 1;
    }

    .metric-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.08);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.15);
        font-size: 0.78rem;
        padding: 3px 8px;
        border-radius: 6px;
        margin-right: 6px;
        font-weight: 500;
    }

    .metric-cost {
        display: inline-block;
        background: rgba(245, 158, 11, 0.08);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.15);
        font-size: 0.78rem;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 500;
    }

    /* ── Logs console styling ── */
    .logs-box {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.80rem;
        background: #05060b !important;
        color: #38bdf8;
        padding: 16px;
        border-radius: 10px;
        max-height: 240px;
        overflow-y: auto;
        border: 1px solid rgba(255, 255, 255, 0.05);
        line-height: 1.6;
    }
    .logs-box::-webkit-scrollbar {
        width: 6px;
    }
    .logs-box::-webkit-scrollbar-track {
        background: #05060b;
    }
    .logs-box::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 3px;
    }

    /* ── Chat styling upgrades ── */
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease;
        animation: fadeSlideIn 0.3s ease-out;
    }
    [data-testid="stChatMessage"]:hover {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border-color: rgba(255, 255, 255, 0.05) !important;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Buttons styling ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(255, 255, 255, 0.02) !important;
    }
    .stButton > button:hover {
        background: rgba(99, 102, 241, 0.1) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #c7d2fe !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6366f1, #7c3aed) !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #818cf8, #8b5cf6) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        border-radius: 6px;
    }

    /* ── Custom App Scrollbar ── */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0a0b10; }
    ::-webkit-scrollbar-thumb { background: #1b1d2a; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #3b4261; }

    /* ── Alert boxes ── */
    .stAlert {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border-radius: 12px !important;
        border-left: 4px solid #6366f1 !important;
        color: #e2e8f0 !important;
    }

    /* ── Streamlit Expander custom look ── */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.01) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        transition: all 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border-color: rgba(255, 255, 255, 0.06) !important;
    }

    /* ── File Upload Zone ── */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        background: rgba(255, 255, 255, 0.01) !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #6366f1 !important;
        background: rgba(99, 102, 241, 0.03) !important;
    }

    /* ── Selectbox and slider inputs ── */
    [data-testid="stSelectbox"] > div:first-child {
        border-radius: 10px !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* ── Chat input area styling ── */
    [data-testid="stChatInput"] {
        background-color: rgba(14, 16, 26, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5) !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #f8fafc !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #6366f1 !important;
    }

    /* ── Dividers ── */
    hr {
        border-color: rgba(255, 255, 255, 0.06) !important;
        margin: 16px 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session Caching Logic ──────────────────────────────────────
def get_saved_sessions():
    """List all saved chat session JSON files."""
    session_files = sorted(Path(config.SESSION_DIR).glob("*.json"), key=os.path.getmtime, reverse=True)
    return [f.name for f in session_files]


def save_current_session(name: str = None):
    """Save st.session_state chat messages to a JSON file."""
    if not st.session_state.messages:
        return
    
    if not name:
        if st.session_state["current_session_file"]:
            filename = st.session_state["current_session_file"]
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_{timestamp}.json"
    else:
        # Sanitize filename
        safe_name = "".join([c for c in name if c.isalnum() or c in [' ', '_', '-']]).strip().replace(' ', '_')
        filename = f"{safe_name}.json"

    file_path = os.path.join(config.SESSION_DIR, filename)
    data = {
        "doc_names": st.session_state.get("doc_names", []),
        "messages": st.session_state.messages,
        "doc_stats": st.session_state.get("doc_stats", {}),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    st.session_state["current_session_file"] = filename
    log_diagnostic(f"Session saved to {filename}")

    # Enforce limit of max 50 saved sessions to avoid disk clutter
    try:
        session_files = sorted(Path(config.SESSION_DIR).glob("*.json"), key=os.path.getmtime)
        if len(session_files) > 50:
            for old_file in session_files[:-50]:
                try:
                    old_file.unlink()
                    log_diagnostic(f"Deleted oldest session file to keep limit: {old_file.name}")
                except Exception as de:
                    logging.warning(f"Failed to delete old session {old_file.name}: {de}")
    except Exception as se:
        logging.warning(f"Failed to prune saved sessions: {se}")


def load_session(filename: str):
    """Load chat history and document context from session file."""
    file_path = os.path.join(config.SESSION_DIR, filename)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            st.session_state.messages = data.get("messages", [])
            st.session_state["doc_names"] = data.get("doc_names", [])
            st.session_state["doc_stats"] = data.get("doc_stats", {})
            st.session_state["current_session_file"] = filename
            
            # Load corresponding vectorstore if it exists
            if vectorstore_exists():
                st.session_state["vectorstore"] = load_existing_vectorstore()
                
            log_diagnostic(f"Successfully loaded session: {filename}")
        except Exception as e:
            st.error(f"Failed to load session: {e}")
            log_diagnostic(f"Error loading session: {e}")


# ── Sidebar Configurations & Dashboards ────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 8px;">
        <div style="font-size:2rem;">🔍</div>
        <div style="font-size:1.2rem; font-weight:800; background:linear-gradient(90deg,#818cf8,#c7d2fe); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; letter-spacing:-0.3px;">RAGForge</div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">Intelligent Multi-Document Q&A</div>
        <div style="display:inline-block; background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.3); color:#818cf8; font-size:0.68rem; padding:2px 10px; border-radius:20px; margin-top:8px; font-weight:600;">v2.1</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Parameters section ──
    with st.expander("⚙️ RAG Hyperparameters", expanded=False):
        # Provider selector
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["gemini", "openai"],
            index=0 if config.LLM_PROVIDER == "gemini" else 1,
            help="Select generative model engine."
        )
        
        # Temp API key visual warnings
        if llm_provider == "gemini" and not config.GEMINI_API_KEY:
            st.warning("⚠️ GEMINI_API_KEY is not set in .env")
        elif llm_provider == "openai" and not config.OPENAI_API_KEY:
            st.warning("⚠️ OPENAI_API_KEY is not set in .env")

        # Show active model
        active_model = config.LLM_MODEL_GEMINI if llm_provider == "gemini" else config.LLM_MODEL_OPENAI
        st.caption(f"🧠 Model: `{active_model}`")

        # Dynamic retrieval mode
        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            options=["hybrid", "semantic"],
            index=0 if config.RETRIEVAL_MODE == "hybrid" else 1,
            help="Hybrid search merges BM25 keyword matching and vector search."
        )

        # Dynamic query expansion
        use_expansion = st.toggle(
            "Query Expansion",
            value=False,
            help="Generates sub-queries to broaden search recall."
        )

        # Temperature
        temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config.DEFAULT_TEMPERATURE,
            step=0.1
        )

        # Top K Results
        top_k = st.slider(
            "Retrieve Chunks (Top K)",
            min_value=1,
            max_value=10,
            value=config.TOP_K_RESULTS,
            step=1,
            help="Number of document chunks to retrieve and pass to the LLM as context. Higher values improve recall but increase cost."
        )

        # Embedding model display
        emb_model = config.EMBEDDING_MODEL_OPENAI if config.EMBEDDING_PROVIDER == "openai" else config.EMBEDDING_MODEL_LOCAL
        st.caption(f"🖊️ Embeddings: `{emb_model}`")

        # Chunk parameters
        chunk_size = st.slider(
            "Chunk Size (Chars)",
            min_value=200,
            max_value=2000,
            value=config.CHUNK_SIZE,
            step=100
        )
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=50,
            max_value=500,
            value=config.CHUNK_OVERLAP,
            step=50
        )

        # Preset System Prompts
        _PRESET_DESCRIPTIONS = {
            "Strict Q&A": "🛡️ Refuses answers outside context",
            "Detailed Explainer": "📚 Step-by-step with citations",
            "Bullet Summary": "📃 Concise bullet points",
            "Technical Analyst": "🔬 Precise figures & code references",
            "ELI5 Explainer": "👶 Simple, beginner-friendly analogies",
            "Custom": "✏️ Write your own instruction",
        }
        selected_preset = st.selectbox(
            "System Prompt Preset",
            options=list(config.SYSTEM_PRESETS.keys()) + ["Custom"],
            index=0
        )
        st.caption(_PRESET_DESCRIPTIONS.get(selected_preset, ""))

        if selected_preset == "Custom":
            system_prompt = st.text_area("Custom System Prompt", value=config.DEFAULT_SYSTEM_PROMPT, height=120)
        else:
            system_prompt = st.text_area("System Prompt", value=config.SYSTEM_PRESETS[selected_preset], height=120)

    st.divider()

    # ── Document Uploader ──
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=[ext[1:] for ext in config.SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(config.SUPPORTED_EXTENSIONS)}. Max file size: {config.MAX_FILE_SIZE_MB} MB each. Max {config.MAX_DOCUMENTS} documents per session."
    )

    if uploaded_files:
        # Guard: check document count limit
        if len(uploaded_files) > config.MAX_DOCUMENTS:
            st.warning(f"⚠️ You uploaded {len(uploaded_files)} files, but the limit is {config.MAX_DOCUMENTS} per session. Only the first {config.MAX_DOCUMENTS} will be ingested.")
            uploaded_files = uploaded_files[:config.MAX_DOCUMENTS]

        if st.button("⚡ Ingest Documents", use_container_width=True, type="primary"):
            os.makedirs(config.UPLOAD_DIR, exist_ok=True)
            saved_paths = []
            
            # File validation
            invalid_files = []
            for f in uploaded_files:
                ext = Path(f.name).suffix.lower()
                if ext not in config.SUPPORTED_EXTENSIONS:
                    invalid_files.append(f.name)
            
            if invalid_files:
                st.error(f"❌ Unsupported files detected: {', '.join(invalid_files)}")
                log_diagnostic(f"Aborted ingestion: unsupported file extensions {invalid_files}")
            else:
                log_diagnostic(f"Starting ingestion of {len(uploaded_files)} file(s).")
                
                # Ingestion progress indicator (Contribution 17)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_chars = 0
                total_files = len(uploaded_files)
                
                seen_filenames = set()
                for idx, f in enumerate(uploaded_files):
                    if f.name in seen_filenames:
                        log_diagnostic(f"Skipped duplicate uploaded file: {f.name}")
                        continue
                    seen_filenames.add(f.name)
                    status_text.text(f"Saving {f.name}...")
                    safe_name = sanitize_filename(os.path.basename(f.name))
                    save_path = os.path.join(config.UPLOAD_DIR, safe_name)
                    
                    try:
                        with open(save_path, "wb") as out:
                            out.write(f.getbuffer())
                            
                        # Validate if document has readable content / not empty
                        file_size = os.path.getsize(save_path)
                        if file_size == 0:
                            st.error(f"❌ File '{f.name}' is empty.")
                            log_diagnostic(f"File {f.name} is empty (0 bytes).")
                            continue

                        # Validate file size against MAX_FILE_SIZE_MB limit
                        file_size_mb = file_size / (1024 * 1024)
                        if file_size_mb > config.MAX_FILE_SIZE_MB:
                            st.warning(f"⚠️ '{f.name}' is {file_size_mb:.1f} MB — exceeds {config.MAX_FILE_SIZE_MB} MB limit. Skipping.")
                            log_diagnostic(f"Skipped {f.name}: size {file_size_mb:.1f} MB exceeds limit.")
                            continue
                            
                        saved_paths.append(save_path)
                        total_chars += file_size
                    except Exception as fe:
                        st.error(f"❌ Failed to write {f.name}: {fe}")
                        log_diagnostic(f"Error saving {f.name}: {fe}")
                        
                    progress = float(idx + 1) / total_files
                    progress_bar.progress(progress)
                
                status_text.text("Building Vectorstore & running embeddings...")
                
                with st.spinner("Generating vector database..."):
                    try:
                        vs = build_vectorstore(
                            saved_paths, 
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap
                        )
                        st.session_state["vectorstore"] = vs
                        st.session_state["doc_names"] = [f.name for f in uploaded_files]
                        st.session_state["messages"] = []  # reset chat history
                        
                        # Compute Statistics (Contribution 13)
                        total_chunks = 0
                        try:
                            # Safely fetch collection size
                            total_chunks = len(vs.get()["ids"])
                        except Exception:
                            pass
                            
                        stats = {
                            "total_files": len(saved_paths),
                            "total_chunks": total_chunks,
                            "avg_chunk_size": int(total_chars / max(1, total_chunks)) if total_chunks else 0,
                            "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        st.session_state["doc_stats"] = stats
                        
                        st.success(f"✅ Ingestion complete! Chunks: {total_chunks}")
                        log_diagnostic(f"Successfully processed {len(saved_paths)} documents into {total_chunks} chunks.")
                    except Exception as e:
                        st.error(f"❌ Vectorstore error: {str(e)}")
                        log_diagnostic(f"Error during build_vectorstore: {e}")
                
                progress_bar.empty()
                status_text.empty()

    # ── Documents Section ──
    if "doc_names" in st.session_state and st.session_state["doc_names"]:
        st.divider()
        st.markdown("**📂 Document Context:**")
        for name in st.session_state["doc_names"]:
            # Download file support
            file_path = os.path.join(config.UPLOAD_DIR, name)
            if os.path.exists(file_path):
                # Show file size alongside download button
                fsize_mb = os.path.getsize(file_path) / (1024 * 1024)
                col_dl, col_sz = st.columns([3, 1])
                with col_dl:
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label=f"📄 {name}",
                            data=f.read(),
                            file_name=name,
                            key=f"dl_{name}",
                            use_container_width=True
                        )
                with col_sz:
                    st.markdown(
                        f'<div style="font-size:0.7rem;color:#64748b;padding-top:10px;">{fsize_mb:.2f}MB</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(f"📄 `{name}`")

    # ── Statistics Dashboard Panel ──
    if st.session_state["doc_stats"]:
        st.divider()
        st.markdown("**📊 Statistics**")
        stats = st.session_state["doc_stats"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stats-card">'
                f'<div class="stats-label">Files</div>'
                f'<div class="stats-val">{stats["total_files"]}</div>'
                f'</div>', unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div class="stats-card">'
                f'<div class="stats-label">Chunks</div>'
                f'<div class="stats-val">{stats["total_chunks"]}</div>'
                f'</div>', unsafe_allow_html=True
            )

        st.markdown(
            f'<div class="stats-card" style="text-align:center;">'
            f'<div class="stats-label">Avg Chunk Size</div>'
            f'<div class="stats-val">{stats["avg_chunk_size"]} <span style="font-size:0.9rem;color:#64748b;">chars</span></div>'
            f'</div>', unsafe_allow_html=True
        )
        st.caption(f"⏰ Last ingested: {stats['ingested_at']}")

    # ── Load Session / History Panel (Contribution 14) ──
    sessions = get_saved_sessions()
    if sessions:
        st.divider()
        st.markdown("**💾 Load Saved Session**")
        search_term = st.text_input("🔍 Search sessions", placeholder="Type keywords...", key="sess_search_input")
        filtered_sessions = sessions
        if search_term.strip():
            filtered_sessions = [s for s in sessions if search_term.lower() in s.lower()]
            
        selected_session = st.selectbox(
            "Choose a session",
            options=["-- Select --"] + filtered_sessions,
            key="session_select"
        )
        if selected_session != "-- Select --":
            col_l, col_d = st.columns(2)
            with col_l:
                if st.button("📂 Load Session", use_container_width=True):
                    load_session(selected_session)
                    st.rerun()
            with col_d:
                if st.button("🗑️ Delete Session", use_container_width=True):
                    fpath = os.path.join(config.SESSION_DIR, selected_session)
                    try:
                        if os.path.exists(fpath):
                            os.unlink(fpath)
                            log_diagnostic(f"Deleted saved session file: {selected_session}")
                            st.success(f"Deleted {selected_session}")
                            st.rerun()
                    except Exception as de:
                        st.error(f"Failed to delete session: {de}")

    # ── Save current session trigger ──
    if st.session_state.messages:
        st.divider()
        st.markdown("**💾 Save Current Session**")
        session_name_input = st.text_input("Session name", placeholder="Type name...", key="sess_name_input")
        if st.button("💾 Save Session", use_container_width=True):
            save_current_session(session_name_input)
            st.success("Session saved successfully!")
            st.rerun()

        # Clear chat history button
        if st.button("🚫 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state["current_session_file"] = ""
            log_diagnostic("Chat history cleared by user.")
            st.rerun()

    # ── Reset UI ──
    if "vectorstore" in st.session_state:
        st.divider()
        if st.button("🗑️ Reset Application", use_container_width=True):
            log_diagnostic("Triggered clear & reset.")
            try:
                st.session_state["vectorstore"].delete_collection()
            except Exception:
                pass
            
            # Clear cache reference
            from rag_pipeline import clear_vectorstore_cache
            try:
                clear_vectorstore_cache()
            except Exception:
                pass

            for key in ["vectorstore", "messages", "doc_names", "doc_stats", "current_session_file"]:
                st.session_state.pop(key, None)

            try:
                clear_upload_dir()
            except Exception:
                pass

            try:
                if os.path.exists(config.CHROMA_DB_DIR):
                    shutil.rmtree(config.CHROMA_DB_DIR)
            except Exception:
                pass

            st.success("Cleaned successfully!")
            st.rerun()

    st.divider()
    # Show message count if conversation is active
    if st.session_state.messages:
        msg_count = len(st.session_state.messages)
        user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
        st.markdown(
            f'<div style="text-align:center; font-size:0.75rem; color:#64748b; margin-bottom:6px;">'
            f'💬 {user_msgs} question{"s" if user_msgs != 1 else ""} · {msg_count} messages'
            f'</div>', unsafe_allow_html=True
        )
    st.caption("RAGForge v2.1 • Built by [Saurabh Gaur](https://github.com/DEVsaurabhgaur)")

# ── Main Area ──────────────────────────────────────────────
# ── Hero Header ──
docs_ready = "vectorstore" in st.session_state
status_text_hero = "Documents loaded — ask anything below" if docs_ready else "Upload documents to begin"
status_dot = '<span class="dot"></span>' if docs_ready else ''
st.markdown(
    f'<div class="hero-header">'
    f'<p class="hero-title">🔍 Chat with Your Documents</p>'
    f'<p class="hero-subtitle">Grounded answers with source citations, powered by RAG + LLMs</p>'
    f'<div class="hero-badge">{status_dot} {status_text_hero}</div>'
    f'</div>',
    unsafe_allow_html=True
)

# API Key warnings in main area
if llm_provider == "gemini" and not config.GEMINI_API_KEY:
    st.warning("⚠️ **GEMINI_API_KEY is missing!** Generative answers will fail. Please add your key to `.env` or set it in your environment.")
elif llm_provider == "openai" and not config.OPENAI_API_KEY:
    st.warning("⚠️ **OPENAI_API_KEY is missing!** Generative answers will fail. Please add your key to `.env` or set it in your environment.")

# Welcome/Guide Box
if not docs_ready:
    st.info("👈 Upload your documents (PDF, TXT, MD) in the sidebar and click **Ingest Documents** to begin.")
else:
    if not st.session_state.messages:
        st.success("✅ Documents ready. Ask your first question below to start the retrieval conversation.")

# Export Current Conversation (Contribution 15)
if st.session_state.messages:
    # ── Export buttons line ──
    col_exp1, col_exp2, _ = st.columns([1.5, 1.5, 7])
    
    # 1. Export as Markdown
    md_content = f"# Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    for msg in st.session_state.messages:
        role_label = "**User**" if msg["role"] == "user" else "**Assistant**"
        md_content += f"### {role_label}\n{msg['content']}\n\n"
        if "metrics" in msg and msg["metrics"]:
            m = msg["metrics"]
            md_content += f"*Metadata: Input Tokens: {m.get('input_tokens')}, Output Tokens: {m.get('output_tokens')}, Cost: ${m.get('cost'):.6f}*\n\n"
            
    with col_exp1:
        st.download_button(
            label="📝 Export as Markdown",
            data=md_content,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
        
    # 2. Export as JSON
    json_data = json.dumps(st.session_state.messages, indent=4)
    with col_exp2:
        st.download_button(
            label="📋 Export as JSON",
            data=json_data,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# Render Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Metrics (Contribution 16)
        if msg["role"] == "assistant" and "metrics" in msg and msg["metrics"]:
            metrics = msg["metrics"]
            st.markdown(
                f'<div style="margin-top: -6px; margin-bottom: 8px;">'
                f'<span class="metric-badge">⚡ In: {metrics.get("input_tokens", 0)} | Out: {metrics.get("output_tokens", 0)} tokens</span>'
                f'<span class="metric-cost">💰 Cost: ${metrics.get("cost", 0.0):.6f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        # Display Citations
        if "sources" in msg and msg["sources"]:
            with st.expander("📎 View Sources"):
                for i, src in enumerate(msg["sources"]):
                    file_name = src.get("source_file", "unknown")
                    page = src.get("page", "?")
                    snippet = src.get("snippet", "")
                    
                    # Safe word highlights inside sources snippet (Contribution 18)
                    highlighted_snippet = snippet
                    # Search for original query from the preceding user message to highlight words
                    # Find user question matching this response
                    msg_idx = st.session_state.messages.index(msg)
                    if msg_idx > 0:
                        prev_msg = st.session_state.messages[msg_idx - 1]
                        if prev_msg["role"] == "user":
                            highlighted_snippet = highlight_keywords(snippet, prev_msg["content"])

                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="badge">Source {i+1}</span> &nbsp;'
                        f'<strong>{file_name}</strong> — Page {page}<br><br>'
                        f'<span style="font-size:0.84rem; line-height:1.4;">{highlighted_snippet}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat Input & Ingestion
MAX_QUERY_CHARS = 2000
if question := st.chat_input("Ask a question about your documents..."):
    if "vectorstore" not in st.session_state:
        st.warning("⚠️ Please upload and ingest documents first!")
    elif len(question.strip()) == 0:
        st.warning("⚠️ Please enter a non-empty question.")
    elif len(question) > MAX_QUERY_CHARS:
        st.warning(f"⚠️ Your question is {len(question)} characters. Limit is {MAX_QUERY_CHARS}. Please shorten your query.")
    else:
        # Show User Message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Process Answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving contexts & synthesising answer..."):
                try:
                    log_diagnostic(f"Executing query: '{question}'")
                    
                    # Execute RAG query (passing full params)
                    result = query_rag(
                        question=question,
                        vectorstore=st.session_state["vectorstore"],
                        chat_history=st.session_state.messages[:-1],  # exclude current question
                        retrieval_mode=retrieval_mode,
                        k_results=top_k,
                        temperature=temperature,
                        system_prompt=system_prompt,
                        use_expansion=use_expansion
                    )
                    
                    answer = result["answer"]
                    source_docs = result["sources"]
                    metrics = result["metrics"]

                    # Display Answer Content
                    st.markdown(answer)
                    
                    # Display metrics immediately (Contribution 16)
                    ret_t = metrics.get("retrieval_time", 0.0)
                    gen_t = metrics.get("generation_time", 0.0)
                    st.markdown(
                        f'<div style="margin-top: -6px; margin-bottom: 8px;">'
                        f'<span class="metric-badge">⚡ In: {metrics.get("input_tokens", 0)} | Out: {metrics.get("output_tokens", 0)} tokens ({ret_t}s retrieve · {gen_t}s generate)</span>'
                        f'<span class="metric-cost">💰 Cost: ${metrics.get("cost", 0.0):.6f}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # Serialize citations
                    serialized_sources = [
                        {
                            "source_file": d.metadata.get("source_file", "unknown"),
                            "page": d.metadata.get("page", 0) + 1 if isinstance(d.metadata.get("page"), int) else d.metadata.get("page", "?"),
                            "snippet": truncate_text(d.page_content, 350),
                        }
                        for d in source_docs
                    ]

                    # Display Sources
                    if serialized_sources:
                        with st.expander("📎 View Sources"):
                            for i, src in enumerate(serialized_sources):
                                highlighted_snippet = highlight_keywords(src["snippet"], question)
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<span class="badge">Source {i+1}</span> &nbsp;'
                                    f'<strong>{src["source_file"]}</strong> — Page {src["page"]}<br><br>'
                                    f'<span style="font-size:0.84rem; line-height:1.4;">{highlighted_snippet}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    # Append to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": serialized_sources,
                        "metrics": metrics
                    })
                    
                    log_diagnostic("Query execution complete successfully.")
                    
                    # Autosave session progress (Contribution 14)
                    save_current_session()
                    
                except Exception as e:
                    err_msg = f"❌ Error generating response: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "sources": [],
                        "metrics": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
                    })
                    log_diagnostic(f"Query generation crashed: {e}")

# ── Diagnostic Log View at bottom of application ──
st.divider()
log_count = len(st.session_state["logs"])
with st.expander(f"🛠️ Diagnostics & Pipeline Log Console ({log_count} events)", expanded=False):
    if st.session_state["logs"]:
        # Action row: clear + download
        dcol1, dcol2, _ = st.columns([1, 1.5, 5])
        with dcol1:
            if st.button("🗑️ Clear Logs", key="clear_logs_btn"):
                st.session_state["logs"] = []
                st.rerun()
        with dcol2:
            logs_plain = "\n".join(st.session_state["logs"])
            st.download_button(
                label="⬇️ Download Logs",
                data=logs_plain,
                file_name=f"ragforge_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="dl_logs_btn"
            )
        logs_html = "<br>".join(st.session_state["logs"])
        st.markdown(f'<div class="logs-box">{logs_html}</div>', unsafe_allow_html=True)
    else:
        st.caption("No logs recorded yet.")
