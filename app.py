"""
app.py — RAGForge Streamlit UI
Run with: streamlit run app.py
"""
import os
import shutil
import streamlit as st
from pathlib import Path

import config
from rag_pipeline import (
    build_vectorstore,
    load_existing_vectorstore,
    vectorstore_exists,
    query_rag,
)
from utils import ensure_dirs, truncate_text

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title='RAGForge',
    page_icon='🔍',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Ensure required dirs exist
ensure_dirs()

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .source-card {
        background: #1e1e2e;
        border-left: 3px solid #7c3aed;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .badge {
        background: #7c3aed22;
        color: #a78bfa;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 RAGForge")
    st.caption("Intelligent Multi-Document Q&A")
    st.divider()

    # Provider info
    provider_label = config.LLM_PROVIDER.upper()
    embed_label = "OpenAI" if config.EMBEDDING_PROVIDER == "openai" else "Local (MiniLM)"
    st.markdown(f"**LLM:** `{provider_label}` &nbsp;|&nbsp; **Embeddings:** `{embed_label}`", unsafe_allow_html=True)
    st.divider()

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs to query",
    )

    if uploaded_files:
        if st.button("⚡ Process Documents", use_container_width=True, type="primary"):
            os.makedirs(config.UPLOAD_DIR, exist_ok=True)
            pdf_paths = []

            for f in uploaded_files:
                save_path = os.path.join(config.UPLOAD_DIR, f.name)
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())
                pdf_paths.append(save_path)

            with st.spinner(f"Embedding {len(pdf_paths)} document(s)... this may take a minute."):
                try:
                    vs = build_vectorstore(pdf_paths)
                    st.session_state["vectorstore"] = vs
                    st.session_state["doc_names"] = [f.name for f in uploaded_files]
                    st.session_state["messages"] = []  # reset chat on new docs
                    st.success(f"✅ {len(uploaded_files)} document(s) ready!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Show loaded docs
    if "doc_names" in st.session_state:
        st.divider()
        st.markdown("**Loaded Documents:**")
        for name in st.session_state["doc_names"]:
            st.markdown(f"📄 `{name}`")

    # Load existing DB option
    if vectorstore_exists() and "vectorstore" not in st.session_state:
        st.divider()
        if st.button("📂 Load Previous Session", use_container_width=True):
            with st.spinner("Loading vectorstore from disk..."):
                try:
                    st.session_state["vectorstore"] = load_existing_vectorstore()
                    st.success("✅ Previous session loaded!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Clear button
    if "vectorstore" in st.session_state:
        st.divider()
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            # Safe delete/clear of collection
            try:
                st.session_state["vectorstore"].delete_collection()
            except Exception:
                pass

            for key in ["vectorstore", "messages", "doc_names"]:
                st.session_state.pop(key, None)

            # Clean up uploaded PDF files from disk
            from utils import clear_upload_dir
            try:
                clear_upload_dir()
            except Exception:
                pass

            # Safe cleanup of directory
            try:
                if os.path.exists(config.CHROMA_DB_DIR):
                    shutil.rmtree(config.CHROMA_DB_DIR)
            except Exception:
                pass

            st.rerun()

    st.divider()
    st.caption("Built by [Saurabh Gaur](https://github.com/DEVsaurabhgaur)")

# ── Main Area ──────────────────────────────────────────────────
st.title("Chat with Your Documents")

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome / status message
if "vectorstore" not in st.session_state:
    st.info("👈 Upload PDFs from the sidebar and click **Process Documents** to start.")
else:
    if not st.session_state.messages:
        st.success("✅ Documents loaded — ask anything below!")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📎 View Sources"):
                for i, src in enumerate(msg["sources"]):
                    file_name = src.get("source_file", "unknown")
                    page = src.get("page", "?")
                    snippet = src.get("snippet", "")
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="badge">Source {i+1}</span> &nbsp;'
                        f'<strong>{file_name}</strong> — Page {page}<br><br>'
                        f'<code style="font-size:0.82rem">{snippet}</code>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    if "vectorstore" not in st.session_state:
        st.warning("⚠️ Please upload and process documents first!")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    result = query_rag(question, st.session_state["vectorstore"])
                    answer = result["answer"]
                    source_docs = result["sources"]

                    st.markdown(answer)

                    # Serialize sources for session state (Documents aren't JSON-serializable)
                    serialized_sources = [
                        {
                            "source_file": d.metadata.get("source_file", "unknown"),
                            "page": d.metadata.get("page", 0) + 1 if isinstance(d.metadata.get("page"), int) else d.metadata.get("page", "?"),
                            "snippet": truncate_text(d.page_content, 300),
                        }
                        for d in source_docs
                    ]

                    if serialized_sources:
                        with st.expander("📎 View Sources"):
                            for i, src in enumerate(serialized_sources):
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<span class="badge">Source {i+1}</span> &nbsp;'
                                    f'<strong>{src["source_file"]}</strong> — Page {src["page"]}<br><br>'
                                    f'<code style="font-size:0.82rem">{src["snippet"]}</code>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": serialized_sources,
                    })

                except Exception as e:
                    err_msg = f"❌ Error generating answer: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "sources": [],
                    })
