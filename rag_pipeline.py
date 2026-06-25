"""
rag_pipeline.py — Core RAG logic: load, embed, query
"""
import os
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

import config
from utils import clean_text


def get_embeddings():
    """Return embedding model based on config."""
    if config.EMBEDDING_PROVIDER == 'openai':
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_OPENAI,
            api_key=config.OPENAI_API_KEY
        )
    else:  # local / free — no API key needed
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_LOCAL)


def get_llm():
    """Return LLM based on config."""
    if config.LLM_PROVIDER == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.LLM_MODEL_OPENAI,
            api_key=config.OPENAI_API_KEY
        )
    else:  # gemini — free tier available
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_GEMINI,
            google_api_key=config.GEMINI_API_KEY
        )


def load_and_split_pdf(pdf_path: str) -> List[Document]:
    """Load a single PDF and split into chunks with enriched metadata."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # each page = one Document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=['\n\n', '\n', '. ', ' ', ''],
    )
    chunks = splitter.split_documents(pages)

    # Enrich metadata + clean text
    filename = Path(pdf_path).name
    for chunk in chunks:
        chunk.metadata['source_file'] = filename
        chunk.page_content = clean_text(chunk.page_content)

    # Filter out empty chunks
    chunks = [c for c in chunks if len(c.page_content.strip()) > 20]
    return chunks


def build_vectorstore(pdf_paths: List[str]) -> Chroma:
    """
    Embed all PDFs and store in ChromaDB.
    Overwrites existing collection. Returns vectorstore.
    """
    all_chunks: List[Document] = []
    for path in pdf_paths:
        chunks = load_and_split_pdf(path)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    embeddings = get_embeddings()

    # Try to delete the collection first to avoid PermissionError on Windows when files are locked.
    try:
        if os.path.exists(config.CHROMA_DB_DIR):
            db = Chroma(
                persist_directory=config.CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_name=config.COLLECTION_NAME
            )
            db.delete_collection()
    except Exception:
        # Fallback to shutil if database isn't initialized or has other issues
        import shutil
        try:
            if os.path.exists(config.CHROMA_DB_DIR):
                shutil.rmtree(config.CHROMA_DB_DIR)
        except Exception:
            pass

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DB_DIR,
        collection_name=config.COLLECTION_NAME,
    )
    return vectorstore


def load_existing_vectorstore() -> Chroma:
    """Load a previously built vectorstore from disk (if it exists)."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
    )


def vectorstore_exists() -> bool:
    """Check if a persisted vectorstore already exists on disk."""
    return (
        os.path.exists(config.CHROMA_DB_DIR)
        and any(Path(config.CHROMA_DB_DIR).iterdir())
    )


def query_rag(question: str, vectorstore: Chroma) -> Dict:
    """
    Run a RAG query.
    Returns dict with 'answer' (str) and 'sources' (List[Document]).
    """
    # 1. Retrieve relevant chunks
    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': config.TOP_K_RESULTS}
    )
    docs = retriever.invoke(question)

    if not docs:
        return {
            'answer': 'I could not find any relevant content in the uploaded documents.',
            'sources': [],
        }

    # 2. Build context string
    context_parts = []
    for i, doc in enumerate(docs):
        src = doc.metadata.get('source_file', 'unknown')
        page = doc.metadata.get('page', '?')
        page_display = page + 1 if isinstance(page, int) else page
        context_parts.append(
            f'[Source {i+1}: {src}, Page {page_display}]\n{doc.page_content}'
        )
    context = '\n\n---\n\n'.join(context_parts)

    # 3. Build strict RAG prompt
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided document context.
If the answer is not in the context, say exactly: "I could not find this in the uploaded documents."
Do NOT use any outside knowledge or make things up.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, concise answer. At the end, list the exact sources (file name + page) you used."""

    # 4. Get LLM answer
    llm = get_llm()
    response = llm.invoke(prompt)

    return {
        'answer': response.content,
        'sources': docs,
    }
