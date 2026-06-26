"""
rag_pipeline.py — Core RAG logic: load, embed, query
"""
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

import config
from utils import clean_text, count_tokens, estimate_cost

# Cache dictionary to prevent repeated database connections and file locks on Windows
_VECTORSTORE_CACHE: dict = {}


def clear_vectorstore_cache() -> None:
    """Clear the in-memory vectorstore cache to free references and allow re-initialization."""
    _VECTORSTORE_CACHE.clear()


def get_embeddings():
    """Return embedding model based on config LLM provider setting."""
    if config.EMBEDDING_PROVIDER == 'openai':
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_OPENAI,
            api_key=config.OPENAI_API_KEY
        )
    else:  # local / free — no API key needed
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_LOCAL)


def get_llm(temperature: float = None):
    """Return LLM based on config provider setting.

    Args:
        temperature: Override the default temperature. If None, uses config.DEFAULT_TEMPERATURE.

    Returns:
        A LangChain chat model (ChatOpenAI or ChatGoogleGenerativeAI).
    """
    temp = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
    if config.LLM_PROVIDER == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.LLM_MODEL_OPENAI,
            api_key=config.OPENAI_API_KEY,
            temperature=temp
        )
    else:  # gemini — free tier available
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_GEMINI,
            google_api_key=config.GEMINI_API_KEY,
            temperature=temp
        )


def load_and_split_document(file_path: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
    """Load a single PDF, TXT, or MD file and split into chunks with enriched metadata."""
    file_path_obj = Path(file_path)
    ext = file_path_obj.suffix.lower()
    
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
            pages = loader.load()
        elif ext in ['.txt', '.md']:
            loader = TextLoader(file_path, encoding='utf-8')
            pages = loader.load()
            # Plain text files have no page numbers; assign 0
            for p in pages:
                p.metadata['page'] = 0
        else:
            # Fallback text loader
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                pages = loader.load()
            except Exception:
                logging.error(f"Unsupported file format for: {file_path}")
                return []
    except Exception as e:
        logging.error(f"Failed to load or parse document {file_path}: {e}")
        return []

    c_size = chunk_size or config.CHUNK_SIZE
    c_overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=['\n\n', '\n', '. ', ' ', ''],
    )
    chunks = splitter.split_documents(pages)

    # Enrich metadata + clean text
    filename = file_path_obj.name
    for chunk in chunks:
        chunk.metadata['source_file'] = filename
        if 'page' not in chunk.metadata:
            chunk.metadata['page'] = 0
        chunk.page_content = clean_text(chunk.page_content)

    # Filter out empty or too small chunks (e.g. noise characters)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 20]
    return chunks


def build_vectorstore(file_paths: List[str], chunk_size: int = None, chunk_overlap: int = None) -> Chroma:
    """
    Embed all documents and store in ChromaDB.
    Clears cache and deletes existing collection first. Returns vectorstore.
    """
    _VECTORSTORE_CACHE.clear()

    all_chunks: List[Document] = []
    for path in file_paths:
        chunks = load_and_split_document(path, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No text could be extracted from the uploaded files.")

    embeddings = get_embeddings()

    # Try to clean up directory to avoid lockups
    try:
        if os.path.exists(config.CHROMA_DB_DIR):
            db = Chroma(
                persist_directory=config.CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_name=config.COLLECTION_NAME
            )
            db.delete_collection()
    except Exception as e:
        logging.warning(f"Could not delete Chroma collection via API: {e}. Attempting folder deletion.")
        import shutil
        try:
            if os.path.exists(config.CHROMA_DB_DIR):
                shutil.rmtree(config.CHROMA_DB_DIR)
        except Exception as se:
            logging.error(f"Failed to delete database directory: {se}")

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DB_DIR,
        collection_name=config.COLLECTION_NAME,
    )
    
    # Cache the vectorstore reference
    cache_key = (config.CHROMA_DB_DIR, config.COLLECTION_NAME)
    _VECTORSTORE_CACHE[cache_key] = vectorstore

    return vectorstore


def load_existing_vectorstore() -> Chroma:
    """Load a previously built vectorstore from disk (if it exists) with caching."""
    cache_key = (config.CHROMA_DB_DIR, config.COLLECTION_NAME)
    if cache_key in _VECTORSTORE_CACHE:
        return _VECTORSTORE_CACHE[cache_key]

    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
    )
    _VECTORSTORE_CACHE[cache_key] = vectorstore
    return vectorstore


def vectorstore_exists() -> bool:
    """Check if a persisted vectorstore already exists on disk."""
    return (
        os.path.exists(config.CHROMA_DB_DIR)
        and any(Path(config.CHROMA_DB_DIR).iterdir())
    )


def rerank_documents(docs: List[Document], query: str) -> List[Document]:
    """Rerank retrieved documents using keyword-overlap scoring with length normalization.

    Scoring factors:
    - Keyword overlap: count of matching query terms in document
    - Length normalization: divides by log of document length to avoid bias toward longer docs
    - Phrase match bonus: 1.5 extra points if the full query appears verbatim
    """
    query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
    if not query_words:
        return docs

    import math
    scored_docs = []
    for doc in docs:
        content_words = set(re.findall(r'\b\w{3,}\b', doc.page_content.lower()))
        doc_len = max(1, len(content_words))
        overlap = len(query_words.intersection(content_words))
        # Length-normalize: reward density over sheer count
        normalized_overlap = overlap / math.log1p(doc_len)
        phrase_match = 1.5 if query.lower() in doc.page_content.lower() else 0.0
        score = normalized_overlap + phrase_match
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs]


def expand_query(query: str, llm) -> List[str]:
    """Generate search query variations using the LLM to improve retrieval coverage."""
    if not query or not query.strip():
        return []
    prompt = f"""You are a search expert. Generate exactly 2 alternative search queries (variations) for the user's question to retrieve better document contexts.
Write only the queries, one per line. Do not write any explanations, numbers, or bullet points.

QUESTION: {query}
"""
    try:
        response = llm.invoke(prompt)
        variations = [line.strip() for line in response.content.split('\n') if line.strip()]
        cleaned_variations = []
        for v in variations:
            v_clean = re.sub(r'^\d+[\.\-\s]+', '', v).strip('"\'')
            if v_clean:
                cleaned_variations.append(v_clean)
        return [query] + cleaned_variations[:2]
    except Exception as e:
        logging.warning(f"Query expansion failed: {e}")
        return [query]


def reformulate_question(question: str, chat_history: List[Dict[str, Any]], llm) -> str:
    """Reformulate follow-up query to incorporate conversational context from chat history."""
    if not question or not question.strip():
        return question
    if not chat_history:
        return question

    # Format recent history (limit to last 4 messages to save tokens)
    history_str = ""
    for msg in chat_history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        history_str += f"{role}: {content}\n"

    prompt = f"""Given the following chat history and a follow-up question, rewrite the follow-up question to be a standalone search query.
Do NOT answer the question, just output the rewritten standalone query.

CHAT HISTORY:
{history_str}

FOLLOW-UP QUESTION: {question}

STANDALONE QUESTION:"""
    try:
        response = llm.invoke(prompt)
        standalone = response.content.strip()
        if standalone:
            return standalone
    except Exception as e:
        logging.warning(f"Context-based question reformulation failed: {e}")
    return question


def get_hybrid_retriever(vectorstore: Chroma, k: int = 4) -> Any:
    """Create a hybrid retriever using BM25 and Chroma semantic search."""
    chroma_retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': k}
    )

    try:
        # Load all documents from vectorstore to train BM25 retriever
        data = vectorstore.get()
        documents = []
        if data and 'documents' in data and data['documents']:
            for text, meta in zip(data['documents'], data['metadatas']):
                documents.append(Document(page_content=text, metadata=meta or {}))

        if documents:
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = k

            ensemble_retriever = EnsembleRetriever(
                retrievers=[chroma_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
            return ensemble_retriever
    except Exception as e:
        logging.warning(f"Could not initialize BM25 retriever: {e}. Falling back to semantic search.")
    
    return chroma_retriever


def validate_context_constraints(answer: str) -> bool:
    """Verify if the generated response claims that context was insufficient.

    Returns True if the answer appears valid/grounded, False if it signals
    that no relevant information was found in the context.
    """
    refusal_phrases = [
        "could not find this",
        "not in the provided context",
        "not mentioned in the context",
        "insufficient information",
        "no information provided",
        "the context does not contain",
        "not available in the context",
        "cannot find this information",
        "no relevant information",
        "outside the scope of the documents",
    ]
    answer_lower = answer.lower()
    return not any(phrase in answer_lower for phrase in refusal_phrases)


def query_rag(
    question: str,
    vectorstore: Chroma,
    chat_history: List[Dict[str, Any]] = None,
    retrieval_mode: str = 'hybrid',
    k_results: int = 4,
    temperature: float = 0.3,
    system_prompt: str = None,
    use_expansion: bool = False
) -> Dict[str, Any]:
    """
    Execute conversational RAG query pipeline.
    Returns dictionary with response text, retrieved source docs, and token/cost metrics.
    """
    llm = get_llm(temperature=temperature)
    
    # 1. Reformulate question using conversational history (Contribution 7)
    standalone_query = reformulate_question(question, chat_history or [], llm)

    # 2. Get Retriever (Hybrid BM25 + Semantic or Semantic-Only)
    if retrieval_mode == 'hybrid':
        retriever = get_hybrid_retriever(vectorstore, k=k_results)
    else:
        retriever = vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': k_results}
        )

    # 3. Retrieve documents, optionally with LLM Query Expansion (Contribution 6)
    if use_expansion:
        queries = expand_query(standalone_query, llm)
        docs = []
        seen = set()
        for q in queries:
            q_docs = retriever.invoke(q)
            for doc in q_docs:
                key = (doc.metadata.get('source_file'), doc.metadata.get('page'), doc.page_content[:100])
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)
        # Limit to top k after expansion merging
        docs = docs[:k_results]
    else:
        docs = retriever.invoke(standalone_query)

    # 4. Rerank documents locally (Contribution 5)
    docs = rerank_documents(docs, standalone_query)

    if not docs:
        return {
            'answer': 'I could not find any relevant content in the uploaded documents.',
            'sources': [],
            'metrics': {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0}
        }

    # 5. Build strict context string (Contribution 8)
    context_parts = []
    for i, doc in enumerate(docs):
        src = doc.metadata.get('source_file', 'unknown')
        page = doc.metadata.get('page', 0)
        page_display = page + 1 if isinstance(page, int) else page
        context_parts.append(f"[Source {i+1}: {src}, Page {page_display}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    sys_instruction = system_prompt or config.DEFAULT_SYSTEM_PROMPT

    # Assemble concise prompt template
    prompt = f"""{sys_instruction}

CONTEXT:
{context}

QUESTION: {standalone_query}

Provide a clear, concise answer. At the end, list the exact sources (file name + page) you used."""

    # 6. Call LLM
    response = llm.invoke(prompt)
    answer = response.content

    # 7. Hallucination guard check (Contribution 9)
    if not validate_context_constraints(answer):
        # Double check if LLM hallucinated sources when it failed to retrieve
        answer = "I could not find this in the uploaded documents."

    # 8. Calculate token usage and estimated cost
    model_name = config.LLM_MODEL_OPENAI if config.LLM_PROVIDER == 'openai' else config.LLM_MODEL_GEMINI
    input_tokens = count_tokens(prompt, model_name=model_name)
    output_tokens = count_tokens(answer, model_name=model_name)
    cost = estimate_cost(input_tokens, output_tokens, provider=config.LLM_PROVIDER)

    return {
        'answer': answer,
        'sources': docs,
        'metrics': {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost
        }
    }

