"""
utils.py — Helper functions for RAGForge
"""
import re
import os
from pathlib import Path
from typing import Union

# Precompiled regular expressions for optimization
_RE_MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
_RE_NON_PRINTABLE = re.compile(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]')
_RE_MULTIPLE_SPACES = re.compile(r'[ \t]{2,}')
_RE_WORDS = re.compile(r'\b\w+\b')
_RE_FILENAME_SPECIAL = re.compile(r'[^\w\s\-.]')
_RE_WHITESPACE = re.compile(r'[\s]+')
_RE_COLL_START_END = re.compile(r'^[a-zA-Z0-9].*[a-zA-Z0-9]$')
_RE_COLL_ALLOWED = re.compile(r'^[a-zA-Z0-9_\-.]+$')
_RE_IPV4 = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
_RE_WORDS_GE3 = re.compile(r'\b\w{3,}\b')


def clean_text(text: str) -> str:
    """Remove excessive whitespace, fix common PDF extraction artifacts."""
    # Remove multiple blank lines
    text = _RE_MULTIPLE_NEWLINES.sub('\n\n', text)
    # Remove non-printable characters (except newlines/tabs)
    text = _RE_NON_PRINTABLE.sub('', text)
    # Collapse multiple spaces
    text = _RE_MULTIPLE_SPACES.sub(' ', text)
    return text.strip()


def get_pdf_files_in_dir(directory: str) -> list:
    """Return list of all .pdf file paths in a directory."""
    return [str(p) for p in Path(directory).glob('*.pdf')]


def format_source_display(source_file: str, page: int | str) -> str:
    """Format a source reference for display."""
    if page is None:
        p_str = "N/A"
    else:
        p_str = str(page).strip()
        if not p_str:
            p_str = "N/A"
    return f"{source_file} — Page {p_str}"


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text to max_chars, appending ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + '...'


def word_count(text: str) -> int:
    """Count the number of words in a string in a memory-efficient way."""
    return sum(1 for _ in _RE_WORDS.finditer(text))


def sanitize_filename(name: str) -> str:
    """
    Sanitize incoming filename parameter by removing directory traversal patterns and stripping special symbols.
    """
    """Sanitize a string to be safe for use as a filename."""
    safe = _RE_FILENAME_SPECIAL.sub('', name).strip()
    safe = _RE_WHITESPACE.sub('_', safe)
    return safe[:100]  # Truncate to 100 chars max


def get_file_size_mb(file_path: str) -> float:
    """Return the size of a file in megabytes, rounded to 2 decimal places."""
    size_bytes = os.path.getsize(file_path)
    return round(size_bytes / (1024 * 1024), 2)


_STOPWORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "not", "no",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "they", "them", "their",
])


def remove_stopwords(text: str) -> str:
    """Remove common English stopwords from a text string, preserving word boundaries."""
    words = _RE_WORDS.findall(text)
    filtered = [w for w in words if w.lower() not in _STOPWORDS]
    return " ".join(filtered)


def is_valid_collection_name(name: str) -> bool:
    """Validate a ChromaDB collection name against ChromaDB naming constraints.

    Rules:
    - Must be 3-63 characters
    - Must start and end with an alphanumeric character
    - May only contain alphanumeric characters, underscores, hyphens, or periods
    - Must not contain consecutive periods
    - Must not be a valid IPv4 address

    Returns:
        True if the name is valid, False otherwise.
    """
    if not (3 <= len(name) <= 63):
        return False
    if not _RE_COLL_START_END.match(name):
        return False
    if not _RE_COLL_ALLOWED.match(name):
        return False
    if '..' in name:
        return False
    if _RE_IPV4.match(name):
        return False
    return True


def ensure_dirs():
    """Create required directories if they don't exist."""
    from config import CHROMA_DB_DIR, UPLOAD_DIR, SESSION_DIR
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(SESSION_DIR, exist_ok=True)


def clear_upload_dir():
    """Delete all files in the upload directory (cleanup helper)."""
    from config import UPLOAD_DIR
    for f in Path(UPLOAD_DIR).glob('*'):
        if f.is_file():
            f.unlink()


def count_tokens(text: str, model_name: str = 'gpt-4o-mini') -> int:
    """Estimate or count the number of tokens in a text string."""
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')
        return len(encoding.encode(text))
    except Exception:
        # Fallback heuristic: 1 token ~ 4 characters
        return max(1, len(text) // 4)


def estimate_cost(input_tokens: int, output_tokens: int, provider: str = 'openai', model: str = '') -> float:
    """Estimate cost in USD based on input/output token counts and specific model pricing."""
    p_lower = provider.lower()
    m_lower = model.lower()

    if p_lower == 'openai':
        if 'gpt-4o-mini' in m_lower or not model:
            # gpt-4o-mini pricing: $0.15 / 1M input, $0.60 / 1M output tokens
            return (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
        elif 'gpt-4o' in m_lower:
            # gpt-4o pricing: $2.50 / 1M input, $10.00 / 1M output tokens
            return (input_tokens * 2.50 + output_tokens * 10.00) / 1_000_000
        else:
            # fallback generic openai pricing
            return (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
    else:  # gemini / other
        if 'gemini-2.5-flash' in m_lower or not model:
            # gemini-2.5-flash: $0.075 / 1M input, $0.30 / 1M output tokens
            return (input_tokens * 0.075 + output_tokens * 0.30) / 1_000_000
        elif 'gemini-2.5-pro' in m_lower:
            # gemini-2.5-pro: $1.25 / 1M input, $5.00 / 1M output tokens
            return (input_tokens * 1.25 + output_tokens * 5.00) / 1_000_000
        else:
            # fallback gemini/generic pricing
            return (input_tokens * 0.075 + output_tokens * 0.30) / 1_000_000


def highlight_keywords(text: str, query: str) -> str:
    """Highlight query keywords inside text using safe HTML styling tags."""
    import html
    escaped_text = html.escape(text)

    # Extract alphanumeric words of length >= 3
    words = _RE_WORDS_GE3.findall(query.lower())
    if not words:
        return escaped_text

    # Sort words by length descending to match longer words first
    words = sorted(list(set(words)), key=len, reverse=True)
    for word in words:
        pattern = re.compile(rf'\b({re.escape(word)})\b', re.IGNORECASE)
        escaped_text = pattern.sub(
            r'<mark style="background-color: rgba(245, 158, 11, 0.25); color: #fbbf24; padding: 2px 4px; border-radius: 4px; font-weight: bold;">\1</mark>',
            escaped_text
        )
    return escaped_text

