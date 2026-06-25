"""
utils.py — Helper functions for RAGForge
"""
import re
import os
from pathlib import Path


def clean_text(text: str) -> str:
    """Remove excessive whitespace, fix common PDF extraction artifacts."""
    # Remove multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove non-printable characters (except newlines/tabs)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def get_pdf_files_in_dir(directory: str) -> list:
    """Return list of all .pdf file paths in a directory."""
    return [str(p) for p in Path(directory).glob('*.pdf')]


def format_source_display(source_file: str, page: int | str) -> str:
    """Format a source reference for display."""
    return f"{source_file} — Page {page}"


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text to max_chars, appending ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + '...'


def ensure_dirs():
    """Create required directories if they don't exist."""
    from config import CHROMA_DB_DIR, UPLOAD_DIR
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def clear_upload_dir():
    """Delete all files in the upload directory (cleanup helper)."""
    from config import UPLOAD_DIR
    for f in Path(UPLOAD_DIR).glob('*'):
        if f.is_file():
            f.unlink()
