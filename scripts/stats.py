"""
scripts/stats.py — Print RAGForge document and session statistics to the console.

Usage:
    python scripts/stats.py
"""
import os
import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def hr(char: str = "─", width: int = 60) -> str:
    return char * width


def fmt_size(bytes_val: int) -> str:
    """Human-readable file size."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 ** 2:
        return f"{bytes_val / 1024:.1f} KB"
    else:
        return f"{bytes_val / 1024 ** 2:.2f} MB"


def print_upload_stats():
    upload_dir = Path(config.UPLOAD_DIR)
    files = list(upload_dir.glob("*")) if upload_dir.exists() else []
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    print(hr())
    print(f"  📂  Uploaded Documents ({config.UPLOAD_DIR})")
    print(hr())
    if not files:
        print("  (no documents uploaded)")
    else:
        for f in sorted(files):
            if f.is_file():
                size = fmt_size(f.stat().st_size)
                print(f"  • {f.name:<40} {size:>10}")
        print(hr("─"))
        print(f"  Total: {len(files)} file(s) — {fmt_size(total_size)}")
    print()


def print_session_stats():
    session_dir = Path(config.SESSION_DIR)
    sessions = sorted(session_dir.glob("*.json"), key=os.path.getmtime, reverse=True) if session_dir.exists() else []

    print(hr())
    print(f"  💾  Saved Sessions ({config.SESSION_DIR})")
    print(hr())
    if not sessions:
        print("  (no sessions saved)")
    else:
        for s in sessions:
            try:
                with open(s, "r", encoding="utf-8") as f:
                    data = json.load(f)
                msg_count = len(data.get("messages", []))
                ts = data.get("timestamp", "unknown")[:19]
                doc_count = len(data.get("doc_names", []))
                print(f"  • {s.name:<38} {msg_count:>4} msgs  {doc_count} docs  [{ts}]")
            except Exception:
                print(f"  • {s.name:<38} (unreadable)")
        print(hr("─"))
        print(f"  Total: {len(sessions)} session(s)")
    print()


def print_vectorstore_stats():
    db_dir = Path(config.CHROMA_DB_DIR)
    print(hr())
    print(f"  🗄️   ChromaDB ({config.CHROMA_DB_DIR})")
    print(hr())
    if not db_dir.exists() or not any(db_dir.iterdir()):
        print("  (no vectorstore found — ingest documents first)")
    else:
        total_size = sum(f.stat().st_size for f in db_dir.rglob("*") if f.is_file())
        file_count = sum(1 for f in db_dir.rglob("*") if f.is_file())
        print(f"  Size: {fmt_size(total_size)}  ({file_count} internal files)")
    print()


if __name__ == "__main__":
    print()
    print(f"  🔍 RAGForge {config.APP_VERSION} — Statistics Report")
    print()
    print_upload_stats()
    print_session_stats()
    print_vectorstore_stats()
    print(hr("═"))
    print()
