"""
scripts/list_sessions.py — Pretty-print all saved chat sessions and their metadata.

Usage:
    python scripts/list_sessions.py
"""
import os
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def format_timestamp(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso_str[:16] if iso_str else "unknown"


def main():
    session_dir = Path(config.SESSION_DIR)
    if not session_dir.exists():
        print("No session directory found. No sessions have been saved yet.")
        return

    sessions = sorted(session_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
    if not sessions:
        print("No saved sessions found.")
        return

    print()
    print(f"  💾  RAGForge Saved Sessions ({len(sessions)} total)")
    print("  " + "─" * 70)
    print(f"  {'File':<32} {'Msgs':>5}  {'Docs':>5}  {'Timestamp':<18}  {'Documents'}")
    print("  " + "─" * 70)

    for session_path in sessions:
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            msgs = len(data.get("messages", []))
            docs = data.get("doc_names", [])
            ts = format_timestamp(data.get("timestamp", ""))
            doc_names = ", ".join(docs[:2]) + ("..." if len(docs) > 2 else "")
            print(f"  {session_path.name:<32} {msgs:>5}  {len(docs):>5}  {ts:<18}  {doc_names}")
        except Exception as e:
            print(f"  {session_path.name:<32} (error reading: {e})")

    print("  " + "─" * 70)
    print()


if __name__ == "__main__":
    main()
