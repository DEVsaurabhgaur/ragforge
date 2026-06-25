"""
scripts/export_sessions.py — Export all saved sessions to a single JSON file.
Usage: python scripts/export_sessions.py
"""
import os, json, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def export_all():
    sessions = []
    session_dir = config.SESSION_DIR
    if not os.path.exists(session_dir):
        print("No sessions directory found.")
        return

    files = [f for f in os.listdir(session_dir) if f.endswith(".json")]
    if not files:
        print("No session files found.")
        return

    for fname in sorted(files):
        with open(os.path.join(session_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_file"] = fname
            sessions.append(data)

    out = "all_sessions_export.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)

    print(f"Exported {len(sessions)} session(s) to {out}")


if __name__ == "__main__":
    export_all()
