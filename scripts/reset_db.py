"""
scripts/reset_db.py — CLI utility to safely reset the ChromaDB vectorstore.
Usage: python scripts/reset_db.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def reset_db():
    if os.path.exists(config.CHROMA_DB_DIR):
        shutil.rmtree(config.CHROMA_DB_DIR)
        print(f"Deleted vectorstore at: {config.CHROMA_DB_DIR}")
    else:
        print("No vectorstore found — nothing to delete.")

    if os.path.exists(config.UPLOAD_DIR):
        for f in os.listdir(config.UPLOAD_DIR):
            os.remove(os.path.join(config.UPLOAD_DIR, f))
        print(f"Cleared upload directory: {config.UPLOAD_DIR}")

    print("Reset complete.")


if __name__ == "__main__":
    reset_db()
