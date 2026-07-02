"""
scripts/reset_db.py — CLI utility to safely reset the ChromaDB vectorstore.
Usage: python scripts/reset_db.py
"""
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def reset_db():
    """Purge vector store collection database and upload directory contents."""
    if os.path.exists(config.CHROMA_DB_DIR):
        try:
            shutil.rmtree(config.CHROMA_DB_DIR)
            print(f"Deleted vectorstore at: {config.CHROMA_DB_DIR}")
        except OSError as e:
            print(f"Warning: Could not delete vectorstore directory {config.CHROMA_DB_DIR} ({e}). It might be locked by another process.")
    else:
        print("No vectorstore found — nothing to delete.")

    if os.path.exists(config.UPLOAD_DIR):
        deleted_count = 0
        for f in os.listdir(config.UPLOAD_DIR):
            fpath = os.path.join(config.UPLOAD_DIR, f)
            try:
                if os.path.isfile(fpath) or os.path.islink(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
                deleted_count += 1
            except Exception as e:
                print(f"Warning: Could not delete {fpath} ({e})")
        print(f"Cleared {deleted_count} file(s) from upload directory: {config.UPLOAD_DIR}")

    print("Reset complete.")


if __name__ == "__main__":
    reset_db()
