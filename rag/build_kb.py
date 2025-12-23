# rag/build_kb.py
from rag_store import RAGStore

if __name__ == "__main__":
    kb_folder = "knowledge"
    out_dir = "rag/artifacts/kb_index"

    store = RAGStore(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store.build_from_folder(kb_folder, chunk_size=220, chunk_overlap=40)
    store.save(out_dir)

    print(f"Built KB index at: {out_dir}")