from __future__ import annotations

import os
import json
import glob
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss not installed. Run: pip install faiss-cpu") from e

from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: Dict[str, Any]


def read_text_files(folder: str) -> List[Tuple[str, str]]:
    paths = sorted(glob.glob(os.path.join(folder, "**/*.*"), recursive=True))
    docs: List[Tuple[str, str]] = []
    for p in paths:
        if not (p.endswith(".md") or p.endswith(".txt")):
            continue
        with open(p, "r", encoding="utf-8") as f:
            docs.append((p, f.read()))
    return docs


def chunk_text(
    text: str,
    chunk_size: int = 350,
    chunk_overlap: int = 60
) -> List[str]:
    """
    Simple word-based chunking.
    chunk_size and overlap are in words.
    """
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words).strip())
        i += step
    return [c for c in chunks if c]


class RAGStore:
    """
    Local FAISS-based vector store for RAG.
    Saves:
      - faiss index
      - chunks metadata json
      - embedding model name
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[Chunk] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        return emb

    def build_from_folder(
        self,
        kb_folder: str,
        chunk_size: int = 350,
        chunk_overlap: int = 60
    ) -> None:
        docs = read_text_files(kb_folder)
        all_chunks: List[Chunk] = []

        for path, text in docs:
            for idx, c in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
                chunk_id = f"{os.path.basename(path)}::chunk{idx}"
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=c,
                        meta={"source_path": path, "chunk_index": idx}
                    )
                )

        if not all_chunks:
            raise RuntimeError(f"No chunks built. Check folder: {kb_folder}")

        embeddings = self._embed([c.text for c in all_chunks])
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.chunks = all_chunks

    def save(self, out_dir: str) -> None:
        if self.index is None:
            raise RuntimeError("Index is empty. Build it first.")

        os.makedirs(out_dir, exist_ok=True)

        faiss.write_index(self.index, os.path.join(out_dir, "kb.index"))

        payload = {
            "model_name": self.model_name,
            "chunks": [
                {"chunk_id": c.chunk_id, "text": c.text, "meta": c.meta}
                for c in self.chunks
            ],
        }
        with open(os.path.join(out_dir, "kb_chunks.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dir_path: str) -> "RAGStore":
        chunks_path = os.path.join(dir_path, "kb_chunks.json")
        index_path = os.path.join(dir_path, "kb.index")

        if not os.path.exists(chunks_path) or not os.path.exists(index_path):
            raise RuntimeError("Missing kb_chunks.json or kb.index in the folder.")

        with open(chunks_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        store = cls(model_name=payload["model_name"])
        store.index = faiss.read_index(index_path)
        store.chunks = [
            Chunk(chunk_id=x["chunk_id"], text=x["text"], meta=x["meta"])
            for x in payload["chunks"]
        ]
        return store

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not loaded. Build or load first.")
        if not query.strip():
            return []

        q_emb = self._embed([query])  # (1, d)
        scores, idxs = self.index.search(q_emb, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.chunks):
                continue
            c = self.chunks[idx]
            results.append(
                {
                    "score": float(score),
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "meta": c.meta,
                }
            )

        return results
    
    
    def test(self):
        store = self.load("artifacts/kb_index")

        while True:
            q = input("\nQuery: ").strip()
            if q.lower() in {"exit", "quit"}:
                break

            hits = store.retrieve(q, k=5)
            for i, h in enumerate(hits, 1):
                print(f"\n[{i}] score={h['score']:.4f} id={h['chunk_id']}")
                print(f"source={h['meta'].get('source_path')}")
                print(h["text"][:600])    
        

if __name__ == "__main__":
    RAGStore.test()