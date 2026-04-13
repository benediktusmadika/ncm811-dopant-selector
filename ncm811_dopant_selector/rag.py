from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List

from .constants import LOGGER_NAME
from .models import Chunk
from .openai_client import OpenAIClient
from .optional_deps import np
from .utils import sha256_text

LOG = logging.getLogger(LOGGER_NAME)


class EmbeddingIndex:
    def __init__(self, cache_dir: Path, embedding_model: str):
        self.cache_dir = cache_dir
        self.embedding_model = embedding_model
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._chunks: List[Chunk] = []
        self._vecs = None  # numpy array shape (N, D)
        self._chunk_ids: List[str] = []

    def build(self, client: OpenAIClient, chunks: List[Chunk], refresh: bool = False) -> None:
        if np is None:
            raise RuntimeError("numpy is required for embeddings RAG. Install numpy.")

        # Fingerprint (chunks + model)
        h = hashlib.sha256()
        h.update(self.embedding_model.encode("utf-8"))
        for c in chunks:
            h.update(c.chunk_id.encode("utf-8"))
            h.update(sha256_text(c.text).encode("utf-8"))
        fp = h.hexdigest()[:16]

        vec_path = self.cache_dir / f"emb_{fp}.npz"
        meta_path = self.cache_dir / f"emb_{fp}.meta.json"

        if (not refresh) and vec_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("embedding_model") == self.embedding_model:
                data = np.load(vec_path)
                self._vecs = data["vecs"]
                self._chunk_ids = list(meta["chunk_ids"])
                self._chunks = chunks
                LOG.info("Loaded embedding index from cache (%s chunks).", len(self._chunk_ids))
                return

        # Build embeddings
        LOG.info("Computing embeddings for %d chunks (model=%s)...", len(chunks), self.embedding_model)
        texts = [c.text for c in chunks]
        # Batch to avoid API limits
        batch_size = 64
        vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vecs.extend(client.embed_texts(batch))

        mat = np.array(vecs, dtype=np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        mat = mat / norms

        np.savez_compressed(vec_path, vecs=mat)
        meta_path.write_text(json.dumps({
            "embedding_model": self.embedding_model,
            "chunk_ids": [c.chunk_id for c in chunks],
        }, indent=2), encoding="utf-8")

        self._vecs = mat
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._chunks = chunks
        LOG.info("Built embedding index (%d chunks).", len(chunks))

    def search(self, client: OpenAIClient, query: str, top_k: int = 10) -> List[Chunk]:
        if np is None:
            raise RuntimeError("numpy is required for embeddings RAG.")
        if self._vecs is None:
            raise RuntimeError("Embedding index not built.")
        qvec = client.embed_texts([query])[0]
        q = np.array(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self._vecs @ q  # cosine similarity
        idxs = np.argsort(-sims)[:top_k]
        out = [self._chunks[int(i)] for i in idxs]
        return out
