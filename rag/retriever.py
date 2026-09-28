"""
RAG — Retriever
Builds a FAISS vector store from embedded chunks and retrieves top-k relevant chunks.
Falls back to keyword-based retrieval when FAISS/embeddings are unavailable.
"""

import os
import json
import logging
import pickle
from typing import List, Dict, Optional

from config import VECTOR_STORE_PATH, TOP_K_RETRIEVAL
from rag.embeddings import embed_texts, is_available as embeddings_available

logger = logging.getLogger(__name__)

_STORE_INDEX  = os.path.join(VECTOR_STORE_PATH, "index.faiss")
_STORE_CHUNKS = os.path.join(VECTOR_STORE_PATH, "chunks.pkl")


class VectorRetriever:
    """
    FAISS-backed retriever with keyword fallback.
    """

    def __init__(self):
        self.chunks: List[Dict] = []
        self.index = None
        self._faiss_available = False
        self._try_load_faiss()

    def _try_load_faiss(self):
        try:
            import faiss
            self._faiss = faiss
            self._faiss_available = True
        except ImportError:
            logger.warning("faiss-cpu not installed — using keyword fallback retrieval.")

    # ── Build / persist ───────────────────────────────────────

    def build(self, embedded_chunks: List[Dict]):
        """Build index from a list of embedded chunk dicts."""
        self.chunks = [c for c in embedded_chunks if c.get("embedding") is not None]

        if not self.chunks:
            logger.warning("No embedded chunks — retriever in keyword-only mode.")
            self.chunks = embedded_chunks
            return

        if not self._faiss_available:
            return

        import numpy as np
        vectors = np.array([c["embedding"] for c in self.chunks], dtype="float32")
        dim = vectors.shape[1]
        self.index = self._faiss.IndexFlatIP(dim)  # inner product (cosine after normalise)
        self._faiss.normalize_L2(vectors)
        self.index.add(vectors)
        logger.info("FAISS index built: %d vectors, dim=%d", len(self.chunks), dim)

    def save(self):
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        if self.index and self._faiss_available:
            self._faiss.write_index(self.index, _STORE_INDEX)
        with open(_STORE_CHUNKS, "wb") as fh:
            pickle.dump(self.chunks, fh)
        logger.info("Vector store saved → %s", VECTOR_STORE_PATH)

    def load(self) -> bool:
        if not os.path.exists(_STORE_CHUNKS):
            return False
        with open(_STORE_CHUNKS, "rb") as fh:
            self.chunks = pickle.load(fh)
        if self._faiss_available and os.path.exists(_STORE_INDEX):
            self.index = self._faiss.read_index(_STORE_INDEX)
        logger.info("Vector store loaded: %d chunks", len(self.chunks))
        return True

    # ── Retrieval ─────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        if not self.chunks:
            return []
        if self.index is not None and self._faiss_available:
            return self._retrieve_faiss(query, top_k)
        return self._retrieve_keyword(query, top_k)

    def _retrieve_faiss(self, query: str, top_k: int) -> List[Dict]:
        import numpy as np
        qvec = embed_texts([query])
        if not qvec:
            return self._retrieve_keyword(query, top_k)
        q = np.array(qvec, dtype="float32")
        self._faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            chunk.pop("embedding", None)
            results.append(chunk)
        return results

    def _retrieve_keyword(self, query: str, top_k: int) -> List[Dict]:
        """Simple TF-style keyword retrieval fallback."""
        import re
        q_tokens = set(re.split(r'\W+', query.lower()))
        scored = []
        for chunk in self.chunks:
            text_tokens = set(re.split(r'\W+', chunk["text"].lower()))
            overlap = len(q_tokens & text_tokens)
            if overlap > 0:
                c = dict(chunk)
                c["score"] = overlap
                c.pop("embedding", None)
                scored.append(c)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
