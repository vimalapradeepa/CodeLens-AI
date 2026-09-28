"""
RAG — Embeddings
Generates sentence embeddings using sentence-transformers (local, no API key needed).
Falls back gracefully if sentence-transformers is not installed.
"""

import logging
from typing import List, Dict, Optional

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model = None
_MODEL_AVAILABLE = False


def _load_model():
    global _model, _MODEL_AVAILABLE
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
        _MODEL_AVAILABLE = True
        logger.info("Embedding model loaded: %s", EMBEDDING_MODEL)
    except ImportError:
        logger.warning("sentence-transformers not installed — embeddings unavailable.")
    except Exception as exc:
        logger.warning("Failed to load embedding model: %s", exc)
    return _model


def embed_texts(texts: List[str]) -> Optional[List]:
    """Return a list of numpy arrays, or None if unavailable."""
    model = _load_model()
    if model is None:
        return None
    try:
        return model.encode(texts, show_progress_bar=False).tolist()
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        return None


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Add 'embedding' field to each chunk dict.
    chunks: [{text, source, chunk_index}, …]
    Returns same list with embedding added (or None if unavailable).
    """
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i] if embeddings else None
    return chunks


def is_available() -> bool:
    _load_model()
    return _MODEL_AVAILABLE
