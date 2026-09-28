"""
RAG — Document Processor
Splits repository knowledge text into chunks for embedding.
Uses a smaller chunk size than general text so that each chunk
covers a focused concept (one class, one file, one method group)
rather than dozens of mixed entries — improving retrieval precision.
"""

from typing import List, Dict
from config import CHUNK_OVERLAP

# Use a smaller chunk size for code knowledge (overrides config for this module)
_CODE_CHUNK_SIZE = 120   # words per chunk — keeps each chunk focused


def chunk_text(text: str, source: str = "") -> List[Dict]:
    """
    Split `text` into overlapping chunks of _CODE_CHUNK_SIZE words.
    Returns list of {text, source, chunk_index}.
    """
    if not text.strip():
        return []

    words  = text.split()
    chunks = []
    start  = 0
    idx    = 0

    while start < len(words):
        end   = start + _CODE_CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append({"text": chunk, "source": source, "chunk_index": idx})
        idx   += 1
        start  = end - CHUNK_OVERLAP

    return chunks


def process_knowledge_for_rag(shared_knowledge, repo_name: str) -> List[Dict]:
    """
    Convert SharedKnowledge for one repo into RAG-ready chunks.
    """
    text = shared_knowledge.flat_text_for_rag(repo_name)
    return chunk_text(text, source=repo_name)


def process_all_repos(shared_knowledge) -> List[Dict]:
    """Process all repos in SharedKnowledge into a flat chunk list."""
    all_chunks = []
    for repo_name in shared_knowledge.all_repos():
        all_chunks.extend(process_knowledge_for_rag(shared_knowledge, repo_name))
    return all_chunks
