"""
BM25 keyword search index — sits alongside the FAISS semantic index.

Stores the same StoredChunk list so that results from both retrieval
methods reference the identical objects and can be merged easily.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi

from rag.faiss_store import StoredChunk


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"\w+", text.lower())


class BM25Store:
    """In-memory BM25 index over chunk content."""

    def __init__(self) -> None:
        self._chunks: List[StoredChunk] = []
        self._corpus_tokens: List[List[str]] = []
        self._index: BM25Okapi | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: List[StoredChunk]) -> None:
        """Build (or rebuild) the BM25 index from a list of StoredChunks."""
        self._chunks = list(chunks)
        self._corpus_tokens = [_tokenize(c.content) for c in self._chunks]
        if self._corpus_tokens:
            self._index = BM25Okapi(self._corpus_tokens)
        else:
            self._index = None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[StoredChunk, float]]:
        """Return top-k chunks ranked by BM25 score."""
        if self._index is None or not self._chunks:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)

        # Get top-k indices by score (descending)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[Tuple[StoredChunk, float]] = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._chunks[idx], float(scores[idx])))
        return results
