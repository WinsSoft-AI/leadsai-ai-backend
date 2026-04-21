from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np


@dataclass
class StoredChunk:
    id: int
    source: str
    type: str
    content: str
    metadata: Dict[str, Any]


class FaissVectorStore:
    """
    Minimal in-memory FAISS index for prototype purposes.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self._chunks: List[StoredChunk] = []
        self._next_id = 0

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        assert vectors.shape[0] == len(chunks), "Vectors and chunks length mismatch"
        if vectors.dtype != "float32":
            vectors = vectors.astype("float32")

        start_id = self._next_id
        ids = np.arange(start_id, start_id + vectors.shape[0])

        self.index.add(vectors)

        for i, chunk in zip(ids, chunks):
            self._chunks.append(
                StoredChunk(
                    id=int(i),
                    source=chunk.get("source", ""),
                    type=chunk.get("type", "general"),
                    content=chunk.get("content", ""),
                    metadata=chunk.get("metadata") or {},
                )
            )
            self._next_id += 1

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[StoredChunk, float]]:
        if self.size == 0:
            return []
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        if query_vector.dtype != "float32":
            query_vector = query_vector.astype("float32")

        distances, indices = self.index.search(query_vector, top_k)
        results: List[Tuple[StoredChunk, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[idx], float(dist)))
        return results

