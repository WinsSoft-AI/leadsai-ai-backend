"""
Hybrid search — merges FAISS semantic results and BM25 keyword results
using Reciprocal Rank Fusion (RRF) to produce a single re-ranked list.

RRF Score = Σ  1 / (k + rank_i)    for each retriever i

This method is ranking-agnostic: it doesn't depend on the absolute score
values from either retriever, only their relative ordering.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from rag.faiss_store import StoredChunk

# RRF constant — controls how much weight is given to lower-ranked items.
# Higher k → more uniform weighting; lower k → top results dominate.
RRF_K = 60


def reciprocal_rank_fusion(
    semantic_results: List[Tuple[StoredChunk, float]],
    keyword_results: List[Tuple[StoredChunk, float]],
    top_k: int = 5,
    semantic_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> List[Tuple[StoredChunk, float]]:
    """
    Merge two ranked result lists using weighted Reciprocal Rank Fusion.

    Args:
        semantic_results: (chunk, distance) from FAISS — lower distance = better.
        keyword_results:  (chunk, bm25_score) from BM25 — higher score = better.
        top_k:            How many final results to return.
        semantic_weight:  Weight multiplier for semantic retriever.
        keyword_weight:   Weight multiplier for keyword retriever.

    Returns:
        Merged list of (chunk, rrf_score), sorted by score descending.
    """
    # Accumulate RRF scores keyed by chunk id
    scores: Dict[int, float] = {}
    chunk_map: Dict[int, StoredChunk] = {}

    # Semantic results (already sorted by distance ascending — rank 1 = best)
    for rank, (chunk, _dist) in enumerate(semantic_results, start=1):
        cid = chunk.id
        chunk_map[cid] = chunk
        scores[cid] = scores.get(cid, 0.0) + semantic_weight * (1.0 / (RRF_K + rank))

    # BM25 results (already sorted by score descending — rank 1 = best)
    for rank, (chunk, _score) in enumerate(keyword_results, start=1):
        cid = chunk.id
        chunk_map[cid] = chunk
        scores[cid] = scores.get(cid, 0.0) + keyword_weight * (1.0 / (RRF_K + rank))

    # Sort by fused score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [(chunk_map[cid], score) for cid, score in ranked]
