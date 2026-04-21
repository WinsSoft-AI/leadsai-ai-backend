from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from pipeline.embedder import embed_texts, get_gemini_client as get_embed_client
from pipeline.enricher import chunk_structured_page
from pipeline.data_store import ScrapedDataStore, DATA_DIR
from rag.faiss_store import FaissVectorStore
from rag.bm25_store import BM25Store
from rag.hybrid import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

GEMINI_MODEL_EMBEDDING = os.getenv("GEMINI_MODEL_EMBEDDING", "gemini-embedding-001")

# ---------------------------------------------------------------------------
# Multi-domain vector store registry
# ---------------------------------------------------------------------------
domain_stores: Dict[str, FaissVectorStore] = {}
bm25_stores: Dict[str, BM25Store] = {}


def get_store(domain: str) -> FaissVectorStore | None:
    return domain_stores.get(domain)


def get_bm25(domain: str) -> BM25Store | None:
    return bm25_stores.get(domain)


def ensure_store(domain: str, dim: int) -> FaissVectorStore:
    if domain not in domain_stores:
        domain_stores[domain] = FaissVectorStore(dim=dim)
    return domain_stores[domain]


def rebuild_bm25(domain: str) -> None:
    """Build/rebuild BM25 index from the FAISS store's chunks."""
    faiss_store = domain_stores.get(domain)
    if faiss_store and faiss_store.size > 0:
        bm25 = BM25Store()
        bm25.build(faiss_store._chunks)
        bm25_stores[domain] = bm25
        logger.info(f"📖 [{domain}] BM25 index built: {bm25.size} docs")


def reset_store(domain: str) -> None:
    domain_stores.pop(domain, None)
    bm25_stores.pop(domain, None)


async def process_and_index(domain: str, enriched_data: List[Dict[str, Any]]) -> int:
    """Chunk, embed, and index Gemini-enriched data into a domain-specific store."""
    if not enriched_data:
        return 0

    all_chunks: List[Dict[str, Any]] = []
    for item in enriched_data:
        structured = item.get("structured", {})
        url = item.get("url", "")
        if structured:
            chunks = chunk_structured_page(page_url=url, structured=structured)
            all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    logger.info(f"📊 [{domain}] Embedding {len(all_chunks)} chunks...")
    try:
        embed_client = get_embed_client()
        texts = [c["content"] for c in all_chunks]
        matrix = embed_texts(embed_client, texts=texts, model=GEMINI_MODEL_EMBEDDING)

        if matrix.size > 0:
            if matrix.shape[0] < len(all_chunks):
                logger.warning(f"Partial embedding: {matrix.shape[0]}/{len(all_chunks)}")
                all_chunks = all_chunks[: matrix.shape[0]]

            store = ensure_store(domain, dim=matrix.shape[1])
            store.add(matrix, all_chunks)
            rebuild_bm25(domain)   # Build BM25 after FAISS
            return len(all_chunks)
    except Exception as e:
        logger.error(f"Embedding/Indexing failed for {domain}: {e}")
    return 0


async def load_all_domains():
    """Load ALL existing enriched data from disk into per-domain stores."""
    if DATA_DIR.exists():
        domains = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
        for domain in domains:
            store = ScrapedDataStore(domain)
            enriched = store.load_all_enriched()
            if enriched:
                logger.info(f"📁 Loading {domain}...")
                count = await process_and_index(domain, enriched)
                if count > 0:
                    logger.info(f"   ✅ {domain}: {count} chunks indexed")


def hybrid_search(domain: str, question: str, top_k: int = 5) -> List[Tuple[Any, float]]:
    """Perform hybrid search (FAISS + BM25) for a given domain and question."""
    vs = get_store(domain)
    if not vs or vs.size == 0:
        return []

    embed_client = get_embed_client()
    q_vec = embed_texts(embed_client, [question], model=GEMINI_MODEL_EMBEDDING)
    q_vec = np.squeeze(q_vec, axis=0)

    # Fetch more candidates from each retriever for better fusion
    retrieval_k = top_k * 3

    semantic_results = vs.search(q_vec, top_k=retrieval_k)

    # BM25 keyword search
    bm25 = get_bm25(domain)
    keyword_results = bm25.search(question, top_k=retrieval_k) if bm25 else []

    # Reciprocal Rank Fusion
    if keyword_results:
        results = reciprocal_rank_fusion(
            semantic_results, keyword_results, top_k=top_k
        )
        logger.info(
            f"🔀 Hybrid: {len(semantic_results)} semantic + {len(keyword_results)} BM25 → {len(results)} fused"
        )
    else:
        results = semantic_results[:top_k]
        logger.info(f"🔍 Semantic only: {len(results)} results")

    return results
