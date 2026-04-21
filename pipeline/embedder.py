from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from google import genai

logger = logging.getLogger(__name__)

# Maximum number of texts per embedding API call
BATCH_SIZE = 20


def get_gemini_client(api_key: str | None = None) -> genai.Client:
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def embed_texts(
    client: genai.Client,
    texts: Sequence[str],
    model: str = "gemini-embedding-001",
) -> np.ndarray:
    """
    Return a 2D numpy array (n_texts, dim) with embeddings.
    Handles batching to avoid API limits on large payloads.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    all_vectors = []
    text_list = list(texts)

    for i in range(0, len(text_list), BATCH_SIZE):
        batch = text_list[i : i + BATCH_SIZE]
        logger.info(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} texts)")
        try:
            result = client.models.embed_content(model=model, contents=batch)
            vectors = [np.array(e.values, dtype="float32") for e in result.embeddings]  # type: ignore[attr-defined]
            all_vectors.extend(vectors)
        except Exception as e:
            logger.warning(f"Embedding batch failed: {e}")
            # Skip this batch rather than crashing
            continue

    if not all_vectors:
        return np.zeros((0, 0), dtype="float32")

    return np.vstack(all_vectors)
