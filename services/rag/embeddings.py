from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load and cache the sentence transformer model."""

    return SentenceTransformer(MODEL_NAME)


def get_embedding(text: str) -> np.ndarray:
    """Generate a normalized embedding for a single text."""

    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return np.asarray(embedding, dtype="float32")


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Generate normalized embeddings for a batch of texts."""

    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype="float32")
