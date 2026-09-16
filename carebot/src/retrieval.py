"""Retrieval layer: loads data/index.json and returns the top-k most
relevant chunks for a query using cosine similarity."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_PATH = DATA_DIR / "index.json"
EMBEDDING_MODEL = "text-embedding-3-small"


def load_index() -> tuple[list[dict], np.ndarray]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "data/index.json not found. Run `python src/ingest.py` first."
        )
    records = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    matrix = np.array([r["embedding"] for r in records], dtype=np.float32)
    return records, matrix


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix_norm @ query_norm


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return np.array(response.data[0].embedding, dtype=np.float32)


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Return the top_k chunks most relevant to the query, each with a
    similarity score, so the caller can decide whether the match is
    strong enough to answer from."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    records, matrix = load_index()
    query_vec = embed_query(client, query)
    scores = cosine_similarity(query_vec, matrix)

    ranked_idx = np.argsort(-scores)[:top_k]
    results = []
    for idx in ranked_idx:
        results.append(
            {
                "source": records[idx]["source"],
                "text": records[idx]["text"],
                "score": float(scores[idx]),
            }
        )
    return results
