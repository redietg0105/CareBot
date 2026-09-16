"""
Ingest the clinic knowledge-base documents in data/ into a local vector
index. This mirrors the "ETL pipeline that ingests, cleans, and chunks
clinical source data into the retrieval layer" step of CareBot.

Usage:
    python src/ingest.py

Produces:
    data/index.json  -- list of {"id", "source", "text", "embedding"}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_PATH = DATA_DIR / "index.json"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 20


def load_documents() -> list[dict]:
    """Read every .txt file in data/ as a source document."""
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        docs.append({"source": path.name, "text": text})
    if not docs:
        raise FileNotFoundError(f"No .txt knowledge-base files found in {DATA_DIR}")
    return docs


def chunk_text(text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word-count chunks so retrieval can
    return focused passages instead of whole documents."""
    words = text.split()
    if len(words) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_index() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    client = OpenAI(api_key=api_key)

    documents = load_documents()
    records = []
    chunk_id = 0
    all_chunks = []
    chunk_meta = []

    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append(chunk)
            chunk_meta.append({"id": chunk_id, "source": doc["source"], "text": chunk})
            chunk_id += 1

    # Batch embeddings in groups of 16 to keep requests small.
    batch_size = 16
    embeddings: list[list[float]] = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        embeddings.extend(embed_batch(client, batch))

    for meta, embedding in zip(chunk_meta, embeddings):
        records.append({**meta, "embedding": embedding})

    INDEX_PATH.write_text(json.dumps(records), encoding="utf-8")
    print(f"Indexed {len(records)} chunks from {len(documents)} documents -> {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
