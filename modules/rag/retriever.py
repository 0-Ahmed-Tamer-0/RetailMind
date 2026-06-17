"""rag/retriever.py — Query the FAISS index

What this file does
────────────────────
Takes a user question, embeds it with the same MiniLM model used
at index time, searches the FAISS index for the most similar chunks,
and returns them ranked by cosine similarity.

Why same model at query and index time?
────────────────────────────────────────
Semantic search only works when query and document vectors live in
the same embedding space. Using a different model at query time would
be like searching a French dictionary with an English query — the
geometry wouldn't match. MiniLM is loaded once and cached globally
so repeated queries don't pay the model-load cost.

Source filtering
─────────────────
Queries often imply a specific data source:
  "what should we reorder"     → inventory source most relevant
  "what are customers saying"  → review_topics most relevant
  "show me zone traffic"       → cv_foot_traffic most relevant

We support optional source_filter to restrict retrieval to specific
CSVs. When None, all chunks are searched (general queries benefit
from cross-source retrieval — e.g. a question about a product might
pull from both sales forecast and inventory).
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR  = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT_DIR / "data" / "rag_index"

# ── Global cache — loaded once per process ───────────────────────────────────
_embedder: SentenceTransformer | None = None
_index:    faiss.Index | None         = None
_chunks:   list[dict]  | None         = None


def _load_index(index_dir: Path = INDEX_DIR):
    """Load FAISS index + chunks from disk into module-level cache."""
    global _index, _chunks
    if _index is not None:
        return

    index_path  = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}.\n"
            f"Run: python modules/rag/indexer.py"
        )

    _index  = faiss.read_index(str(index_path))
    _chunks = json.loads(chunks_path.read_text(encoding="utf-8"))


def _get_embedder() -> SentenceTransformer:
    """Load MiniLM once, reuse forever."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def retrieve(
    query:         str,
    top_k:         int = 5,
    source_filter: list[str] | None = None,
) -> list[dict]:
    """
    Retrieve top-k chunks most semantically similar to the query.

    Parameters
    ----------
    query         : user's natural language question
    top_k         : number of chunks to return (default 5)
    source_filter : restrict to specific sources e.g. ["inventory", "sales_forecast"]
                    None = search all sources

    Returns
    -------
    List of chunk dicts, each with:
        text       — the human-readable chunk text (fed to LLM as context)
        source     — which CSV this came from
        score      — cosine similarity score (0–1, higher = more relevant)
        + any source-specific metadata fields

    How retrieval works
    ───────────────────
    1. Embed query with MiniLM → 384-dim vector, L2-normalised
    2. FAISS inner product search → cosine similarity scores
    3. If source_filter set: re-rank results keeping only matching sources,
       then take top_k from those
    4. Return ranked list with scores attached

    Why not filter before FAISS search?
    Pre-filtering would require a filtered sub-index or sequential scan.
    Searching the full index then filtering is faster for our scale
    (<500 chunks) and simpler to maintain.
    """
    _load_index()

    embedder = _get_embedder()
    query_vec = embedder.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).reshape(1, -1).astype(np.float32)

    # Search — retrieve more than top_k if we'll filter after
    search_k = min(len(_chunks), top_k * 4 if source_filter else top_k)
    scores, indices = _index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(_chunks[idx])
        chunk["score"] = float(score)

        if source_filter and chunk.get("source") not in source_filter:
            continue

        results.append(chunk)
        if len(results) >= top_k:
            break

    return results


def retrieve_by_source(source: str, top_n: int = 5) -> list[dict]:
    """
    Return all chunks from a specific source, no query needed.
    Useful for 'show me all inventory decisions' type queries
    where no semantic matching is needed — just dump the source.
    """
    _load_index()
    matched = [c for c in _chunks if c.get("source") == source]
    return matched[:top_n]
