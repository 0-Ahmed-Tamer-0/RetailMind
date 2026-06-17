"""rag/indexer.py — Build FAISS index from all CSV outputs

What this file does
────────────────────
Reads every output CSV and converts each meaningful row into a
human-readable text chunk. Each chunk is then embedded with MiniLM
and stored in a FAISS index for fast similarity search at query time.

Why human-readable chunks instead of raw CSV rows?
───────────────────────────────────────────────────
The LLM reads retrieved chunks as context. If chunks are raw numbers
("22086,C,1680,3527.6,0.5,...") the LLM produces worse answers than
if chunks are natural sentences ("Product 22086 is a C-class item with
current stock of 1,680 units..."). Pre-formatting at index time means
the retriever returns LLM-ready text, not data that needs interpretation.

Chunk design per CSV
─────────────────────
Each CSV gets its own chunking strategy based on what a retail owner
would want to know from that data source. The goal is one chunk =
one self-contained business fact that stands alone without context.

Index structure saved to disk
──────────────────────────────
  rag_index/
    index.faiss      ← FAISS flat L2 index (vectors)
    chunks.json      ← list of {text, source, metadata} dicts
    
Flat L2 vs IVF:
  We use IndexFlatIP (inner product on normalized vectors = cosine sim).
  IVF would be faster for millions of vectors but we have <500 chunks —
  flat search is instant at this scale and has no quantization error.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT_DIR  = Path(__file__).resolve().parents[2]
OUT_DIR   = ROOT_DIR / "data" / "outputs"
INDEX_DIR = ROOT_DIR / "data" / "rag_index"


# ── Chunk builders ────────────────────────────────────────────────────────────

def _chunks_sales(df: pd.DataFrame) -> list[dict]:
    """
    One chunk per product summarising its full forecast window.
    Aggregating all dates per product avoids 4001 tiny single-day chunks
    that would dilute retrieval — a query about product 85123A should
    retrieve one comprehensive chunk, not 400 date-specific ones.
    """
    chunks = []
    for pid, grp in df.groupby("product_id"):
        total_predicted = grp["predicted_sales"].sum()
        avg_low         = grp["predicted_low"].mean()
        avg_high        = grp["predicted_high"].mean()
        days            = len(grp)
        latest_actual   = grp.sort_values("date")["actual_sales"].iloc[-1]

        text = (
            f"Sales forecast for product {pid}: "
            f"predicted {total_predicted:,.0f} units over {days} days, "
            f"confidence range {avg_low:,.0f}–{avg_high:,.0f} units. "
            f"Most recent actual sales: {latest_actual:,.1f} units."
        )
        chunks.append({"text": text, "source": "sales_forecast", "product_id": str(pid)})
    return chunks


def _chunks_inventory(inv_df: pd.DataFrame, disc_df: pd.DataFrame) -> list[dict]:
    """
    One chunk per product combining inventory + discount decision.
    Merging both into one chunk means a query like 'what should I do
    about product X' retrieves one complete action summary.
    """
    merged = inv_df.merge(disc_df[["product_id", "discount_action", "discount_rate"]],
                          on="product_id", how="left")
    chunks = []
    for _, row in merged.iterrows():
        disc_part = (
            f"Discount recommended: {row['discount_rate']}% markdown."
            if row["discount_action"] == "DISCOUNT"
            else "No discount needed."
        )
        text = (
            f"Inventory status for product {row['product_id']} "
            f"(ABC class {row['abc_class']}): "
            f"current stock {row['current_stock']:,} units, "
            f"7-day forecast demand {row['predicted_demand_7d']:,.0f} units. "
            f"Decision: {row['decision']}. "
            f"{row['reason']}. "
            f"{disc_part}"
        )
        chunks.append({
            "text":       text,
            "source":     "inventory",
            "product_id": str(row["product_id"]),
            "decision":   row["decision"],
            "abc_class":  row["abc_class"],
        })
    return chunks


def _chunks_segments(seg_df: pd.DataFrame) -> list[dict]:
    """One chunk per customer segment."""
    chunks = []
    for _, row in seg_df.iterrows():
        text = (
            f"Customer segment '{row['segment_name']}': "
            f"{row['customer_count']:,} customers, "
            f"average recency {row['avg_recency']:.0f} days since last purchase, "
            f"average purchase frequency {row['avg_frequency']:.1f} orders, "
            f"average spend £{row['avg_monetary']:,.2f}."
        )
        chunks.append({
            "text":         text,
            "source":       "segments",
            "segment_name": row["segment_name"],
            "cluster_id":   int(row["cluster_id"]),
        })
    return chunks


def _chunks_review_topics(topics_df: pd.DataFrame) -> list[dict]:
    """
    Two chunks: one summarising ALL negative topics, one for ALL positive.

    Why aggregate instead of one chunk per topic?
    ──────────────────────────────────────────────
    A question like "are customers satisfied?" needs the LLM to see the
    full picture — all complaint themes together and all praise themes
    together — so it can synthesize a complete answer.

    With individual topic chunks, retrieval might return only 1-2 topics
    (the ones closest to the query vector) and the LLM gives a partial answer.

    With aggregated chunks, one retrieval hit gives the LLM everything it
    needs to say: "The main complaints are X, Y, Z. The main praises are A, B, C."
    That is the natural business answer format.
    """
    chunks = []
    for sentiment in ["negative", "positive"]:
        sub = topics_df[topics_df["sentiment"] == sentiment].sort_values(
            "review_count", ascending=False
        )
        if sub.empty:
            continue

        label = "complaints" if sentiment == "negative" else "praises"
        total = sub["review_count"].sum()

        # Build a natural summary: "Theme (N reviews): key words"
        topic_lines = []
        for _, row in sub.iterrows():
            pct = row["review_count"] / total * 100
            topic_lines.append(
                f"- {row['theme']} ({row['review_count']:,} reviews, {pct:.0f}%): "
                f"{row['top_words']}"
            )

        text = (
            f"Customer {label} summary ({sentiment} reviews, "
            f"{total:,} total):" + "".join(topic_lines)
        )
        chunks.append({
            "text":      text,
            "source":    "review_topics",
            "sentiment": sentiment,
        })
    return chunks


def _chunks_review_insights(insights_df: pd.DataFrame) -> list[dict]:
    """
    Monthly trend chunks — recent months only (last 24) to keep index lean.
    Early months (2000-2006) have very few reviews and add noise.
    """
    df = insights_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(24)  # last 24 months

    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Review trend for {row['date'].strftime('%B %Y')}: "
            f"average rating {row['avg_rating']:.2f}/5, "
            f"VADER sentiment {row['avg_vader']:.3f}, "
            f"{row['positive_ratio']*100:.1f}% positive reviews, "
            f"{row['negative_ratio']*100:.1f}% negative, "
            f"based on {row['review_count']:,} reviews."
        )
        chunks.append({
            "text":   text,
            "source": "review_insights",
            "date":   str(row["date"].date()),
        })
    return chunks


def _chunks_review_keywords(kw_df: pd.DataFrame) -> list[dict]:
    """
    Two chunks: one summarising top negative keywords, one positive.
    Single aggregated chunk is better than 40 individual keyword chunks
    — a query about complaints retrieves one complete picture.
    """
    chunks = []
    for sentiment in ["negative", "positive"]:
        sub = kw_df[kw_df["sentiment"] == sentiment].head(10)
        if sub.empty:
            continue
        words = ", ".join(sub["keyword"].tolist())
        label = "complaints" if sentiment == "negative" else "praise"
        text = (
            f"Top keywords from {sentiment} customer reviews ({label}): "
            f"{words}."
        )
        chunks.append({"text": text, "source": "review_keywords", "sentiment": sentiment})
    return chunks


def _chunks_cv(cv_df: pd.DataFrame) -> list[dict]:
    """One chunk per zone summarising traffic statistics."""
    chunks = []
    for zone_id, grp in cv_df.groupby("zone_id"):
        avg_count  = grp["people_count"].mean()
        peak_count = grp["people_count"].max()
        total_obs  = len(grp)
        text = (
            f"Foot traffic in zone {zone_id}: "
            f"average {avg_count:.1f} people per frame, "
            f"peak count {peak_count} people, "
            f"observed across {total_obs} frames."
        )
        chunks.append({
            "text":    text,
            "source":  "cv_foot_traffic",
            "zone_id": str(zone_id),
        })
    return chunks


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(
    out_dir:   Path = OUT_DIR,
    index_dir: Path = INDEX_DIR,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[faiss.Index, list[dict]]:
    """
    Build FAISS index from all output CSVs.

    Steps
    -----
    1. Load each CSV and build human-readable chunks
    2. Embed all chunks with MiniLM
    3. L2-normalize embeddings (enables cosine similarity via inner product)
    4. Build FAISS IndexFlatIP and add all vectors
    5. Save index + chunk metadata to disk

    Returns
    -------
    (faiss_index, chunks_list)
    """
    print("[indexer] Loading output CSVs…")

    all_chunks: list[dict] = []

    # Sales forecast
    sales_path = out_dir / "sales_forecast.csv"
    if sales_path.exists():
        all_chunks += _chunks_sales(pd.read_csv(sales_path))
        print(f"  sales_forecast      → {len(all_chunks)} chunks so far")

    # Inventory + discount
    inv_path  = out_dir / "inventory_recommendations.csv"
    disc_path = out_dir / "discount_recommendations.csv"
    if inv_path.exists() and disc_path.exists():
        prev = len(all_chunks)
        all_chunks += _chunks_inventory(
            pd.read_csv(inv_path), pd.read_csv(disc_path)
        )
        print(f"  inventory+discount  → +{len(all_chunks)-prev} chunks")

    # Customer segments
    seg_path = out_dir / "customer_segments.csv"
    if seg_path.exists():
        prev = len(all_chunks)
        all_chunks += _chunks_segments(pd.read_csv(seg_path))
        print(f"  customer_segments   → +{len(all_chunks)-prev} chunks")

    # Review topics (primary insight source)
    topics_path = out_dir / "review_topics.csv"
    if topics_path.exists():
        prev = len(all_chunks)
        all_chunks += _chunks_review_topics(pd.read_csv(topics_path))
        print(f"  review_topics       → +{len(all_chunks)-prev} chunks")

    # Review trends
    insights_path = out_dir / "review_insights.csv"
    if insights_path.exists():
        prev = len(all_chunks)
        all_chunks += _chunks_review_insights(pd.read_csv(insights_path))
        print(f"  review_insights     → +{len(all_chunks)-prev} chunks")

    # Review keywords
    kw_path = out_dir / "review_keywords.csv"
    if kw_path.exists():
        prev = len(all_chunks)
        all_chunks += _chunks_review_keywords(pd.read_csv(kw_path))
        print(f"  review_keywords     → +{len(all_chunks)-prev} chunks")

    # CV foot traffic
    cv_path = out_dir / "cv_foot_traffic.csv"
    if cv_path.exists():
        prev = len(all_chunks)
        all_chunks += _chunks_cv(pd.read_csv(cv_path))
        print(f"  cv_foot_traffic     → +{len(all_chunks)-prev} chunks")

    print(f"\n[indexer] Total chunks: {len(all_chunks)}")

    # ── Embed ──
    print(f"[indexer] Embedding with {model_name}…")
    embedder = SentenceTransformer(model_name)
    texts    = [c["text"] for c in all_chunks]
    vectors  = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 norm → cosine sim = dot product
        show_progress_bar=True,
    )

    # ── Build FAISS index ──
    dim   = vectors.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatIP(dim)  # inner product on normalized = cosine similarity
    index.add(vectors.astype(np.float32))
    print(f"[indexer] FAISS index built: {index.ntotal} vectors, dim={dim}")

    # ── Save ──
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "chunks.json").write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[indexer] Saved → {index_dir}/index.faiss + chunks.json")

    return index, all_chunks


if __name__ == "__main__":
    build_index()
