"""output_generator.py — Reviews analysis pipeline
Produces 3 output files:
  review_insights.csv  — monthly sentiment trends
  review_keywords.csv  — discriminative keywords per sentiment
  review_topics.csv    — LDA topics per sentiment (the useful insights)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.reviews_analysis.sentiment_model import (
    clean_amazon,
    label_sentiment,
    extract_keywords,
    extract_topics,
    compute_trends,
)

DEFAULT_DATA_PATH    = ROOT_DIR / "data" / "raw" / "reviews_analysis" / "Reviews.csv"
DEFAULT_OUT_INSIGHTS = ROOT_DIR / "data" / "outputs" / "review_insights.csv"
DEFAULT_OUT_KEYWORDS = ROOT_DIR / "data" / "outputs" / "review_keywords.csv"
DEFAULT_OUT_TOPICS   = ROOT_DIR / "data" / "outputs" / "review_topics.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv",    default=str(DEFAULT_DATA_PATH))
    p.add_argument("--out_insights", default=str(DEFAULT_OUT_INSIGHTS))
    p.add_argument("--out_keywords", default=str(DEFAULT_OUT_KEYWORDS))
    p.add_argument("--out_topics",   default=str(DEFAULT_OUT_TOPICS))
    p.add_argument("--sample_n",     type=int, default=50_000)
    p.add_argument("--n_topics",     type=int, default=5,
                   help="LDA topics per sentiment class (default 5)")
    p.add_argument("--resample",     default="ME")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Load + clean ──
    print(f"[reviews] Loading {args.input_csv}…")
    df_raw = pd.read_csv(args.input_csv, engine="python", on_bad_lines="skip")
    print(f"[reviews] Raw rows: {len(df_raw):,}")

    df = clean_amazon(df_raw, sample_n=args.sample_n)
    print(f"[reviews] Sampled: {len(df):,} rows")
    print(f"[reviews] Score distribution:\n{df['Score'].value_counts().sort_index()}\n")

    # ── 2. Sentiment labelling ──
    df = label_sentiment(df)
    print(f"[reviews] Sentiment counts:\n{df['sentiment'].value_counts()}\n")

    # ── 3. Trends ──
    trends = compute_trends(df, resample=args.resample)

    # ── 4. Keywords (for dashboard word cloud) ──
    kw_neg = extract_keywords(df, "negative", top_n=20)
    kw_pos = extract_keywords(df, "positive", top_n=20)
    keywords = pd.concat([kw_neg, kw_pos], ignore_index=True)

    # ── 5. LDA topics (for chatbot insights) ──
    topics_neg = extract_topics(df, "negative", n_topics=args.n_topics)
    topics_pos = extract_topics(df, "positive", n_topics=args.n_topics)
    topics = pd.concat([topics_neg, topics_pos], ignore_index=True)

    # ── 6. Save all three ──
    out_base = Path(args.out_insights).parent
    out_base.mkdir(parents=True, exist_ok=True)

    trends.to_csv(args.out_insights, index=False)
    keywords.to_csv(args.out_keywords, index=False)
    topics.to_csv(args.out_topics, index=False)

    print(f"\n[reviews] Saved → {args.out_insights}")
    print(f"[reviews] Saved → {args.out_keywords}")
    print(f"[reviews] Saved → {args.out_topics}")

    print("\n── Negative topics ──")
    print(topics[topics["sentiment"] == "negative"]
          [["theme", "top_words", "review_count"]].to_string(index=False))

    print("\n── Positive topics ──")
    print(topics[topics["sentiment"] == "positive"]
          [["theme", "top_words", "review_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
