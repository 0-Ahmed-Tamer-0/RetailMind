"""induce_stopwords.py — Offline corpus-based stopword induction
Run ONCE on the full Amazon Fine Food Reviews dataset.
Saves induced_stopwords.txt which sentiment_model.py loads at runtime.

Usage
-----
python modules/reviews_analysis/induce_stopwords.py \
    --input_csv "data/raw/reviews_analysis/Reviews.csv" \
    --output_txt "data/raw/reviews_analysis/induced_stopwords.txt"

How it works
────────────
We treat a word as a domain stopword if it meets ANY of these criteria:

Criterion 1 — High document frequency (DF > threshold)
  Words appearing in more than 15% of all reviews are so common
  they carry no discriminating information for topic modeling.
  Example: "good" appears in ~65% of reviews → pure noise for LDA.

  This is the IDF part of TF-IDF applied at corpus level:
  low IDF = high DF = not informative.

Criterion 2 — Low inter-class discriminative power (diff < threshold)
  A word that appears at similar rates in positive AND negative reviews
  is not associated with either sentiment → it's a neutral filler word.
  
  discriminative_power = |freq_in_positive - freq_in_negative|
  
  Example:
    "food"  → pos: 42%, neg: 39% → diff: 3%  → stopword (not discriminative)
    "stale" → pos:  1%, neg: 28% → diff: 27% → keep (strongly negative signal)

  This is Chi-squared feature selection simplified to one interpretable number.

Result: a data-driven stopword list that reflects YOUR corpus,
not generic English stopwords or a manually maintained domain list.
The list regenerates automatically if you get new data.
"""

from __future__ import annotations

import argparse
import re
import string
from pathlib import Path

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


# ── Minimal seed stopwords ────────────────────────────────────────────────────
# These are the ONLY manually defined stopwords.
# They're shopping/platform words that won't be caught by frequency statistics
# because they're not universally high-frequency (maybe 5-10% of reviews)
# but carry zero topic signal regardless.

SEED_STOP = {
    "amazon", "product", "products", "item", "items",
    "purchased", "purchase", "ordered", "order", "ordering",
    "delivery", "delivered", "shipping", "shipped", "seller",
    "bought", "buy", "buying", "store", "shop",
    "review", "reviews", "star", "stars", "rating",
}


def _tokenize_simple(text: str) -> list[str]:
    """Fast tokenizer — no stopword filtering here, we want all words."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [w for w in text.split() if len(w) > 2]


def induce_stopwords(
    df: pd.DataFrame,
    df_threshold: float = 0.15,
    discriminative_threshold: float = 0.005,
    min_word_freq: int = 200,
) -> tuple[set[str], pd.DataFrame]:
    """
    Compute corpus-based stopwords from a labelled review DataFrame.

    Parameters
    ----------
    df                        : DataFrame with 'Score' and 'Text' columns
    df_threshold              : words in > X% of docs are stopwords (default 15%)
    discriminative_threshold  : words with pos/neg freq diff < X are stopwords (default 2%)
    min_word_freq             : ignore words appearing fewer than N times total
                                (avoids flagging rare words as non-discriminative
                                 just because they happened to appear in few reviews)

    Returns
    -------
    stopwords_set : set of induced stopword strings
    stats_df      : full word statistics DataFrame (useful for inspection/tuning)
    """
    # Binary sentiment: positive (Score >= 4) vs negative (Score <= 2)
    # Exclude Score=3 from induction — neutral reviews blur the signal
    pos_mask = df["Score"] >= 4
    neg_mask = df["Score"] <= 2

    pos_texts = df[pos_mask]["Text"].fillna("").tolist()
    neg_texts = df[neg_mask]["Text"].fillna("").tolist()
    all_texts = df["Text"].fillna("").tolist()

    n_pos = len(pos_texts)
    n_neg = len(neg_texts)
    n_all = len(all_texts)

    print(f"[induce] Positive reviews: {n_pos:,}")
    print(f"[induce] Negative reviews: {n_neg:,}")
    print(f"[induce] Total reviews:    {n_all:,}")
    print("[induce] Building word statistics… (this takes 2-3 minutes on 500K reviews)")

    # Count documents containing each word (not occurrences — documents)
    # Using sets per review for document frequency
    def _doc_freq_counter(texts: list[str]) -> dict[str, int]:
        counter: dict[str, int] = {}
        for text in texts:
            # Use set so each word counts once per review
            for word in set(_tokenize_simple(text)):
                counter[word] = counter.get(word, 0) + 1
        return counter

    all_counts  = _doc_freq_counter(all_texts)
    pos_counts  = _doc_freq_counter(pos_texts)
    neg_counts  = _doc_freq_counter(neg_texts)

    # Filter to words with enough total occurrences
    vocab = {w for w, c in all_counts.items() if c >= min_word_freq}
    print(f"[induce] Vocabulary size (freq >= {min_word_freq}): {len(vocab):,} words")

    # Compute statistics
    rows = []
    for word in vocab:
        total_docs = all_counts.get(word, 0)
        doc_freq   = total_docs / n_all

        pos_freq = pos_counts.get(word, 0) / n_pos
        neg_freq = neg_counts.get(word, 0) / n_neg
        disc     = abs(pos_freq - neg_freq)

        rows.append({
            "word":             word,
            "doc_freq":         round(doc_freq, 4),
            "pos_freq":         round(pos_freq, 4),
            "neg_freq":         round(neg_freq, 4),
            "discriminative":   round(disc, 4),
        })

    stats = pd.DataFrame(rows).sort_values("doc_freq", ascending=False)

    # Apply criteria
    is_high_df    = stats["doc_freq"]       > df_threshold
    is_low_disc   = stats["discriminative"] < discriminative_threshold
    is_base_stop  = stats["word"].isin(set(stopwords.words("english")))

    stats["is_stopword"] = is_high_df | is_low_disc | is_base_stop

    induced = set(stats[stats["is_stopword"]]["word"].tolist())
    induced |= SEED_STOP  # always include seed words

    print(f"\n[induce] Stopwords induced:     {len(induced):,}")
    print(f"[induce]   from high DF (>{df_threshold:.0%}):   "
          f"{int(is_high_df.sum()):,}")
    print(f"[induce]   from low disc (<{discriminative_threshold:.0%}): "
          f"{int(is_low_disc.sum()):,}")
    print(f"[induce]   from NLTK base:         "
          f"{int(is_base_stop.sum()):,}")

    # Show top 20 most frequent non-stopwords (sanity check)
    kept = stats[~stats["is_stopword"]].head(20)
    print("\n[induce] Top 20 kept words (should be topic-relevant):")
    print(kept[["word", "doc_freq", "discriminative"]].to_string(index=False))

    # Show 10 newly induced domain stopwords (not in NLTK base)
    new_domain = stats[
        stats["is_stopword"] &
        ~is_base_stop &
        ~stats["word"].isin(SEED_STOP)
    ].head(15)
    print("\n[induce] Top 15 newly induced domain stopwords:")
    print(new_domain[["word", "doc_freq", "discriminative"]].to_string(index=False))

    return induced, stats


def main():
    parser = argparse.ArgumentParser(
        description="Induce domain stopwords from Amazon Fine Food Reviews."
    )
    parser.add_argument(
        "--input_csv",
        default="data/raw/reviews_analysis/Reviews.csv",
    )
    parser.add_argument(
        "--output_txt",
        default="data/raw/reviews_analysis/induced_stopwords.txt",
        help="One stopword per line",
    )
    parser.add_argument(
        "--output_stats",
        default="data/raw/reviews_analysis/stopword_stats.csv",
        help="Full word statistics CSV for inspection",
    )
    parser.add_argument(
        "--df_threshold", type=float, default=0.15,
        help="Document frequency threshold (default 0.15 = 15%%)",
    )
    parser.add_argument(
        "--disc_threshold", type=float, default=0.005,
        help="Discriminative power threshold (default 0.02 = 2%%)",
    )
    args = parser.parse_args()

    print(f"[induce] Loading {args.input_csv}…")
    df = pd.read_csv(args.input_csv, engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["Text", "Score"])
    print(f"[induce] Loaded {len(df):,} reviews\n")

    stopwords_set, stats = induce_stopwords(
        df,
        df_threshold=args.df_threshold,
        discriminative_threshold=args.disc_threshold,
    )

    # Save stopwords
    out_path = Path(args.output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(stopwords_set)), encoding="utf-8")
    print(f"\n[induce] Saved stopwords → {out_path}")

    # Save stats
    stats_path = Path(args.output_stats)
    stats.to_csv(stats_path, index=False)
    print(f"[induce] Saved stats     → {stats_path}")
    print("\nDone. Now re-run output_generator.py to use the induced stopwords.")


if __name__ == "__main__":
    main()
