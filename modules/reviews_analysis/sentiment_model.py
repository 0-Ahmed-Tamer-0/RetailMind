"""sentiment_model.py — Review sentiment analysis
Dataset: Amazon Fine Food Reviews (Kaggle/snap-stanford)
Columns used: Score, Text, Summary, Time

Sentiment signals
─────────────────
Three independent signals, majority vote decides the label:
  1. Score-based  : star rating converted to polarity
                    4-5 = positive, 1-2 = negative, 3 = neutral
  2. VADER        : rule-based NLP on review text
                    compound >= 0.05 = positive, <= -0.05 = negative
  3. TextBlob     : pattern-based sentiment on review text
                    polarity > 0.1 = positive, < -0.1 = negative

Majority vote (2 out of 3 must agree):
  Stronger claim than single-signal — two independent models
  agreeing on "negative" is much more reliable than one alone.
  When all three disagree → neutral.

Keyword extraction
──────────────────
  Problem with generic food keywords ("coffee", "tea", "taste"):
  these appear in EVERY review regardless of sentiment.
  Solution: TF-IDF-style filtering — keep only words that appear
  significantly MORE in negative reviews than positive ones and
  vice versa. This surfaces opinion words, not category words.
"""

from __future__ import annotations

import re
import string
from collections import Counter

import nltk
import pandas as pd
import numpy as np

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger", quiet=True)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

_SIA = SentimentIntensityAnalyzer()

# ── Stop words ────────────────────────────────────────────────────────────────

_BASE_STOP = set(stopwords.words("english"))

# ── Minimal seed stopwords (hardcoded) ───────────────────────────────────────
# Platform/shopping words corpus statistics won't reliably catch
# because they sit at ~5-10% document frequency (below DF threshold)
# but carry zero topic signal regardless.
# Everything else is handled by the induced stopwords file.
_SEED_STOP = {
    "amazon", "product", "products", "item", "items",
    "purchased", "purchase", "ordered", "order",
    "delivery", "delivered", "shipping", "shipped", "seller",
}


def _load_induced_stopwords(
    path: str = "data/raw/reviews_analysis/induced_stopwords.txt",
) -> set[str]:
    """
    Load corpus-induced stopwords computed offline by induce_stopwords.py.

    Two induction criteria used when building the file:
      1. Document frequency > 15%  — too common to discriminate topics
      2. Discriminative power < 2% — appears equally in positive and negative reviews

    Falls back gracefully if file not found so the module still runs
    before induction has been executed (just with lower LDA quality).
    """
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.exists():
        alt = _Path(__file__).resolve().parents[2] / path
        if alt.exists():
            p = alt

    if not p.exists():
        print(
            f"[sentiment] WARNING: induced_stopwords.txt not found.\n"
            f"  Run: python modules/reviews_analysis/induce_stopwords.py\n"
            f"  Falling back to seed stopwords only."
        )
        return set()

    words = {
        w.strip().lower()
        for w in p.read_text(encoding="utf-8").splitlines()
        if w.strip()
    }
    print(f"[sentiment] Loaded {len(words):,} induced stopwords from {p.name}")
    return words


_DOMAIN_STOP = _SEED_STOP | _load_induced_stopwords()
_STOP = _BASE_STOP | _DOMAIN_STOP


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_amazon(df: pd.DataFrame, sample_n: int = 50_000) -> pd.DataFrame:
    """
    Clean raw Amazon Fine Food Reviews CSV.

    1. Drop nulls in Summary/Text (~27 rows)
    2. Parse Unix timestamp → datetime
    3. Stratified sample by Score (preserves class distribution)
    4. Compute helpfulness ratio
    5. Combine Summary + '. ' + Text into review_text
    """
    df = df.dropna(subset=["Summary", "Text"]).copy()
    df["date"] = pd.to_datetime(df["Time"], unit="s")

    if len(df) > sample_n:
        df = (
            df.groupby("Score", group_keys=False)
            .apply(lambda g: g.sample(
                n=min(len(g), int(sample_n * len(g) / len(df))),
                random_state=42,
            ))
            .reset_index(drop=True)
        )

    df["helpfulness"] = df.apply(
        lambda r: r["HelpfulnessNumerator"] / r["HelpfulnessDenominator"]
        if r["HelpfulnessDenominator"] > 0 else 0.0,
        axis=1,
    )

    # Combine: Summary is often the most opinionated sentence
    df["review_text"] = df["Summary"].fillna("") + ". " + df["Text"].fillna("")
    return df.reset_index(drop=True)


# ── Sentiment labelling ───────────────────────────────────────────────────────

def _score_signal(score: int) -> str:
    if score >= 4: return "positive"
    if score <= 2: return "negative"
    return "neutral"


def _vader_signal(text: str) -> str:
    c = _SIA.polarity_scores(str(text))["compound"]
    if c >= 0.05:  return "positive"
    if c <= -0.05: return "negative"
    return "neutral"


def _textblob_signal(text: str) -> str:
    p = TextBlob(str(text)).sentiment.polarity
    if p > 0.1:  return "positive"
    if p < -0.1: return "negative"
    return "neutral"


def _majority(signals: list[str]) -> str:
    """
    Majority vote across 3 signals.
    If no majority (all different) → neutral.
    """
    c = Counter(signals)
    top, count = c.most_common(1)[0]
    return top if count >= 2 else "neutral"


def label_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns: vader_compound, signal_score, signal_vader,
                 signal_textblob, sentiment (majority vote)

    Keeping individual signals in the output lets the chatbot say
    "2 out of 3 models agree this is negative" — explainable AI.
    """
    df = df.copy()

    print("[sentiment] Running VADER…")
    df["vader_compound"] = df["review_text"].apply(
        lambda t: _SIA.polarity_scores(str(t))["compound"]
    )

    print("[sentiment] Running TextBlob…")
    df["textblob_polarity"] = df["review_text"].apply(
        lambda t: TextBlob(str(t)).sentiment.polarity
    )

    df["signal_score"]    = df["Score"].apply(_score_signal)
    df["signal_vader"]    = df["vader_compound"].apply(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    )
    df["signal_textblob"] = df["textblob_polarity"].apply(
        lambda p: "positive" if p > 0.1 else ("negative" if p < -0.1 else "neutral")
    )

    df["sentiment"] = df.apply(
        lambda r: _majority([r["signal_score"], r["signal_vader"], r["signal_textblob"]]),
        axis=1,
    )
    return df


# ── Keyword extraction ────────────────────────────────────────────────────────

def _tokenize(text: str, pos_filter: bool = False) -> list[str]:
    """
    Tokenize review text with two optional filtering gates.

    Gate 1 — Stopword removal (always on):
        Removes induced corpus stopwords + NLTK base stopwords.
        Handles contraction artifacts (dont, ive, wasnt).

    Gate 2 — POS filtering (on when pos_filter=True):
        Keeps only nouns (NN, NNS, NNP, NNPS) and
        adjectives (JJ, JJR, JJS).

        Why only nouns and adjectives?
        ───────────────────────────────
        Topics are defined by THINGS (nouns) and their PROPERTIES
        (adjectives). Verbs, adverbs, determiners, and prepositions
        are grammatically incapable of naming a subject — they describe
        actions and relationships, not entities.

        "the product tasted absolutely horrible and smelled stale"
          → after POS filter: ["product", "horrible", "stale"]
          → LDA input is clean subject-matter vocabulary

        Verbs like "tasted/smelled" and adverbs like "absolutely"
        are dropped — they carry sentiment but not topic identity.

        POS tagging uses NLTK's averaged_perceptron_tagger which runs
        fully offline after the one-time ~2MB model download.

    pos_filter=False for keyword extraction (we want all word types there).
    pos_filter=True  for LDA input (nouns + adjectives only).
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s\'']", " ", text)
    text = re.sub(r"\'\w+", "", text)
    tokens = [w for w in text.split() if w not in _STOP and len(w) > 2]

    if not pos_filter:
        return tokens

    # POS filter — keep only nouns and adjectives
    _KEEP_POS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}
    tagged = nltk.pos_tag(tokens)
    return [word for word, tag in tagged if tag in _KEEP_POS]

def extract_keywords(
    df: pd.DataFrame,
    sentiment_label: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract keywords that are DISTINCTIVE for this sentiment label.

    The key improvement over simple frequency counting:
    We compute a discriminative score:
        score = freq_in_target / (freq_in_all + 1)

    A word scoring high appears often in the target sentiment but
    rarely overall — meaning it's actually associated with that
    sentiment, not just common in food reviews generally.

    This is equivalent to a simplified TF-IDF where:
        TF  = frequency in target sentiment class
        IDF = inverse of frequency across all sentiments

    Helpfulness weighting still applied on top.
    """
    target = df[df["sentiment"] == sentiment_label].copy()
    if target.empty:
        return pd.DataFrame(columns=["keyword", "score", "sentiment"])

    # Count in target (weighted by helpfulness)
    target_counter: dict[str, float] = {}
    for _, row in target.iterrows():
        tokens = _tokenize(str(row["review_text"]), pos_filter=False)
        weight = 1.0 + float(row.get("helpfulness", 0.0))
        for t in tokens:
            target_counter[t] = target_counter.get(t, 0) + weight
        for i in range(len(tokens) - 1):
            bg = tokens[i] + " " + tokens[i + 1]
            target_counter[bg] = target_counter.get(bg, 0) + weight

    # Count across ALL reviews (unweighted — just existence check)
    all_counter: dict[str, float] = {}
    for _, row in df.iterrows():
        tokens = set(_tokenize(str(row["review_text"])))
        for t in tokens:
            all_counter[t] = all_counter.get(t, 0) + 1
        tokens_list = list(tokens)
        for i in range(len(tokens_list) - 1):
            bg = tokens_list[i] + " " + tokens_list[i + 1]
            all_counter[bg] = all_counter.get(bg, 0) + 1

    # Discriminative score
    n_target = max(len(target), 1)
    n_all    = max(len(df), 1)

    scored = {}
    for kw, freq in target_counter.items():
        global_freq = all_counter.get(kw, 0)
        # Normalize by class size, penalize by global frequency
        scored[kw] = (freq / n_target) / ((global_freq / n_all) + 0.01)

    top = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result = pd.DataFrame(top, columns=["keyword", "score"])
    result["sentiment"] = sentiment_label
    result["score"] = result["score"].round(3)
    return result


# ── Trend aggregation ─────────────────────────────────────────────────────────

def compute_trends(df: pd.DataFrame, resample: str = "ME") -> pd.DataFrame:
    """
    Monthly sentiment trends over time.
    Uses 'ME' (month-end) instead of deprecated 'M'.
    """
    df = df.copy().set_index("date").sort_index()

    agg = df.resample(resample).agg(
        avg_rating=("Score", "mean"),
        avg_vader=("vader_compound", "mean"),
        avg_textblob=("textblob_polarity", "mean"),
        review_count=("sentiment", "count"),
        positive_count=("sentiment", lambda x: (x == "positive").sum()),
        negative_count=("sentiment", lambda x: (x == "negative").sum()),
        neutral_count=("sentiment",  lambda x: (x == "neutral").sum()),
    ).reset_index()

    agg["positive_ratio"] = (agg["positive_count"] / agg["review_count"]).round(3)
    agg["negative_ratio"] = (agg["negative_count"] / agg["review_count"]).round(3)
    agg["neutral_ratio"]  = (agg["neutral_count"]  / agg["review_count"]).round(3)
    agg["avg_rating"]     = agg["avg_rating"].round(2)
    agg["avg_vader"]      = agg["avg_vader"].round(3)
    agg["avg_textblob"]   = agg["avg_textblob"].round(3)

    return agg.drop(columns=["positive_count", "negative_count", "neutral_count"])


# ── LDA Topic Modeling ────────────────────────────────────────────────────────

def extract_topics(
    df: pd.DataFrame,
    sentiment_label: str,
    n_topics: int = 5,
    n_top_words: int = 8,
    n_passes: int = 10,
) -> pd.DataFrame:
    """
    Discover latent topics within a sentiment group using LDA.

    Why LDA on top of keywords?
    ───────────────────────────
    Keywords answer "what words appear most?" — useful for word clouds.
    LDA answers "what are customers actually talking about?" — useful
    for business decisions.

    LDA treats each review as a mixture of topics, and each topic as
    a probability distribution over words. After training, each topic
    is represented by its highest-probability words. We then assign a
    human-readable theme name based on those words.

    Parameters
    ----------
    df              : labelled DataFrame (output of label_sentiment)
    sentiment_label : 'positive', 'negative', or 'neutral'
    n_topics        : number of topics to discover (5 is good for 50K reviews)
    n_top_words     : words per topic to surface
    n_passes        : LDA training passes (more = better, slower)

    Returns
    -------
    DataFrame with columns:
        sentiment, topic_id, top_words, review_count, theme
    """
    try:
        from gensim import corpora
        from gensim.models import LdaModel
    except ImportError:
        raise ImportError("Run: pip install gensim")

    subset = df[df["sentiment"] == sentiment_label].copy()
    if len(subset) < 50:
        return pd.DataFrame(columns=["sentiment", "topic_id", "top_words", "review_count", "theme"])

    print(f"[LDA] Training on {len(subset):,} {sentiment_label} reviews…")

    # Tokenize using same stopwords as keyword extraction
    tokenized = [_tokenize(str(t), pos_filter=True) for t in subset["review_text"]]
    tokenized = [t for t in tokenized if len(t) >= 3]  # drop very short reviews

    # Build dictionary + corpus
    dictionary = corpora.Dictionary(tokenized)
    # Filter extremes: ignore words in <5 docs or >50% of docs
    # no_below=10: word must appear in at least 10 reviews (filters typos/rare words)
    # no_above=0.3: word must not appear in more than 30% of reviews (filters generic words)
    # Together these force LDA to focus on the informative middle ground
    dictionary.filter_extremes(no_below=10, no_above=0.3)
    corpus = [dictionary.doc2bow(t) for t in tokenized]

    # Train LDA
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        passes=n_passes,
        alpha="auto",          # learns asymmetric prior — better topic separation
        per_word_topics=False,
    )

    # Extract top words per topic
    rows = []
    for topic_id in range(n_topics):
        top_words_raw = lda.show_topic(topic_id, topn=n_top_words)
        top_words = [w for w, _ in top_words_raw]
        top_words_str = ", ".join(top_words)

        # Auto-name the topic based on its top words
        theme = _name_topic(top_words, sentiment_label)

        # Count reviews where this topic dominates
        topic_counts = [
            max(lda.get_document_topics(doc), key=lambda x: x[1], default=(topic_id, 0))[0]
            for doc in corpus
        ]
        review_count = sum(1 for t in topic_counts if t == topic_id)

        rows.append({
            "sentiment":    sentiment_label,
            "topic_id":     topic_id,
            "top_words":    top_words_str,
            "review_count": review_count,
            "theme":        theme,
        })

    result = pd.DataFrame(rows).sort_values("review_count", ascending=False)
    return result


# ── Candidate theme vocabulary ────────────────────────────────────────────────
# Changing this list requires zero code changes — just edit the strings.
# The embedding model handles the semantic matching automatically.
# Separated by sentiment so matching is always within the right domain.

NEGATIVE_THEMES = [
    "Freshness & Expiry Issues",
    "Damaged Packaging on Arrival",
    "Product Not as Described",
    "Poor Value for Money",
    "Food Safety & Health Concerns",
    "Country of Origin Concerns",
    "Artificial Ingredients Concern",
    "Quantity or Size Disappointment",
    "Pet Food Quality Issues",
    "Customer Service Failures",
]

POSITIVE_THEMES = [
    "Excellent Taste & Flavor",
    "Health & Dietary Benefits",
    "Great Value for Money",
    "Would Buy Again",
    "Pet Food Satisfaction",
    "Fast & Reliable Delivery",
    "High Product Quality",
    "Coffee & Hot Drinks",
    "Natural & Specialty Ingredients",
    "Convenient & Easy to Use",
]

# ── Lazy model cache ──────────────────────────────────────────────────────────
# We load SentenceTransformer once and reuse it across all topic naming calls.
# Lazy loading means importing sentiment_model.py doesn't trigger a 400MB
# model download — only the first extract_topics() call does.

_embedder = None
_theme_embeddings: dict[str, np.ndarray] = {}   # cache: theme_str → embedding


def _get_embedder():
    """
    Load sentence-transformers MiniLM model on first call, then cache it.

    Why all-MiniLM-L6-v2?
    ─────────────────────
    - 384-dimensional embeddings: small enough to be fast
    - Trained on 1B+ sentence pairs: strong semantic understanding
    - Same model used in the RAG pipeline: one download, two uses
    - Runs fully offline after first download (~90MB)
    """
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")
        print("[topic_namer] Loading MiniLM embedding model…")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_theme_embeddings(themes: list[str]) -> np.ndarray:
    """
    Embed all candidate theme strings, using cache to avoid re-embedding
    the same strings on repeated calls (e.g. across 5 topics of same sentiment).

    Returns shape: (n_themes, 384)
    """
    embedder = _get_embedder()
    cache_key = "|".join(themes)

    if cache_key not in _theme_embeddings:
        _theme_embeddings[cache_key] = embedder.encode(
            themes,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit vectors → cosine sim = dot product
        )
    return _theme_embeddings[cache_key]


def _name_topic(top_words: list[str], sentiment: str) -> str:
    """
    Name a topic by finding the semantically closest candidate theme.

    How it works
    ────────────
    1. Join the LDA top_words into a phrase:
       ["stale", "expired", "smell", "rotten"] → "stale expired smell rotten"

    2. Embed that phrase with MiniLM → 384-dim vector

    3. Embed all candidate theme names for this sentiment

    4. Cosine similarity between topic vector and each theme vector.
       Since both are L2-normalised, cosine_sim = dot product.

    5. Return the theme with highest similarity score.

    Why this is better than if/else
    ────────────────────────────────
    The model understands that "rancid" is semantically close to
    "Freshness & Expiry Issues" even though "rancid" isn't in any
    hard-coded word list. It generalises from meaning, not string matching.
    New topics the vocabulary never anticipated still get reasonable names
    as long as the candidate list has a semantically nearby option.

    Comparison with LLM naming (Day 3)
    ────────────────────────────────────
    This approach picks from a fixed vocabulary.
    LLM naming generates free-form names — more creative, potentially
    more precise, but slower and requires Ollama to be running.
    We keep both and compare outputs in the discussion.
    """
    candidates = NEGATIVE_THEMES if sentiment == "negative" else POSITIVE_THEMES

    # Build topic phrase from LDA top words
    topic_phrase = " ".join(top_words)

    # Embed topic phrase
    embedder = _get_embedder()
    topic_vec = embedder.encode(
        topic_phrase,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )                                          # shape: (384,)

    # Embed candidate themes (cached)
    theme_vecs = _get_theme_embeddings(candidates)  # shape: (n_themes, 384)

    # Cosine similarity = dot product (both L2-normalized)
    scores = theme_vecs @ topic_vec            # shape: (n_themes,)

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_theme = candidates[best_idx]

    # Log for transparency — useful during development
    print(f"    topic '{topic_phrase[:40]}…' "
          f"→ '{best_theme}' (similarity: {best_score:.3f})")

    return best_theme
