"""rag/pipeline.py — Main entry point for the RAG + fallback system

What this file does
────────────────────
Orchestrates the full RAG pipeline and implements the adaptive
inference architecture with graceful degradation.

Architecture
─────────────
                    User question
                          │
                    ┌─────▼──────┐
                    │  ollama_   │
                    │ available()│
                    └─────┬──────┘
                   YES    │    NO
          ┌───────────────┤    ├───────────────┐
          ▼               │    │               ▼
   ┌─────────────┐        │    │    ┌─────────────────────┐
   │  retrieve() │        │    │    │  keyword_response() │
   │ FAISS search│        │    │    │  (existing matcher) │
   └──────┬──────┘        │    │    └─────────────────────┘
          │               │    │
   ┌──────▼──────┐        │    │
   │ call_ollama │        │    │
   │ Llama 3.1  │        │    │
   └──────┬──────┘        │    │
          │               │    │
   ┌──────▼──────┐        │    │
   │  format +   │        │    │
   │  citations  │        │    │
   └─────────────┘        │    │

Why keep the keyword matcher?
──────────────────────────────
"Adaptive inference with graceful degradation" — the system remains
fully functional on hardware without enough RAM for Llama 3.1 (8GB+).
Users on constrained hardware get rule-based responses automatically.
Users with capable hardware get LLM-powered responses automatically.
No configuration needed — detection is runtime, transparent to the user.

Query routing
──────────────
Before calling retrieve(), we detect the intent of the query and
set source_filter accordingly. This improves retrieval precision:
  inventory/reorder queries → filter to inventory source
  complaint/review queries  → filter to review_topics
  segment queries           → filter to segments
  general queries           → no filter (cross-source retrieval)
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.rag.retriever import retrieve, retrieve_by_source
from modules.rag.prompter  import call_ollama, ollama_available


# ── Intent → source mapping ───────────────────────────────────────────────────
# Maps detected intent keywords to FAISS source filters.
# This is not the same as the old keyword matcher — it's routing,
# not answering. The LLM still generates the actual answer.

_INTENT_SOURCES: list[tuple[set[str], list[str]]] = [
    (
        {"reorder", "stock", "inventory", "order", "supply", "replenish",
         "out of stock", "shortage", "overstock", "discount", "markdown"},
        ["inventory"],
    ),
    (
        # Satisfaction queries must check reviews, not segments
        # "satisfied/satisfaction/happy" are about sentiment, not behavior
        {"complaint", "complain", "dislike", "negative", "problem", "issue",
         "bad review", "customer complaint", "unhappy", "worst", "terrible",
         "satisfied", "satisfaction", "happy with", "feel about",
         "think about", "opinion", "feedback", "what do customers",
         "customers say", "customers think", "customers feel"},
        ["review_topics", "review_keywords", "review_insights"],
    ),
    (
        {"praise", "like", "positive", "love", "best review",
         "what customers love", "top review", "customers like",
         "customers enjoy", "popular"},
        ["review_topics", "review_keywords"],
    ),
    (
        {"review trend", "rating trend", "sentiment over time",
         "rating history", "review history", "monthly rating",
         "rating over", "reviews over"},
        ["review_insights"],
    ),
    (
        {"segment", "customer type", "loyal", "champion", "at risk",
         "customer group", "rfm", "who are my customers",
         "customer base", "types of customers"},
        ["segments"],
    ),
    (
        {"forecast", "predict", "demand", "sales forecast", "next week",
         "next month", "how much will", "projected", "expected sales"},
        ["sales_forecast"],
    ),
    (
        {"zone", "foot traffic", "traffic", "busy", "busiest",
         "people count", "store traffic", "which zone", "most people"},
        ["cv_foot_traffic"],
    ),
]


def _detect_sources(query: str) -> list[str] | None:
    """
    Return source filter list if query intent is clear, else None.
    None = search all sources (cross-source retrieval for general queries).
    """
    q = query.lower()
    for keywords, sources in _INTENT_SOURCES:
        if any(kw in q for kw in keywords):
            return sources
    return None


def _format_citations(chunks: list[dict]) -> str:
    """
    Build a compact citation footer showing which data sources were used.
    Displayed below the LLM answer in the Streamlit chat UI.
    """
    if not chunks:
        return ""
    sources_seen = []
    for c in chunks:
        label = c.get("source", "").replace("_", " ").title()
        if label not in sources_seen:
            sources_seen.append(label)
    return "📊 *Sources: " + " · ".join(sources_seen) + "*"


# ── Main entry point ──────────────────────────────────────────────────────────

def rag_response(
    question:   str,
    top_k:      int  = 5,
    stream:     bool = False,
) -> dict:
    """
    Full RAG pipeline: retrieve → prompt → respond.

    Returns
    -------
    dict with keys:
        text       — LLM answer string (or generator if stream=True)
        citations  — formatted source attribution string
        chunks     — raw retrieved chunks (for debug/UI display)
        mode       — "rag" always (pipeline.py only called when Ollama available)
    """
    source_filter = _detect_sources(question)

    # top_k=3: enough context for factual retail answers, reduces prompt size
    # smaller prompt = faster CPU inference (token processing is the bottleneck)
    # Special case 1: satisfaction/review queries need BOTH pos+neg topic chunks
    # Semantic search might only return the neg chunk (closest to "satisfied?")
    # Force retrieval of both so LLM sees complete picture
    satisfaction_words = {
        "satisfied", "satisfaction", "happy with", "feel about",
        "customers think", "customers say", "overall review",
        "are customers", "good reviews", "bad reviews",
    }
    q_lower_rag = question.lower()

    if any(w in q_lower_rag for w in satisfaction_words):
        from modules.rag.retriever import retrieve_by_source
        topic_chunks = retrieve_by_source("review_topics", top_n=10)
        insight_chunks = retrieve(question, top_k=2,
                                  source_filter=["review_insights"])
        chunks = topic_chunks + insight_chunks
    # Special case 2: top-N forecast → dump all sales chunks (not semantic)
    elif any(w in q_lower_rag for w in
             {"top 10", "top10", "top products", "all products",
              "best products", "highest forecast", "most demand"}):
        from modules.rag.retriever import retrieve_by_source
        chunks = retrieve_by_source("sales_forecast", top_n=20)
    else:
        chunks = retrieve(question, top_k=3, source_filter=source_filter)
        if not chunks and source_filter:
            chunks = retrieve(question, top_k=3, source_filter=None)

    answer = call_ollama(question, chunks, stream=stream)

    return {
        "text":      answer,
        "citations": _format_citations(chunks),
        "chunks":    chunks,
        "mode":      "rag",
    }


def _build_heatmap(data: dict, query: str = "") -> dict | None:
    """
    Build zone heatmap response.

    Priority:
    1. Overlay images (annotated camera frames with bounding boxes)
       — stored in data/outputs/cv_zone_overlays/zone_*_overlay.png
       — generated by the CV module's output_generator.py
       — shows WHERE people stand in the store (spatial)
    2. Plotly bar chart fallback
       — when overlay images don't exist
       — shows WHICH zones have most traffic (comparative)

    Why images first?
    ──────────────────
    Overlay images show the actual camera view with detected people
    highlighted — a retail owner can see exactly which store areas
    are crowded. A bar chart only shows zone counts without spatial context.
    """
    from pathlib import Path as _Path
    import pandas as pd

    ROOT  = _Path(__file__).resolve().parents[2]
    OVERLAYS = ROOT / "data" / "outputs" / "cv_zone_overlays"

    # ── Option 1: real overlay images ──
    if OVERLAYS.exists():
        all_pngs = sorted(OVERLAYS.glob("zone_*_overlay.png"))
        if all_pngs:
            # Try to extract a specific zone ID from the query
            # e.g. "show zone 41C heatmap" → filter to zone_41C_overlay.png
            import re as _re
            zone_match = _re.search(
                r"zone[_\s]?([0-9]+[a-zA-Z]?)", query.lower()
            )
            if zone_match:
                zone_token = zone_match.group(1).upper().replace(" ", "_")
                filtered = [p for p in all_pngs if zone_token in p.name.upper()]
                pngs = filtered if filtered else all_pngs
            else:
                pngs = all_pngs   # no zone specified → show all

            cv = data.get("cv")
            if isinstance(cv, pd.DataFrame) and not cv.empty:
                zone_stats = (
                    cv.groupby("zone_id")["people_count"]
                    .agg(avg_people="mean", peak_people="max")
                    .reset_index()
                    .sort_values("avg_people", ascending=False)
                )
                busiest = zone_stats.iloc[0]
                if zone_match:
                    text = f"Overlay for zone **{zone_token}** from CV module."
                else:
                    text = (
                        f"Zone overlay heatmaps — **{busiest['zone_id']}** "
                        f"is the busiest (avg {busiest['avg_people']:.1f} people, "
                        f"peak {busiest['peak_people']:.0f})."
                    )
            else:
                text = "Zone overlay heatmaps from CV module:"

            return {
                "text":         text,
                "fig":          None,
                "table":        None,
                "image_path":   str(pngs[0]) if pngs else None,
                "extra_images": [str(p) for p in pngs[1:]] if len(pngs) > 1 else [],
                "citations":    "📊 *Sources: Cv Foot Traffic*",
                "mode":         "rag",
                "chunks":       [],
            }

    # ── Option 2: bar chart fallback ──
    try:
        import plotly.express as px
        cv = data.get("cv")
        if not isinstance(cv, pd.DataFrame) or cv.empty:
            return None

        zone_stats = (
            cv.groupby("zone_id")["people_count"]
            .agg(avg_people="mean", peak_people="max", observations="count")
            .reset_index()
            .sort_values("avg_people", ascending=False)
        )
        fig = px.bar(
            zone_stats,
            x="zone_id", y="avg_people",
            color="avg_people",
            color_continuous_scale="Reds",
            title="Average Foot Traffic by Zone",
            labels={"zone_id": "Zone", "avg_people": "Avg People per Frame"},
            template="plotly_dark", text="avg_people",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            margin=dict(t=50, b=20, l=0, r=0),
        )
        busiest = zone_stats.iloc[0]
        text = (
            f"Zone **{busiest['zone_id']}** is the busiest with an average of "
            f"**{busiest['avg_people']:.1f} people** per frame "
            f"(peak: {busiest['peak_people']:.0f} people)."
            f"*Note: overlay images not found in cv_zone_overlays/ — "
            f"run the CV module to generate annotated zone images.*"
        )
        return {
            "text": text, "fig": fig, "table": zone_stats,
            "image_path": None, "extra_images": None,
            "citations": "📊 *Sources: Cv Foot Traffic*",
            "mode": "rag", "chunks": [],
        }
    except Exception as e:
        print(f"[pipeline] heatmap error: {e}")
        return None


_HEATMAP_KEYWORDS = {
    "heatmap", "heat map", "zone heatmap",
    "show heatmap", "show heat", "overlay",
    "zone overlay", "zone map", "traffic map",
}
# Note: "busiest zone/zones", "foot traffic", "which zone" go through RAG
# as text answers — only explicit heatmap/visual requests trigger the chart


# ── Plot keywords ────────────────────────────────────────────────────────────
_PLOT_KEYWORDS = {
    "plot", "chart", "graph", "show sales", "sales trend",
    "show forecast", "forecast chart", "visualize", "draw",
}

_RATING_KEYWORDS = {
    "rating trend", "rating chart", "plot rating", "show rating",
    "sentiment trend", "review trend", "rating over time",
    "plot reviews", "chart rating",
}


def _extract_product_id(question: str, data: dict) -> str | None:
    """
    Extract product ID from a chat query like 'plot product 85123A'.
    Tries to match any token in the question against known product IDs.
    """
    import pandas as pd
    sales = data.get("sales")
    if not isinstance(sales, pd.DataFrame):
        return None
    known = set(sales["product_id"].astype(str).unique())
    tokens = question.upper().replace(",", " ").split()
    for token in tokens:
        if token in known:
            return token
    return None


def _build_product_chart(product_id: str, data: dict) -> dict | None:
    """
    Build actual vs predicted sales chart for a specific product.
    Shows:
      - Actual sales (solid line)
      - Predicted sales (dashed line)
      - Confidence interval (shaded band)
    This is a forecast validation chart — shows how well Prophet
    tracks real sales, not just the forecast alone.
    """
    try:
        import plotly.graph_objects as go
        import pandas as pd

        sales = data.get("sales")
        if not isinstance(sales, pd.DataFrame):
            return None

        df = sales[sales["product_id"].astype(str) == str(product_id)].copy()
        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        fig = go.Figure()

        # Confidence interval band
        if "predicted_low" in df.columns and "predicted_high" in df.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([df["date"], df["date"][::-1]]),
                y=pd.concat([df["predicted_high"], df["predicted_low"][::-1]]),
                fill="toself",
                fillcolor="rgba(99,110,250,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence interval",
                hoverinfo="skip",
            ))

        # Predicted sales
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["predicted_sales"],
            mode="lines",
            name="Predicted",
            line=dict(color="#636EFA", dash="dash", width=2),
        ))

        # Actual sales
        if "actual_sales" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["actual_sales"],
                mode="lines",
                name="Actual",
                line=dict(color="#EF553B", width=2),
            ))

        fig.update_layout(
            title=f"Product {product_id} — Actual vs Forecast Sales",
            xaxis_title="Date",
            yaxis_title="Sales (£)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=60, b=20, l=0, r=0),
            hovermode="x unified",
        )

        total_pred = df["predicted_sales"].sum()
        text = (
            f"Sales forecast chart for product **{product_id}**. "
            f"Total predicted sales: **£{total_pred:,.0f}** over {len(df)} days. "
            f"Blue dashed line = Prophet forecast · Red line = actual sales · "
            f"Shaded band = 80% confidence interval."
        )
        return {
            "text": text, "fig": fig, "table": None,
            "image_path": None, "extra_images": None,
            "citations": "📊 *Sources: Sales Forecast*",
            "mode": "rag", "chunks": [],
        }
    except Exception as e:
        print(f"[pipeline] product chart error: {e}")
        return None


def _build_rating_chart(data: dict, date_range: str = "All") -> dict | None:
    """
    Build review rating trend chart with optional date filtering.
    Shows avg_rating and sentiment ratios over time.
    """
    try:
        import plotly.graph_objects as go
        import pandas as pd

        insights = data.get("review_insights")
        if not isinstance(insights, pd.DataFrame) or insights.empty:
            return None

        df = insights.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Date range filter
        if date_range == "Last 1 year":
            cutoff = df["date"].max() - pd.DateOffset(years=1)
            df = df[df["date"] >= cutoff]
        elif date_range == "Last 2 years":
            cutoff = df["date"].max() - pd.DateOffset(years=2)
            df = df[df["date"] >= cutoff]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["date"], y=df["avg_rating"],
            mode="lines+markers",
            name="Avg Rating",
            line=dict(color="#636EFA", width=2),
            yaxis="y1",
        ))

        fig.add_trace(go.Scatter(
            x=df["date"], y=df["positive_ratio"] * 100,
            mode="lines",
            name="Positive %",
            line=dict(color="#00CC96", width=1.5, dash="dot"),
            yaxis="y2",
        ))

        fig.add_trace(go.Scatter(
            x=df["date"], y=df["negative_ratio"] * 100,
            mode="lines",
            name="Negative %",
            line=dict(color="#EF553B", width=1.5, dash="dot"),
            yaxis="y2",
        ))

        fig.update_layout(
            title="Review Rating & Sentiment Trend",
            xaxis_title="Date",
            yaxis=dict(title="Avg Rating (1–5)", range=[0, 5]),
            yaxis2=dict(
                title="Sentiment %",
                overlaying="y", side="right",
                range=[0, 100],
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=60, b=20, l=0, r=0),
            hovermode="x unified",
        )

        latest = df.iloc[-1]
        text = (
            f"Rating trend chart. Most recent period: "
            f"avg rating **{latest['avg_rating']:.2f}/5**, "
            f"**{latest['positive_ratio']*100:.0f}%** positive, "
            f"**{latest['negative_ratio']*100:.0f}%** negative."
        )
        return {
            "text": text, "fig": fig, "table": None,
            "image_path": None, "extra_images": None,
            "citations": "📊 *Sources: Review Insights*",
            "mode": "rag", "chunks": [],
        }
    except Exception as e:
        print(f"[pipeline] rating chart error: {e}")
        return None


def get_response(
    question: str,
    data:     dict,  # loaded CSV DataFrames from chatbot.py (used by keyword fallback)
    ctx,             # ChatContext from response_logic.py
) -> tuple[dict, object]:
    """
    Adaptive entry point called by chatbot.py.

    Checks Ollama availability at runtime:
      - Available → RAG pipeline (intelligent mode)
      - Unavailable → keyword matcher (graceful degradation)

    This is the single function chatbot.py calls — it handles
    routing transparently without the UI needing to know which
    mode is active. A small indicator in the response payload
    tells the UI which mode was used so it can show a badge.

    Parameters
    ----------
    question : user's message
    data     : dict of DataFrames loaded by chatbot.py
    ctx      : ChatContext (filters, selected product, etc.)

    Returns
    -------
    (payload_dict, updated_ctx)
    payload_dict keys: text, citations, chunks, mode, fig, table, image_path
    """
    # ── Visual intents: handle before RAG (RAG returns text, not figures) ──
    q_lower = question.lower()

    # Heatmap — show overlay images or zone bar chart
    if any(kw in q_lower for kw in _HEATMAP_KEYWORDS):
        heatmap_payload = _build_heatmap(data, query=question)
        if heatmap_payload:
            return heatmap_payload, ctx

    # Busiest zone — bar chart showing people count per zone
    if any(kw in q_lower for kw in {"busiest zone", "busiest zones", "which zone",
                                     "most traffic", "zone traffic", "zone chart",
                                     "traffic chart", "people per zone"}):
        zone_payload = _build_heatmap(data, query=question)
        if zone_payload:
            # Override text to be zone-comparison framing, not heatmap framing
            zone_payload["text"] = zone_payload["text"].replace(
                "Zone overlay heatmaps", "Zone traffic comparison"
            )
            return zone_payload, ctx

    # Product sales chart
    if any(kw in q_lower for kw in _PLOT_KEYWORDS):
        pid = _extract_product_id(question, data)
        if pid:
            chart = _build_product_chart(pid, data)
            if chart:
                return chart, ctx
        else:
            # No product ID found — ask user to specify
            return {
                "text": (
                    "Which product would you like to plot? "
                    "Please include the product ID, for example: "
                    "*plot product 85123A*"
                ),
                "fig": None, "table": None, "image_path": None,
                "extra_images": None, "citations": "",
                "mode": "rag", "chunks": [],
            }, ctx

    # Rating trend chart
    if any(kw in q_lower for kw in _RATING_KEYWORDS):
        rating = _build_rating_chart(data)
        if rating:
            return rating, ctx

    # ── Adaptive mode — check session state toggle first ──
    # response_mode is set by the sidebar toggle in chatbot.py:
    #   "keyword" → always use rule-based matcher (instant)
    #   "ollama"  → always try Ollama (fail loudly if down)
    #   "auto"    → use Ollama if available, else keyword (default)
    #
    # We read it from a module-level variable set by chatbot.py
    # because pipeline.py has no direct access to st.session_state.
    import streamlit as _st
    try:
        _forced_mode = _st.session_state.get("response_mode", "ollama")
    except Exception:
        _forced_mode = "ollama"

    _use_ollama = (
        _forced_mode == "ollama" and ollama_available()
    ) or (
        _forced_mode == "auto" and ollama_available()
    )

    if _use_ollama:
        try:
            result = rag_response(question, top_k=3)
            payload = {
                "text":       result["text"],
                "citations":  result["citations"],
                "chunks":     result["chunks"],
                "mode":       "rag",
                "fig":        None,
                "table":      None,
                "image_path": None,
            }
            return payload, ctx

        except Exception as e:
            print(f"[pipeline] RAG error: {e} — falling back to keyword matcher")

    # ── Keyword fallback ──
    from modules.chatbot.response_logic import get_response as keyword_get_response
    payload, updated_ctx = keyword_get_response(question, data, ctx)
    payload["mode"]      = "keyword"
    payload["citations"] = ""
    payload["chunks"]    = []
    return payload, updated_ctx
