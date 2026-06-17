"""
response_logic.py

Offline chatbot brain (Option B, upgraded):
- Loads your project outputs from data/outputs/
- Detects intent + entities (product_id, segment_name, zone_id)
- Returns a payload your Streamlit UI can render:
    {"text": str, "table": pd.DataFrame|None, "fig": plotly.graph_objs.Figure|None,
     "image_path": str|None, "extra_images": list[str]|None}

This version:
- Removes all foot-traffic + heatmap logic
- Adds zone overlay images only (cv_zone_overlays/)
- Supports queries like: "show heatmap for zone 1" (mapped to overlays)
"""

from __future__ import annotations

import re
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go


# ----------------------------
# Paths (assume you run Streamlit from repo root)
# ----------------------------
OUTPUTS_DIR = Path("data/outputs")
OVERLAYS_DIR = OUTPUTS_DIR / "cv_zone_overlays"

FILES = {
    "cv": OUTPUTS_DIR / "cv_foot_traffic.csv",
    "sales": OUTPUTS_DIR / "sales_forecast.csv",
    "inventory": OUTPUTS_DIR / "inventory_recommendations.csv",
    "discount": OUTPUTS_DIR / "discount_recommendations.csv",
    "review_insights": OUTPUTS_DIR / "review_insights.csv",
    "review_keywords": OUTPUTS_DIR / "review_keywords.csv",
    "segments": OUTPUTS_DIR / "customer_segments.csv",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _normalize_string_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].astype(str).str.strip()
    return out


# ----------------------------
# Context (keeps selections across chat)
# ----------------------------
@dataclass
class ChatContext:
    selected_product_id: Optional[str] = None
    selected_segment_name: Optional[str] = None
    selected_zone_id: Optional[str] = None
    # Review chart preferences (set from UI)
    review_range: str = "All"  # All | Last 30 days | Last 90 days | Last 1 year
    review_resample: str = "Monthly"  # Daily | Weekly | Monthly
    review_smooth: bool = True


# ----------------------------
# Data loading
# ----------------------------
def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except Exception:
            return None


def _auto_find_csv(required_cols: set[str]) -> Optional[Path]:
    """Best-effort: scan data/outputs for a CSV that contains required columns."""
    if not OUTPUTS_DIR.exists():
        return None
    for p in OUTPUTS_DIR.glob("*.csv"):
        try:
            head = pd.read_csv(p, nrows=5)
            cols = {str(c).strip() for c in head.columns}
            if required_cols.issubset(cols):
                return p
        except Exception:
            continue
    return None


_ZONE_RE_1 = re.compile(r"\bzone\s*[_:\-]?\s*(\d{1,3})\s*([a-cA-C])?\b")
_ZONE_RE_2 = re.compile(r"\b(\d{1,3})\s*([a-cA-C])\b")


def _canonical_zone_id(zone_num: str, suffix: Optional[str]) -> str:
    zone_num = str(zone_num).strip()
    if suffix:
        return f"{zone_num}_{suffix.upper()}"
    return zone_num


def parse_zone_from_text(text: str) -> Optional[str]:
    t = text.strip()

    m = _ZONE_RE_1.search(t)
    if m:
        return _canonical_zone_id(m.group(1), m.group(2))

    # allow "41a heatmap" without the word "zone"
    m = _ZONE_RE_2.search(t)
    if m:
        return _canonical_zone_id(m.group(1), m.group(2))

    return None


def load_overlays() -> Dict[str, str]:
    """
    Scan data/outputs/cv_zone_overlays for image files and return:
      { "1": "path/to/file.png", "41_A": "...", ... }
    Accepts flexible filenames; tries to extract zone id from the filename.
    """
    overlays: Dict[str, str] = {}
    if not OVERLAYS_DIR.exists():
        return overlays

    for p in list(OVERLAYS_DIR.glob("*.png")) + list(OVERLAYS_DIR.glob("*.jpg")) + list(OVERLAYS_DIR.glob("*.jpeg")):
        stem = p.stem.lower()

        # Try to match common patterns like:
        # zone_1, zone1, overlay_zone_1, zone_41_a, 41_a, etc.
        m = re.search(r"(?:^|[^0-9a-z])zone[_\- ]?(\d{1,3})(?:[_\- ]?([a-c]))?(?:$|[^0-9a-z])", stem)
        if not m:
            m = re.search(r"(?:^|[^0-9a-z])(\d{1,3})(?:[_\- ]?([a-c]))?(?:$|[^0-9a-z])", stem)
        if not m:
            continue

        zone_id = _canonical_zone_id(m.group(1), m.group(2))
        overlays[zone_id] = str(p)

    return overlays


def load_outputs() -> Dict[str, Any]:
    """Load all outputs. Missing files are returned as None."""
    data: Dict[str, Any] = {}
    
    cv_path = FILES["cv"] if FILES["cv"].exists() else _auto_find_csv({"timestamp", "zone_id", "people_count"})
    sales_path = FILES["sales"] if FILES["sales"].exists() else _auto_find_csv({"date", "product_id", "predicted_sales"})
    inv_path = FILES["inventory"] if FILES["inventory"].exists() else _auto_find_csv({"product_id", "decision", "recommended_order"})
    disc_path = FILES["discount"] if FILES["discount"].exists() else _auto_find_csv({"product_id", "discount_action", "discount_rate"})
    ri_path = FILES["review_insights"] if FILES["review_insights"].exists() else _auto_find_csv({"date", "avg_rating"})
    rk_path = FILES["review_keywords"] if FILES["review_keywords"].exists() else _auto_find_csv({"sentiment", "keyword", "count"})
    seg_path = FILES["segments"] if FILES["segments"].exists() else _auto_find_csv({"cluster_id", "segment_name"})

    data["cv"] = _safe_read_csv(cv_path) if cv_path else None
    data["sales"] = _safe_read_csv(sales_path) if sales_path else None
    data["inventory"] = _safe_read_csv(inv_path) if inv_path else None
    data["discount"] = _safe_read_csv(disc_path) if disc_path else None
    data["review_insights"] = _safe_read_csv(ri_path) if ri_path else None
    data["review_keywords"] = _safe_read_csv(rk_path) if rk_path else None
    data["segments"] = _safe_read_csv(seg_path) if seg_path else None

    # Overlays only (no foot traffic / no heatmap)
    data["overlays"] = load_overlays()

    return data


def outputs_status() -> Dict[str, Any]:
    """Human-friendly status for the Setup page."""
    required = {
        "cv_foot_traffic.csv": ({"timestamp", "zone_id", "people_count"}, FILES["cv"]),
        "sales_forecast.csv": ({"date", "product_id", "predicted_sales"}, FILES["sales"]),
        "inventory_recommendations.csv": ({"product_id", "decision", "recommended_order"}, FILES["inventory"]),
        "discount_recommendations.csv": ({"product_id", "discount_action", "discount_rate"}, FILES["discount"]),
        "review_insights.csv": ({"date", "avg_rating"}, FILES["review_insights"]),
        "review_keywords.csv": ({"sentiment", "keyword", "count"}, FILES["review_keywords"]),
        "customer_segments.csv": ({"cluster_id", "segment_name"}, FILES["segments"]),
    }

    status: Dict[str, Any] = {}
    for filename, (cols, expected_path) in required.items():
        exists = expected_path.exists()
        found = None
        if not exists:
            cand = _auto_find_csv(cols)
            found = str(cand) if cand else None

        mtime = None
        if exists:
            try:
                mtime = pd.Timestamp(expected_path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                mtime = None

        status[filename] = bool(exists)

    overlays = load_overlays()
    status["cv_zone_overlays"] = {
        "exists": bool(OVERLAYS_DIR.exists()),
        "overlay_count": len(overlays),
        "example_zones": sorted(list(overlays.keys()))[:10],
    }
    return status


# ----------------------------
# Helpers for payloads
# ----------------------------
def _missing_payload(name: str) -> Dict[str, Any]:
    return {"text": f"I couldn't find `{name}` in `data/outputs/`. Please generate outputs first.", "table": None, "fig": None, "image_path": None}


def _df_head_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    try:
        return df.head(n)
    except Exception:
        return df


# ----------------------------
# Entity extraction
# ----------------------------
PRODUCT_ID_RE = re.compile(r"\b([A-Za-z0-9]{3,15})\b")


def extract_product_id(message: str, sales_df: Optional[pd.DataFrame]) -> Optional[str]:
    if sales_df is None or "product_id" not in sales_df.columns:
        return None

    m = re.search(r"product\s*[:#-]?\s*([A-Za-z0-9]{3,15})", message, flags=re.IGNORECASE)
    if m:
        cand = m.group(1)
        if cand in set(sales_df["product_id"].astype(str)):
            return str(cand)

    known = set(sales_df["product_id"].astype(str))
    for tok in PRODUCT_ID_RE.findall(message):
        if tok in known:
            return str(tok)

    return None


def extract_segment_name(message: str, seg_df: Optional[pd.DataFrame]) -> Optional[str]:
    if seg_df is None or "segment_name" not in seg_df.columns:
        return None

    seg_names = [str(s) for s in seg_df["segment_name"].dropna().unique()]
    msg = message.lower()
    for name in seg_names:
        if name.lower() in msg:
            return name

    # Common shortcuts
    if "frequent" in msg:
        return _match_contains(seg_names, "frequent")
    if "promo" in msg or "discount" in msg:
        return _match_contains(seg_names, "promo")
    if "high" in msg and "spend" in msg:
        return _match_contains(seg_names, "high")
    if "low" in msg and ("engage" in msg or "inactive" in msg):
        return _match_contains(seg_names, "low")

    return None


def _match_contains(options: list[str], needle: str) -> Optional[str]:
    for opt in options:
        if needle.lower() in opt.lower():
            return opt
    return None


def extract_zone_id(message: str, overlays: Dict[str, str]) -> Optional[str]:
    if not overlays:
        return None

    z = parse_zone_from_text(message)
    if z and z in overlays:
        return z

    # If message contains exact zone_id token (e.g., "41_A")
    msg_tokens = set(re.findall(r"[A-Za-z0-9_\-]+", message))
    for zone_id in overlays.keys():
        if zone_id in msg_tokens:
            return zone_id

    return None


# ----------------------------
# Intent scoring
# ----------------------------
INTENTS = {
    "greeting": {
        "keywords": ["hi", "hello", "hey", "salam"],
        "examples": ["hi", "hello", "hey"],
    },
    "help": {
        "keywords": ["help", "commands", "features", "what can you do"],
        "examples": ["help", "what can you do", "commands"],
    },
    "sales_top10": {
        "keywords": ["top", "top 10", "top-10", "highest", "predicted", "forecast top"],
        "examples": ["top 10 products", "highest predicted sales"],
    },
    "sales_product_trend": {
        "keywords": ["plot", "trend", "actual", "predicted", "forecast for product", "sales trend"],
        "examples": ["plot sales trend for product 85048", "actual vs predicted"],
    },
    "inventory_top": {
        "keywords": ["reorder", "restock", "running out", "inventory recommendation", "stock recommendation"],
        "examples": ["what should we reorder", "reorder list"],
    },
    "discount_top": {
        "keywords": ["discount", "markdown", "promotion", "promo", "apply discount"],
        "examples": ["which products need discount", "recommend promotions"],
    },
    "reviews_keywords_negative": {
        "keywords": ["complaint", "complaints", "problem", "issue", "worst", "negative", "hate", "refund"],
        "examples": ["top complaints", "most common problems"],
    },
    "reviews_keywords_positive": {
        "keywords": ["like", "likes", "love", "best", "good", "positive", "great"],
        "examples": ["top likes", "what do customers like"],
    },
    "reviews_summary": {
        "keywords": ["review", "reviews", "rating", "sentiment", "feedback"],
        "examples": ["review summary", "sentiment trend"],
    },
    "segments_overview": {
        "keywords": ["segment", "segments", "segmentation", "cluster", "customer groups", "clusters"],
        "examples": ["describe customer segments", "show clusters"],
    },
    "busiest_zones": {
    "keywords": ["busiest", "busy", "crowded", "top zones", "busiest zones", "most crowded", "highest traffic", "zone traffic"],
    "examples": ["busiest zones", "top zones by traffic", "most crowded zones", "which zone is busiest"],
    },
    # Overlays (no heatmap / no foot traffic)
    "show_overlay": {
        # include "heatmap" words because users might still say it
        "keywords": ["overlay", "overlays", "zone overlay", "show overlay", "heatmap", "heat map", "zone", "show zone"],
        "examples": ["show overlay for zone 1", "show heatmap for zone 1", "zone 41a overlay"],
    },
}


def _fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _score_intent(msg: str, intent: str) -> float:
    cfg = INTENTS[intent]
    score = 0.0

    # Keyword hits
    for kw in cfg["keywords"]:
        if kw in msg:
            score += 3.0

    # Token-level partial matching
    tokens = re.findall(r"[a-z0-9_\-]+", msg)
    for kw in cfg["keywords"]:
        kw_tokens = re.findall(r"[a-z0-9_\-]+", kw)
        if not kw_tokens:
            continue
        for t in tokens:
            for k in kw_tokens:
                if len(t) >= 4 and len(k) >= 4 and _fuzzy_ratio(t, k) > 0.85:
                    score += 0.5

    # Example similarity bonus
    ex_best = 0.0
    for ex in cfg["examples"]:
        ex_best = max(ex_best, _fuzzy_ratio(msg, ex.lower()))
    score += 2.0 * ex_best

    return score


def detect_intent(message: str) -> str:
    msg = message.lower().strip()

    if any(w in msg for w in ["busiest", "most crowded", "top zones", "highest traffic", "zone traffic"]):
        return "busiest_zones"
    
    # Strong priority: overlays (users may say heatmap but mean overlay)
    if ("overlay" in msg or "heatmap" in msg or "heat map" in msg) and ("zone" in msg or parse_zone_from_text(msg)):
        return "show_overlay"

    scored = [(intent, _score_intent(msg, intent)) for intent in INTENTS.keys()]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_intent, best_score = scored[0]

    if best_score < 0.8:
        return "help"

    return best_intent


# ----------------------------
# Response builders (existing ones kept)
# ----------------------------
def sales_top10(data: Dict[str, Any]) -> Dict[str, Any]:
    df = data.get("sales")
    if df is None:
        return _missing_payload("sales_forecast.csv")

    df = _normalize_columns(df)
    for c in ["date", "product_id", "predicted_sales"]:
        if c not in df.columns:
            return {"text": f"`{c}` column missing in sales forecast.", "table": _df_head_table(df, 10), "fig": None, "image_path": None}

    out = df.copy()
    out["predicted_sales"] = pd.to_numeric(out["predicted_sales"], errors="coerce").fillna(0)
    top = (
        out.groupby("product_id", as_index=False)["predicted_sales"]
        .sum()
        .sort_values("predicted_sales", ascending=False)
        .head(10)
    )

    return {"text": "Top 10 products by predicted demand:", "table": top, "fig": None, "image_path": None}


def sales_product_trend(data: Dict[str, Any], product_id: str) -> Dict[str, Any]:
    df = data.get("sales")
    if df is None:
        return _missing_payload("sales_forecast.csv")

    df = _normalize_columns(df)
    needed = {"date", "product_id"}
    if not needed.issubset(set(df.columns)):
        return {"text": "Sales forecast file missing required columns.", "table": _df_head_table(df, 10), "fig": None, "image_path": None}

    pred_col = "predicted_sales" if "predicted_sales" in df.columns else None
    actual_col = "actual_sales" if "actual_sales" in df.columns else None

    sub = df[df["product_id"].astype(str) == str(product_id)].copy()
    if sub.empty:
        return {"text": f"I couldn't find product `{product_id}` in the sales forecast.", "table": None, "fig": None, "image_path": None}

    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.sort_values("date")

    fig = go.Figure()
    if actual_col:
        fig.add_trace(go.Scatter(x=sub["date"], y=pd.to_numeric(sub[actual_col], errors="coerce"), mode="lines", name="Actual"))
    if pred_col:
        fig.add_trace(go.Scatter(x=sub["date"], y=pd.to_numeric(sub[pred_col], errors="coerce"), mode="lines", name="Predicted"))

    fig.update_layout(title=f"Sales Trend — Product {product_id}", xaxis_title="Date", yaxis_title="Sales")

    return {"text": f"Sales trend for product `{product_id}`:", "table": None, "fig": fig, "image_path": None}


def inventory_top(data: Dict[str, Any]) -> Dict[str, Any]:
    df = data.get("inventory")
    if df is None:
        return _missing_payload("inventory_recommendations.csv")

    df = _normalize_columns(df)
    if "decision" not in df.columns:
        return {"text": "`decision` column missing in inventory recommendations.", "table": _df_head_table(df, 10), "fig": None, "image_path": None}

    sub = df[df["decision"].astype(str).str.upper().str.contains("REORDER", na=False)].copy()
    if sub.empty:
        return {"text": "No reorder recommendations found.", "table": _df_head_table(df, 10), "fig": None, "image_path": None}

    return {"text": "Products recommended for reorder:", "table": sub.head(20), "fig": None, "image_path": None}


def discount_top(data: Dict[str, Any]) -> Dict[str, Any]:
    df = data.get("discount")
    if df is None:
        return _missing_payload("discount_recommendations.csv")

    df = _normalize_columns(df)
    if "discount_action" not in df.columns:
        return {"text": "`discount_action` column missing in discount recommendations.", "table": _df_head_table(df, 10), "fig": None, "image_path": None}

    sub = df[df["discount_action"].astype(str).str.upper().str.contains("DISCOUNT", na=False)].copy()
    if sub.empty:
        return {"text": "No discount recommendations found.", "table": _df_head_table(df, 10), "fig": None, "image_path": None}

    return {"text": "Products recommended for discount:", "table": sub.head(20), "fig": None, "image_path": None}


def reviews_keywords(data: Dict[str, Any], sentiment: str) -> Dict[str, Any]:
    df = data.get("review_keywords")
    if df is None:
        return _missing_payload("review_keywords.csv")

    df = _normalize_columns(df)
    df = _normalize_string_col(df, "sentiment")
    if "sentiment" not in df.columns or "keyword" not in df.columns:
        return {"text": "Review keywords file missing required columns.", "table": _df_head_table(df, 10), "fig": None, "image_path": None}

    sub = df[df["sentiment"].astype(str).str.lower() == sentiment.lower()].copy()
    if sub.empty:
        return {"text": f"No {sentiment} keywords found.", "table": _df_head_table(df, 10), "fig": None, "image_path": None}

    if "count" in sub.columns:
        sub["count"] = pd.to_numeric(sub["count"], errors="coerce").fillna(0)
        sub = sub.sort_values("count", ascending=False)

    return {"text": f"Top {sentiment} keywords:", "table": sub.head(15), "fig": None, "image_path": None}


def reviews_summary(data: Dict[str, Any], ctx: ChatContext) -> Dict[str, Any]:
    df = data.get("review_insights")
    if df is None:
        return _missing_payload("review_insights.csv")

    df = _normalize_columns(df)
    if "date" not in df.columns or "avg_rating" not in df.columns:
        return {"text": "Review insights file missing required columns.", "table": _df_head_table(df, 10), "fig": None, "image_path": None}

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["avg_rating"] = pd.to_numeric(out["avg_rating"], errors="coerce")

    out = out.dropna(subset=["date"]).sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=out["date"], y=out["avg_rating"], mode="lines", name="Avg rating"))
    fig.update_layout(title="Average Rating Over Time", xaxis_title="Date", yaxis_title="Avg Rating")

    return {"text": "Review summary (average rating over time):", "table": None, "fig": fig, "image_path": None}


def segments_overview(data: Dict[str, Any]) -> Dict[str, Any]:
    df = data.get("segments")
    if df is None:
        return _missing_payload("customer_segments.csv")

    df = _normalize_columns(df)

    required = {"segment_name", "customer_count"}
    if not required.issubset(set(df.columns)):
        return {
            "text": "customer_segments.csv missing required columns (segment_name, customer_count).",
            "table": _df_head_table(df, 10),
            "fig": None,
            "image_path": None,
        }

    # Keep useful columns if they exist
    cols_order = [
        "segment_name",
        "customer_count",
        "avg_age",
        "avg_purchase_frequency",
        "avg_purchase_amount",
        "promo_usage_rate",
        "avg_satisfaction",
        "cluster_id",
    ]
    cols = [c for c in cols_order if c in df.columns]

    out = df[cols].copy()
    out["customer_count"] = pd.to_numeric(out["customer_count"], errors="coerce").fillna(0).astype(int)

    # sort by actual customer_count (correct)
    out = out.sort_values("customer_count", ascending=False)

    return {"text": "Customer segments (with summary metrics):", "table": out, "fig": None, "image_path": None}


def busiest_zones(data: Dict[str, Any]) -> Dict[str, Any]:
    df = data.get("cv")
    if df is None:
        return _missing_payload("cv_foot_traffic.csv")

    df = _normalize_columns(df)

    needed = {"zone_id", "people_count"}
    if not needed.issubset(set(df.columns)):
        return {
            "text": "cv_foot_traffic.csv is missing required columns (zone_id, people_count).",
            "table": _df_head_table(df, 10),
            "fig": None,
            "image_path": None,
        }

    out = df.copy()
    out["people_count"] = pd.to_numeric(out["people_count"], errors="coerce").fillna(0)

    grouped = (
        out.groupby("zone_id", as_index=False)
        .agg(
           max_people_per_frame=("people_count", "max"),
           total_people=("people_count", "sum"),
           avg_people_per_frame=("people_count", "mean"),
           frames=("people_count", "size"),
        )
       .sort_values("max_people_per_frame", ascending=False)
    )

    return {
        "text": "Busiest zones (sorted by total people count):",
        "table": grouped.head(15),
        "fig": None,
        "image_path": None,
    }



def show_overlay(data: Dict[str, Any], zone_id: Optional[str]) -> Dict[str, Any]:
    overlays: Dict[str, str] = data.get("overlays") or {}
    if not overlays:
        return {"text": f"I couldn't find any overlay images in `{OVERLAYS_DIR}`.", "table": None, "fig": None, "image_path": None}

    if not zone_id:
        examples = ", ".join(sorted(list(overlays.keys()))[:10])
        return {
            "text": "Which zone overlay do you want? Try: **show overlay for zone 1** (available examples: " + examples + ")",
            "table": None,
            "fig": None,
            "image_path": None,
        }

    path = overlays.get(zone_id)
    if not path:
        # Special handling: if user asked for zone 41 (no suffix), guide them to 41_A/41_B/41_C
        if zone_id == "41":
            choices = [z for z in overlays.keys() if z.startswith("41_")]
            if choices:
                return {
                    "text": "Zone 41 is split. Try one of: " + ", ".join(sorted(choices)),
                    "table": None,
                    "fig": None,
                    "image_path": None,
                }

        examples = ", ".join(sorted(list(overlays.keys()))[:10])
        return {
            "text": f"I couldn't find an overlay image for zone `{zone_id}`. Example zones: {examples}",
            "table": None,
            "fig": None,
            "image_path": None,
        }

    return {"text": f"Overlay image for zone `{zone_id}`:", "table": None, "fig": None, "image_path": path}


# ----------------------------
# Main router
# ----------------------------
def get_response(message: str, data: Dict[str, Any], ctx: ChatContext) -> Tuple[Dict[str, Any], ChatContext]:
    """Return (payload, updated_context)."""

    intent = detect_intent(message)

    # Extract entities from message and update context (non-destructive)
    pid = extract_product_id(message, data.get("sales"))
    if pid:
        ctx.selected_product_id = pid

    seg = extract_segment_name(message, data.get("segments"))
    if seg:
        ctx.selected_segment_name = seg

    overlays = data.get("overlays") or {}
    zone = extract_zone_id(message, overlays)
    if zone:
        ctx.selected_zone_id = zone

    # Route
    if intent == "greeting":
        return (
            {
                "text": "Hey! I can summarize your forecasts, inventory and discounts, reviews, segments, and show zone overlay images. Type **help** for examples.",
                "table": None,
                "fig": None,
                "image_path": None,
            },
            ctx,
        )

    if intent == "help":
        help_text = (
            "Try asking things like:\n"
            "- **Top-10 forecast**\n"
            "- **Plot sales trend for product 85048**\n"
            "- **What should we reorder?**\n"
            "- **Which products need discount?**\n"
            "- **Review summary** / **Top complaints** / **Top likes**\n"
            "- **Describe customer segments**\n"
            "- **Show overlay for zone 1** (you can also say 'heatmap' and I'll still show the overlay)"
        )
        return ({"text": help_text, "table": None, "fig": None, "image_path": None}, ctx)

    if intent == "sales_top10":
        return (sales_top10(data), ctx)

    if intent == "sales_product_trend":
        product_id = ctx.selected_product_id
        if not product_id:
            return (
                {"text": "Tell me the product_id first (example: `plot sales trend for product 85048`).", "table": None, "fig": None, "image_path": None},
                ctx,
            )
        return (sales_product_trend(data, product_id), ctx)

    if intent == "inventory_top":
        return (inventory_top(data), ctx)

    if intent == "discount_top":
        return (discount_top(data), ctx)

    if intent == "reviews_summary":
        return (reviews_summary(data, ctx), ctx)

    if intent == "reviews_keywords_negative":
        return (reviews_keywords(data, "negative"), ctx)

    if intent == "reviews_keywords_positive":
        return (reviews_keywords(data, "positive"), ctx)

    if intent == "segments_overview":
        return (segments_overview(data), ctx)
    
    if intent == "busiest_zones":
        return (busiest_zones(data), ctx)

    if intent == "show_overlay":
        # If user didn't specify zone in this message, use context
        return (show_overlay(data, ctx.selected_zone_id), ctx)

    # Fallback
    return ({"text": "I didn't catch that. Type **help** to see what I can do.", "table": None, "fig": None, "image_path": None}, ctx)
