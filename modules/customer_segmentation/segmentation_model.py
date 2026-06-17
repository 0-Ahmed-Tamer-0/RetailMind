"""segmentation_model.py — Customer segmentation using RFM + KMeans
Dataset: UCI Online Retail II (same file as forecasting — no separate CSV needed)

Core concepts
─────────────
RFM stands for:
  R  Recency   — days since the customer's last purchase
                 Lower = more recently active = more engaged
  F  Frequency — total number of distinct invoices (shopping trips)
                 Higher = more loyal
  M  Monetary  — total money spent across all purchases
                 Higher = more valuable

Why RFM instead of age/satisfaction?
  Those columns do not exist in real transaction data.
  RFM is computable from any purchase history and is the
  industry standard for customer segmentation (used by Amazon,
  Shopify analytics, every major CRM platform).

KMeans on 3 features:
  We StandardScale first so that Monetary (£thousands range)
  does not dominate Recency (days, small numbers) or
  Frequency (count, medium numbers).

Elbow method:
  We compute inertia for k=2..8 so the output_generator can
  pick the optimal k automatically, or the user can override.

Segment naming:
  After clustering, each cluster gets a business-friendly name
  based on its relative RFM profile:
    Champions        — recent, frequent, high spend
    Loyal Customers  — frequent, decent spend
    At Risk          — used to buy often but haven't recently
    Lost / Inactive  — low on all three dimensions
  (names assigned by ranking clusters on each dimension)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ── RFM computation ─────────────────────────────────────────────────────────

def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Compute RFM table from cleaned UCI transaction data.

    Parameters
    ----------
    df            : cleaned DataFrame (output of forecast_model.clean_uci)
    snapshot_date : reference date for Recency calculation.
                    Defaults to max(InvoiceDate) + 1 day so every customer
                    has Recency >= 1.

    Returns
    -------
    DataFrame indexed by Customer ID with columns:
        recency    (int)   days since last purchase
        frequency  (int)   number of distinct invoices
        monetary   (float) total spend in GBP
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].astype(int).astype(str)

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("Customer ID")
        .agg(
            recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            frequency=("Invoice", "nunique"),          # distinct trips, not rows
            monetary=("Sales", "sum"),
        )
        .reset_index()
    )

    return rfm


# ── Optimal k via elbow ─────────────────────────────────────────────────────

def elbow_inertias(rfm: pd.DataFrame, k_range: range = range(2, 9)) -> dict[int, float]:
    """
    Return {k: inertia} for each k in k_range.
    Useful for plotting or automatic elbow detection.
    """
    X = _scale(rfm)
    return {
        k: KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
        for k in k_range
    }


def best_k(rfm: pd.DataFrame, k_range: range = range(2, 9)) -> int:
    """
    Simple elbow detector: pick k where the marginal inertia drop
    is smallest relative to the previous drop (knee point).
    Falls back to k=4 if detection is ambiguous.
    """
    inertias = elbow_inertias(rfm, k_range)
    ks = sorted(inertias)
    drops = [inertias[ks[i]] - inertias[ks[i + 1]] for i in range(len(ks) - 1)]
    # second derivative: where does the drop stop accelerating?
    if len(drops) < 2:
        return 4
    accel = [drops[i] - drops[i + 1] for i in range(len(drops) - 1)]
    knee_idx = int(np.argmax(accel)) + 1      # +1 because accel is offset by 1
    return ks[knee_idx]


# ── Scaling helper ───────────────────────────────────────────────────────────

def _scale(rfm: pd.DataFrame) -> np.ndarray:
    """StandardScale the three RFM columns."""
    return StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])


# ── Clustering ───────────────────────────────────────────────────────────────

def run_kmeans(rfm: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """
    Cluster the RFM table.

    Parameters
    ----------
    rfm : output of compute_rfm()
    k   : number of clusters. If None, auto-detected via elbow method.

    Returns
    -------
    rfm DataFrame with an added 'cluster_id' column.
    """
    if k is None:
        k = best_k(rfm)

    X = _scale(rfm)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    rfm = rfm.copy()
    rfm["cluster_id"] = labels
    return rfm


# ── Segment naming ───────────────────────────────────────────────────────────

def summarize_clusters(rfm_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate clusters into a business-friendly summary table.

    Naming logic (rank-based, no hard-coded thresholds):
      Champions       — lowest recency  (most recent) + highest frequency
      Loyal Customers — highest frequency (not already Champions)
      At Risk         — highest recency (least recent) among mid-spenders
      Promising       — low recency but low frequency (new but engaged)
      Low Engagement  — remainder

    Output columns:
        cluster_id, segment_name, customer_count,
        avg_recency, avg_frequency, avg_monetary
    """
    summary = (
        rfm_clustered.groupby("cluster_id")
        .agg(
            customer_count=("Customer ID", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
    )

    # Start everyone as Low Engagement, then promote
    summary["segment_name"] = "Low Engagement"

    # Champions: most recent AND most frequent
    champ_idx = (summary["avg_recency"].rank() + summary["avg_frequency"].rank(ascending=False)).idxmin()
    summary.loc[champ_idx, "segment_name"] = "Champions"

    # Loyal: highest frequency (not already named)
    remaining = summary[summary["segment_name"] == "Low Engagement"]
    if not remaining.empty:
        loyal_idx = remaining["avg_frequency"].idxmax()
        summary.loc[loyal_idx, "segment_name"] = "Loyal Customers"

    # At Risk: highest recency (longest since purchase) among rest
    remaining = summary[summary["segment_name"] == "Low Engagement"]
    if not remaining.empty:
        risk_idx = remaining["avg_recency"].idxmax()
        summary.loc[risk_idx, "segment_name"] = "At Risk"

    # Promising: lowest recency among rest (new customers)
    remaining = summary[summary["segment_name"] == "Low Engagement"]
    if not remaining.empty:
        promise_idx = remaining["avg_recency"].idxmin()
        summary.loc[promise_idx, "segment_name"] = "Promising"

    # Round for readability
    summary["avg_recency"] = summary["avg_recency"].round(1)
    summary["avg_frequency"] = summary["avg_frequency"].round(1)
    summary["avg_monetary"] = summary["avg_monetary"].round(2)

    return summary
