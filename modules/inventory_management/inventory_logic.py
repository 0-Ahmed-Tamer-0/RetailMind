"""inventory_logic.py — Inventory & discount decisions
Upgraded from fixed-rule to data-driven logic.

Two upgrades over the original
───────────────────────────────
1. Dynamic safety stock (from forecast uncertainty)
   Original used a hardcoded 20% buffer for every product.
   Problem: a product with a tight, confident forecast needs less buffer
   than one with a wide, uncertain forecast. Using the same 20% for both
   either wastes capital (over-ordering confident products) or under-protects
   volatile ones.

   Fix: derive the safety stock ratio from Prophet's confidence interval:
     uncertainty_ratio = mean(predicted_high - predicted_low) / mean(predicted_sales)
     safety_stock_pct  = clamp(uncertainty_ratio, 0.15, 0.40)

   This directly connects inventory decisions to forecast quality —
   a wider confidence interval means more uncertainty means more buffer.
   Fully justified by the data, no hardcoded business assumptions.

2. ABC classification
   Not all products deserve the same reorder rules.
   A products (top 20% by revenue) are high-value — running out is costly,
   so they get a higher safety stock multiplier and a tighter reorder trigger.
   C products (bottom 50%) are low-value — over-ordering them ties up capital,
   so they get a looser trigger and a lower discount threshold.

   This is standard retail inventory management (Pareto principle applied
   to stock control). Used by every major retail ERP system.

   ABC tiers:
     A — top 20% cumulative revenue  → safety_multiplier 1.3, reorder at 110% of demand
     B — next 30% cumulative revenue → safety_multiplier 1.0, reorder at 100% of demand
     C — bottom 50%                  → safety_multiplier 0.7, reorder at  90% of demand

Output columns
──────────────
inventory_recommendations.csv:
    product_id, abc_class, current_stock, predicted_demand_7d,
    safety_stock_pct, safety_stock, required_stock,
    decision, recommended_order, reason

discount_recommendations.csv:
    product_id, abc_class, current_stock, predicted_demand_7d,
    discount_action, discount_rate, reason
"""

from __future__ import annotations
import pandas as pd
import numpy as np


# ── ABC classification ────────────────────────────────────────────────────────

def classify_abc(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ABC class to each product based on cumulative revenue contribution.

    Method
    ------
    1. Sum predicted_sales per product across the full forecast horizon
    2. Sort descending by total revenue
    3. Compute cumulative revenue share
    4. A = products whose cumulative share reaches 0–50%
       B = products whose cumulative share reaches 50–80%
       C = products whose cumulative share reaches 80–100%

    Why cumulative share not simple top-N?
    ───────────────────────────────────────
    Top-N is arbitrary — "top 2 products" means something very different
    if one product has 90% of revenue vs 25% of revenue.
    Cumulative share adapts to your actual revenue distribution.

    Returns
    -------
    DataFrame: product_id, total_revenue, revenue_share, cumulative_share, abc_class
    """
    rev = (
        forecast_df.groupby("product_id")["predicted_sales"]
        .sum()
        .reset_index()
        .rename(columns={"predicted_sales": "total_revenue"})
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )

    total = rev["total_revenue"].sum()
    rev["revenue_share"]    = rev["total_revenue"] / total
    rev["cumulative_share"] = rev["revenue_share"].cumsum()

    def _tier(cum):
        if cum <= 0.50: return "A"
        if cum <= 0.80: return "B"
        return "C"

    rev["abc_class"] = rev["cumulative_share"].apply(_tier)
    return rev[["product_id", "total_revenue", "revenue_share", "cumulative_share", "abc_class"]]


# ── Dynamic safety stock ──────────────────────────────────────────────────────

def _safety_stock_pct(
    horizon_fc: pd.DataFrame,
    abc_class: str,
) -> float:
    """
    Derive safety stock percentage from forecast uncertainty + ABC tier.

    uncertainty_ratio = mean interval width / mean predicted demand
    This measures how wide Prophet's confidence band is relative to
    the point forecast — a proxy for how reliable the forecast is.

    ABC multipliers adjust the base uncertainty ratio:
      A products: multiply by 1.3 (higher stakes → more buffer)
      B products: multiply by 1.0 (standard)
      C products: multiply by 0.7 (lower stakes → less buffer)

    Hard clamps: [0.10, 0.50]
      Never below 10% (minimum prudent buffer)
      Never above 50% (prevents excessive capital lock-up)
    """
    predicted = horizon_fc["predicted_sales"]
    low       = horizon_fc["predicted_low"]
    high      = horizon_fc["predicted_high"]

    mean_demand = predicted.mean()
    if mean_demand <= 0:
        return 0.20  # fallback for zero-demand products

    mean_interval_width = (high - low).mean()
    uncertainty_ratio   = mean_interval_width / mean_demand

    abc_multiplier = {"A": 1.3, "B": 1.0, "C": 0.7}.get(abc_class, 1.0)
    raw = uncertainty_ratio * abc_multiplier

    return float(np.clip(raw, 0.10, 0.50))


# ── Reorder trigger threshold ─────────────────────────────────────────────────

_REORDER_THRESHOLD = {"A": 1.10, "B": 1.00, "C": 0.90}
# A products: reorder when required > 110% of stock (earlier trigger)
# B products: reorder when required > 100% of stock (at the line)
# C products: reorder when required >  90% of stock (later trigger)

_DISCOUNT_THRESHOLD = {"A": 2.0, "B": 1.5, "C": 1.2}
# A products: only discount when stock is 2× demand (high bar — A items sell)
# B products: discount at 1.5× (original threshold)
# C products: discount at 1.2× (low bar — C items sit on shelves)

_DISCOUNT_RATE = {"A": 10, "B": 20, "C": 30}
# A products: small discount (10%) — they sell anyway, protect margin
# B products: moderate discount (20%)
# C products: aggressive discount (30%) — need to move slow stock


# ── Main logic ────────────────────────────────────────────────────────────────

def inventory_and_discount_logic(
    forecast_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    abc_df: pd.DataFrame,
    horizon_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate inventory and discount recommendations.

    Parameters
    ----------
    forecast_df  : sales_forecast.csv — must have columns:
                   date, product_id, predicted_sales, predicted_low, predicted_high
    stock_df     : current_stock.csv — must have columns:
                   product_id, current_stock
    abc_df       : output of classify_abc() — must have:
                   product_id, abc_class
    horizon_days : how many forecast days to sum for demand estimate

    Returns
    -------
    (inventory_df, discount_df) — both DataFrames ready to save as CSV
    """
    forecast_df = forecast_df.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    # Merge ABC class onto stock
    stock = stock_df.merge(abc_df[["product_id", "abc_class"]], on="product_id", how="left")
    stock["abc_class"] = stock["abc_class"].fillna("B")  # default B if not classified

    inventory_rows = []
    discount_rows  = []

    for _, row in stock.iterrows():
        pid           = str(row["product_id"])
        current_stock = float(row["current_stock"])
        abc_class     = str(row["abc_class"])

        # Forecast horizon for this product
        horizon_fc = (
            forecast_df[forecast_df["product_id"] == pid]
            .sort_values("date")
            .head(horizon_days)
        )

        if horizon_fc.empty:
            continue

        predicted_demand = float(horizon_fc["predicted_sales"].sum())
        if predicted_demand <= 0:
            predicted_demand = 0.0

        # Dynamic safety stock
        ss_pct     = _safety_stock_pct(horizon_fc, abc_class)
        safety_stk = predicted_demand * ss_pct
        required   = predicted_demand + safety_stk

        # ── Inventory decision ──
        reorder_trigger = required * _REORDER_THRESHOLD[abc_class]

        if reorder_trigger > current_stock:
            decision     = "REORDER"
            reorder_qty  = int(np.ceil(required - current_stock))
            inv_reason   = (
                f"{abc_class}-class product · "
                f"forecast {predicted_demand:.0f} units · "
                f"safety buffer {ss_pct:.0%} · "
                f"required {required:.0f} > stock {current_stock:.0f}"
            )
        else:
            decision    = "OK"
            reorder_qty = 0
            inv_reason  = (
                f"{abc_class}-class product · "
                f"stock {current_stock:.0f} covers "
                f"forecast {predicted_demand:.0f} + buffer"
            )

        inventory_rows.append({
            "product_id":           pid,
            "abc_class":            abc_class,
            "current_stock":        int(current_stock),
            "predicted_demand_7d":  round(predicted_demand, 1),
            "safety_stock_pct":     round(ss_pct, 3),
            "safety_stock":         round(safety_stk, 1),
            "required_stock":       round(required, 1),
            "decision":             decision,
            "recommended_order":    reorder_qty,
            "reason":               inv_reason,
        })

        # ── Discount decision ──
        disc_trigger = predicted_demand * _DISCOUNT_THRESHOLD[abc_class]

        if predicted_demand > 0 and current_stock > disc_trigger:
            disc_action  = "DISCOUNT"
            disc_rate    = _DISCOUNT_RATE[abc_class]
            disc_reason  = (
                f"Overstock · stock {current_stock:.0f} is "
                f"{current_stock/predicted_demand:.1f}× forecast demand · "
                f"{abc_class}-class markdown rate applied"
            )
        else:
            disc_action = "NONE"
            disc_rate   = 0
            disc_reason = "Stock level within acceptable range"

        discount_rows.append({
            "product_id":          pid,
            "abc_class":           abc_class,
            "current_stock":       int(current_stock),
            "predicted_demand_7d": round(predicted_demand, 1),
            "discount_action":     disc_action,
            "discount_rate":       disc_rate,
            "reason":              disc_reason,
        })

    return pd.DataFrame(inventory_rows), pd.DataFrame(discount_rows)
