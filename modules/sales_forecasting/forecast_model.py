"""forecast_model.py — Sales forecasting using Facebook Prophet
Dataset: UCI Online Retail II
Columns used: Invoice, StockCode, Quantity, InvoiceDate, Price

Core concepts
─────────────
• We compute Sales = Quantity × Price from raw transaction rows.
• Cancelled invoices start with "C" → filtered out (they are negative-quantity
  return records, not real sales).
• Prophet expects exactly two columns: ds (datestamp) and y (target value).
  We aggregate daily per product to get that shape.
• weekly_seasonality captures Mon–Sun sales rhythm (retail is highly weekly).
• yhat can be negative for slow-moving products with a downward trend —
  we clip to 0 because negative sales are meaningless for business decisions.
• We keep yhat_lower / yhat_upper so the chatbot can say
  "between X and Y units" rather than a single overconfident number.
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")          # Prophet prints a lot of Stan noise
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


# ── Cleaning ────────────────────────────────────────────────────────────────

def clean_uci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept raw UCI Online Retail II DataFrame and return a clean version.

    Steps
    -----
    1. Drop rows with no Customer ID (anonymous / system rows).
    2. Remove cancelled invoices (Invoice starts with 'C').
    3. Keep only rows where Quantity > 0 and Price > 0
       (data entry errors exist in the raw file).
    4. Compute Sales = Quantity × Price.
    5. Parse InvoiceDate to datetime if not already.
    """
    df = df.copy()

    # 1. Drop anonymous rows
    df = df.dropna(subset=["Customer ID"])

    # 2. Remove cancellations
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # 3. Remove nonsensical values
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    # 4. Sales value
    df["Sales"] = df["Quantity"] * df["Price"]

    # 5. Datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    return df.reset_index(drop=True)


def load_uci_excel(path: str, sheets: list[str] | None = None) -> pd.DataFrame:
    """
    Load one or both sheets from the UCI Online Retail II Excel file and
    concatenate them into a single cleaned DataFrame.

    Parameters
    ----------
    path   : path to online_retail_II.xlsx
    sheets : list of sheet names to load; defaults to both UCI sheets
    """
    if sheets is None:
        sheets = ["Year 2009-2010", "Year 2010-2011"]

    frames = []
    for sheet in sheets:
        try:
            raw = pd.read_excel(path, sheet_name=sheet)
            frames.append(raw)
        except Exception as exc:
            print(f"[load_uci_excel] Could not load sheet '{sheet}': {exc}")

    if not frames:
        raise ValueError(f"No sheets loaded from {path}")

    combined = pd.concat(frames, ignore_index=True)
    return clean_uci(combined)


# ── Aggregation ─────────────────────────────────────────────────────────────

def make_weekly_product_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    weekly = (
        df.groupby([
            pd.Grouper(key="InvoiceDate", freq="W-MON"),
            "StockCode",
        ])["Sales"]
        .sum()
        .reset_index()
    )
    weekly.columns = ["date", "product_id", "sales"]
    weekly["product_id"] = weekly["product_id"].astype(str)
    return weekly


def build_top_products(df: pd.DataFrame, top_n: int = 10) -> list[str]:
    """
    Return the top_n StockCodes by total revenue.

    We rank by total Sales (not just quantity) because high-unit / low-price
    items could dominate a quantity ranking while contributing little revenue.
    """
    top = (
        df.groupby("StockCode")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .astype(str)
        .tolist()
    )
    return top


# ── Forecasting ─────────────────────────────────────────────────────────────

def forecast_one_product(
    weekly_prod: pd.DataFrame,
    product_id: str,
    periods: int = 30,
) -> pd.DataFrame:
    """
    Fit a Prophet model on one product's history and return a forecast.

    Parameters
    ----------
    weekly_prod : output of make_weekly_product_sales()
    product_id : StockCode string
    periods    : how many future days to predict

    Returns
    -------
    DataFrame with columns: ds, yhat, yhat_lower, yhat_upper, product_id
    Returns empty DataFrame if history is too short (< 14 days).

    Design notes
    ------------
    • weekly_seasonality=True  — retail sales have strong Mon–Sun rhythm.
    • yearly_seasonality=True  — we have 2 years of data, so Prophet can
      learn Christmas / summer peaks.
    • changepoint_prior_scale=0.05 — slightly conservative; prevents Prophet
      from overfitting to short-term noise spikes.
    • yhat clipped to 0 — Prophet's linear trend can predict negative values
      for declining products; that's mathematically valid but meaningless
      for inventory decisions.
    """
    sub = weekly_prod[weekly_prod["product_id"] == product_id].copy()
    sub = sub.rename(columns={"date": "ds", "sales": "y"})

    if len(sub) < 14:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper", "product_id"])

    m = Prophet(
    weekly_seasonality=False,   # weekly data — no intra-week pattern
    yearly_seasonality=True,
    seasonality_mode="multiplicative",  # retail spikes scale with revenue
    changepoint_prior_scale=0.15,       # more flexible trend
    seasonality_prior_scale=10.0,       # stronger seasonal component
    interval_width=0.80,
)
    m.fit(sub[["ds", "y"]])

    future = m.make_future_dataframe(periods=periods, freq="D")
    fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    # Clip negatives — can't sell negative units
    fc["yhat"] = fc["yhat"].clip(lower=0)
    fc["yhat_lower"] = fc["yhat_lower"].clip(lower=0)
    fc["yhat_upper"] = fc["yhat_upper"].clip(lower=0)

    fc["product_id"] = product_id
    return fc


def forecast_top_products(
    weekly_prod: pd.DataFrame,
    product_ids: list[str],
    periods: int = 30,
) -> pd.DataFrame:
    """
    Run forecast_one_product for each product_id and concatenate results.
    Skips products with insufficient history silently.
    """
    all_fc: list[pd.DataFrame] = []
    for pid in product_ids:
        fc = forecast_one_product(weekly_prod, pid, periods=periods)
        if not fc.empty:
            all_fc.append(fc)

    if not all_fc:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper", "product_id"])

    return pd.concat(all_fc, ignore_index=True)
