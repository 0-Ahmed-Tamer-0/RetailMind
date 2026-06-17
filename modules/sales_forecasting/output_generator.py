"""output_generator.py — Sales forecasting pipeline
Reads UCI Online Retail II Excel file (both sheets).
"""

from __future__ import annotations

from pandas import to_datetime
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.sales_forecasting.forecast_model import (
    load_uci_excel,
    build_top_products,
    make_weekly_product_sales,
    forecast_top_products,
)

DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "sales_forecasting" / "online_retail_II.xlsx"
DEFAULT_OUT_PATH  = ROOT_DIR / "data" / "outputs" / "sales_forecast.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Generate sales forecast (top-N products).")
    p.add_argument("--input_excel", default=str(DEFAULT_DATA_PATH),
                   help="Path to online_retail_II.xlsx")
    p.add_argument("--out_csv",     default=str(DEFAULT_OUT_PATH),
                   help="Output path for sales_forecast.csv")
    p.add_argument("--top_n",   type=int, default=10,
                   help="Number of top products to forecast (default 10)")
    p.add_argument("--periods", type=int, default=30,
                   help="Forecast horizon in days (default 30)")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out_csv)

    # ── 1. Load + clean (both sheets, cancellations removed) ──
    print("[forecast] Loading UCI data…")
    df = load_uci_excel(args.input_excel)
    print(f"[forecast] Clean rows: {len(df):,}")

    # ── 2. Top-N products by revenue ──
    top_n = build_top_products(df, top_n=args.top_n)
    print(f"[forecast] Top-{args.top_n} products: {top_n}")

    # ── 3. Daily sales per product ──
    df_top     = df[df["StockCode"].astype(str).isin(top_n)].copy()
    weekly_prod = make_weekly_product_sales(df_top)

    # ── 4. Forecast ──
    print("[forecast] Fitting Prophet models…")
    fc = forecast_top_products(weekly_prod, top_n, periods=args.periods)
    if fc.empty:
        raise RuntimeError("No forecasts generated — check data history length.")

    # ── 5. Merge actuals + forecast, keep confidence intervals ──
    fc["date"] =  to_datetime(fc["ds"])
    fc = fc.drop(columns=["ds"])

    merged = weekly_prod.merge(fc, on=["date", "product_id"], how="left")

    out = merged.rename(columns={
        "sales":       "actual_sales",
        "yhat":        "predicted_sales",
        "yhat_lower":  "predicted_low",
        "yhat_upper":  "predicted_high",
    })[["date", "product_id", "actual_sales", "predicted_sales", "predicted_low", "predicted_high"]]

    # ── 6. Save ──
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[forecast] Saved → {out_path}")
    print(out.head())


if __name__ == "__main__":
    main()
