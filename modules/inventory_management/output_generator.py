"""output_generator.py — Inventory & discount pipeline
Reads:
  - data/outputs/sales_forecast.csv     (from sales forecasting module)
  - data/raw/.../online_retail_II.xlsx  (for stock generation if needed)
Writes:
  - data/outputs/inventory_recommendations.csv
  - data/outputs/discount_recommendations.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.sales_forecasting.forecast_model import load_uci_excel
from modules.inventory_management.inventory_logic import (
    classify_abc,
    inventory_and_discount_logic,
)
from modules.inventory_management.stock_generation import generate_stock_from_sales

DEFAULT_FORECAST  = ROOT_DIR / "data" / "outputs" / "sales_forecast.csv"
DEFAULT_STOCK     = ROOT_DIR / "data" / "raw" / "inventory" / "current_stock.csv"
DEFAULT_SALES     = ROOT_DIR / "data" / "raw" / "sales_forecasting" / "online_retail_II.xlsx"
INV_OUT           = ROOT_DIR / "data" / "outputs" / "inventory_recommendations.csv"
DISC_OUT          = ROOT_DIR / "data" / "outputs" / "discount_recommendations.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--forecast_csv",  default=str(DEFAULT_FORECAST))
    p.add_argument("--stock_csv",     default="",
                   help="current_stock.csv — auto-generated if missing")
    p.add_argument("--sales_excel",   default=str(DEFAULT_SALES),
                   help="online_retail_II.xlsx — used only for stock generation")
    p.add_argument("--out_inventory", default=str(INV_OUT))
    p.add_argument("--out_discount",  default=str(DISC_OUT))
    p.add_argument("--horizon_days",  type=int, default=7)
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Load forecast ──
    print("[inventory] Loading forecast…")
    forecast = pd.read_csv(args.forecast_csv)
    required_cols = {"date", "product_id", "predicted_sales", "predicted_low", "predicted_high"}
    missing = required_cols - set(forecast.columns)
    if missing:
        raise ValueError(f"sales_forecast.csv missing columns: {missing}")
    print(f"[inventory] Forecast rows: {len(forecast):,}  |  Products: {forecast['product_id'].nunique()}")

    # ── 2. ABC classification (from forecast revenue) ──
    print("[inventory] Classifying products A/B/C…")
    abc_df = classify_abc(forecast)
    print(abc_df["abc_class"].value_counts().to_string())

    # ── 3. Load or generate stock ──
    stock_path = Path(args.stock_csv) if args.stock_csv else DEFAULT_STOCK
    if not stock_path.exists():
        print(f"[inventory] Stock file not found — generating from {args.sales_excel}…")
        df_uci = load_uci_excel(args.sales_excel)
        generate_stock_from_sales(df_uci, stock_path, top_n=forecast["product_id"].nunique())
    stock = pd.read_csv(stock_path)
    print(f"[inventory] Stock rows: {len(stock)}")

    # ── 4. Run logic ──
    print("[inventory] Computing recommendations…")
    inv_df, disc_df = inventory_and_discount_logic(
        forecast_df=forecast,
        stock_df=stock,
        abc_df=abc_df,
        horizon_days=args.horizon_days,
    )

    # ── 5. Save ──
    Path(args.out_inventory).parent.mkdir(parents=True, exist_ok=True)
    inv_df.to_csv(args.out_inventory,  index=False)
    disc_df.to_csv(args.out_discount, index=False)

    print(f"\n[inventory] Saved → {args.out_inventory}")
    print(f"[inventory] Saved → {args.out_discount}")

    print("\n── Inventory summary ──")
    print(inv_df["decision"].value_counts().to_string())
    print("\n── By ABC class ──")
    print(inv_df.groupby(["abc_class", "decision"]).size().to_string())
    print("\n── Discount summary ──")
    print(disc_df["discount_action"].value_counts().to_string())
    print("\nSample inventory rows:")
    print(inv_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
