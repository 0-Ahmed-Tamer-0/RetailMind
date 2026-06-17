"""output_generator.py — Customer segmentation pipeline
Reads the same UCI Online Retail II Excel file as the forecasting module.
No separate customer CSV needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from module directory directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in str(sys.path):
    sys.path.insert(0, str(ROOT_DIR))

from modules.sales_forecasting.forecast_model import clean_uci, load_uci_excel
from modules.customer_segmentation.segmentation_model import (
    compute_rfm,
    run_kmeans,
    summarize_clusters,
)

DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "sales_forecasting" / "online_retail_II.xlsx"
DEFAULT_OUT_SUMMARY = ROOT_DIR / "data" / "outputs" / "customer_segments.csv"
DEFAULT_OUT_FULL    = ROOT_DIR / "data" / "outputs" / "customer_rfm_full.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Generate customer segments from UCI Online Retail II.")
    p.add_argument("--input_excel", default=str(DEFAULT_DATA_PATH),
                   help="Path to online_retail_II.xlsx")
    p.add_argument("--out_summary", default=str(DEFAULT_OUT_SUMMARY),
                   help="Output path for segment summary CSV")
    p.add_argument("--out_full", default=str(DEFAULT_OUT_FULL),
                   help="Output path for per-customer RFM + cluster CSV")
    p.add_argument("--k", type=int, default=None,
                   help="Number of clusters (default: auto via elbow method)")
    return p.parse_args()


def main():
    args = parse_args()

    print("[segmentation] Loading UCI data…")
    df = load_uci_excel(args.input_excel)
    print(f"[segmentation] Clean rows: {len(df):,}  |  Unique customers: {df['Customer ID'].nunique():,}")

    print("[segmentation] Computing RFM…")
    rfm = compute_rfm(df)

    print("[segmentation] Clustering…")
    rfm_clustered = run_kmeans(rfm, k=args.k)

    summary = summarize_clusters(rfm_clustered)

    # Save outputs
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)
    rfm_clustered.to_csv(args.out_full, index=False)

    print(f"[segmentation] Saved summary  → {args.out_summary}")
    print(f"[segmentation] Saved full RFM → {args.out_full}")
    print("\nSegment summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
