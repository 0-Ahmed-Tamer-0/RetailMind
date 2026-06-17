"""Generate CV outputs (counts + zone overlays).

Outputs:
- data/outputs/cv_foot_traffic.csv with schema:
    timestamp, zone_id, people_count

- Overlay PNGs (ONLY overlays, no standalone heatmaps):
    data/outputs/cv_zone_overlays/zone_<zoneid>_overlay.png

Notes:
- For normal zone folders: count people per image.
- For wide zone folder "41": split into 3 equal vertical zones (A/B/C) without cutting files.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
except Exception:
    cv2 = None

from model import get_boxes


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_BASE = Path(os.getenv("AI_RETAIL_DATA_DIR", str(ROOT_DIR / "data" / "raw")))

DEFAULT_BASE_DIR = RAW_BASE / "cv_foot_traffic"
DEFAULT_OUT_CSV = ROOT_DIR / "data" / "outputs" / "cv_foot_traffic.csv"
DEFAULT_OVERLAY_DIR = ROOT_DIR / "data" / "outputs" / "cv_zone_overlays"


def parse_args():
    p = argparse.ArgumentParser(description="Generate CV foot traffic output + per-zone overlay images.")
    p.add_argument("--input_dir", type=str, default=str(DEFAULT_BASE_DIR), help="Folder containing zone folders (1/2/41).")
    p.add_argument("--out_csv", type=str, default=str(DEFAULT_OUT_CSV), help="Output cv_foot_traffic.csv")
    p.add_argument("--overlay_dir", type=str, default=str(DEFAULT_OVERLAY_DIR), help="Directory to save overlay PNGs")
    p.add_argument("--seed", type=int, default=7, help="Random seed for choosing overlay frame")
    return p.parse_args()


def _is_image(name: str) -> bool:
    n = name.lower()
    return n.endswith(".png") or n.endswith(".jpg") or n.endswith(".jpeg")


def _is_wide_zone_folder(folder_name: str) -> bool:
    name = folder_name.strip().lower()
    return name == "41" or "wide" in name


def _split_3_equal_zones(w: int):
    a = (0, int(w / 3))
    b = (int(w / 3), int(2 * w / 3))
    c = (int(2 * w / 3), w)
    return {"A": a, "B": b, "C": c}


def _box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _safe_read_image(path: Path):
    if cv2 is None:
        return None
    return cv2.imread(str(path))


def _make_overlay(points_xy, base_img):
    """Return overlay image (BGR) blending heatmap on base image."""
    if cv2 is None or base_img is None or not points_xy:
        return None

    h, w = base_img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for (x, y) in points_xy:
        xi, yi = int(x), int(y)
        if 0 <= xi < w and 0 <= yi < h:
            heat[yi, xi] += 1.0

    heat = cv2.GaussianBlur(heat, (51, 51), 0)
    norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_img, 0.6, colored, 0.4, 0)
    return overlay


def main():
    args = parse_args()
    random.seed(args.seed)

    base_dir = Path(args.input_dir)
    out_csv = Path(args.out_csv)
    overlay_dir = Path(args.overlay_dir)

    rows = []
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for zone_folder in sorted(os.listdir(base_dir)):
        zone_path = base_dir / zone_folder
        if not zone_path.is_dir():
            continue

        images_dir = zone_path / "images"
        if not images_dir.exists():
            continue

        img_files = sorted([p for p in images_dir.iterdir() if p.is_file() and _is_image(p.name)])
        if not img_files:
            continue

        if _is_wide_zone_folder(zone_folder):
            # split each image detections into A/B/C zones
            sample_img = _safe_read_image(img_files[0])
            if sample_img is None:
                continue
            h, w = sample_img.shape[:2]
            splits = _split_3_equal_zones(w)

            points_by_zone = {f"{zone_folder}_{k}": [] for k in splits.keys()}
            frames_by_zone = {f"{zone_folder}_{k}": [] for k in splits.keys()}

            for img_path in img_files:
                img = _safe_read_image(img_path)
                if img is None:
                    continue
                boxes = get_boxes(img_path)

                # group boxes by split
                for box in boxes:
                    cx, cy = _box_center(box)
                    for letter, (xL, xR) in splits.items():
                        if xL <= cx < xR:
                            z = f"{zone_folder}_{letter}"
                            points_by_zone[z].append((cx, cy))
                            break

                # counts per split for this frame
                counts = {f"{zone_folder}_{letter}": 0 for letter in splits.keys()}
                for box in boxes:
                    cx, cy = _box_center(box)
                    for letter, (xL, xR) in splits.items():
                        if xL <= cx < xR:
                            counts[f"{zone_folder}_{letter}"] += 1
                            break

                for z, c in counts.items():
                    rows.append({"timestamp": img_path.name, "zone_id": z, "people_count": int(c)})
                    frames_by_zone[z].append(img_path)
                    
                    
            print({k: len(v) for k, v in points_by_zone.items()})
            
            # overlays per split (CROP OUTPUT, keep full coords)
            for z in points_by_zone:
                print("DEBUG z", z, "points", len(points_by_zone[z]), "frames", len(frames_by_zone[z]))
                if points_by_zone[z] and frames_by_zone[z]:
                    base_img_path = random.choice(frames_by_zone[z])
                    base_img = _safe_read_image(base_img_path)
                    print("DEBUG base_img", base_img_path, "ok" if base_img is not None else "NONE")
                    if base_img is None:
                        continue
                    
                    overlay_full = _make_overlay(points_by_zone[z], base_img)
                    print("DEBUG overlay_full", "ok" if overlay_full is not None else "NONE")
                    if overlay_full is None:
                        continue
                    
                    letter = z.split("_")[-1]
                    xL, xR = splits[letter]
                    overlay_crop = overlay_full[:, xL:xR]
                    print("DEBUG crop shape", overlay_crop.shape)

                    out_img = overlay_dir / f"zone_{z}_overlay.png"
                    ok = cv2.imwrite(str(out_img), overlay_crop)
                    print("DEBUG wrote", out_img, ok)




        else:
            # normal zone: whole image
            zone_id = zone_folder
            points = []
            for img_path in img_files:
                boxes = get_boxes(img_path)
                # if zone_folder == "2":
                #     print("DEBUG Z2", img_path.name, "boxes", len(boxes))
                rows.append({"timestamp": img_path.name, "zone_id": zone_id, "people_count": int(len(boxes))})
                for box in boxes:
                    cx, cy = _box_center(box)
                    points.append((cx, cy))

            # overlay for this zone
            if points:
                base_img_path = random.choice(img_files)
                base_img = _safe_read_image(base_img_path)
                overlay = _make_overlay(points, base_img)
                if overlay is not None:
                    out_img = overlay_dir / f"zone_{zone_id}_overlay.png"
                    cv2.imwrite(str(out_img), overlay)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("Saved:", str(out_csv))
    print("Saved overlays to:", str(overlay_dir))
    print(df.head())


if __name__ == "__main__":
    main()
