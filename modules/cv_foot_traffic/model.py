"""CV Foot Traffic - YOLO model wrapper.

This module keeps YOLO loading in ONE place and avoids the common "train2" bug.

Priority for weights:
1) Env var AI_RETAIL_CV_MODEL (absolute or relative to repo root)
2) Fixed export path (recommended): modules/cv_foot_traffic/weights/best.pt
3) Newest trained run: modules/cv_foot_traffic/runs/detect/train*/weights/best.pt
4) Fallback: modules/cv_foot_traffic/yolo26n.pt
5) Fallback: modules/cv_foot_traffic/yolov8n.pt
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent


def _newest_train_best() -> Path | None:
    runs_dir = MODULE_DIR / "runs" / "detect"
    if not runs_dir.exists():
        return None
    candidates = []
    for p in runs_dir.glob("train*"):
        best = p / "weights" / "best.pt"
        if best.exists():
            candidates.append(best)
    if not candidates:
        return None
    # newest by modified time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_model_path() -> Path:
    env = os.getenv("AI_RETAIL_CV_MODEL", "").strip()
    candidates: List[Path] = []

    if env:
        p = Path(env)
        candidates.append(p if p.is_absolute() else (ROOT_DIR / p))

    candidates += [
        MODULE_DIR / "weights" / "best.pt",
    ]

    newest = None
    if newest is not None:
        candidates.append(newest)

    candidates += [
        MODULE_DIR / "best.pt",
        MODULE_DIR / "yolov8n.pt",
        MODULE_DIR / "yolo26n.pt",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find CV model weights. Tried: "
        + ", ".join(str(p) for p in candidates)
        + ".\nTip: export your trained best.pt to modules/cv_foot_traffic/weights/best.pt"
    )


MODEL_PATH = _resolve_model_path()
print("CV MODEL WEIGHTS:", MODEL_PATH)
model = YOLO(str(MODEL_PATH))


def get_boxes(image_path: str | Path) -> List[Tuple[float, float, float, float]]:
    """Return list of (x1,y1,x2,y2) boxes."""
    results = model(str(image_path), verbose=False)
    if not results or results[0].boxes is None:
        return []
    xyxy = results[0].boxes.xyxy
    if xyxy is None:
        return []
    return [tuple(map(float, b)) for b in xyxy.cpu().numpy().tolist()]


def count_people(image_path: str | Path) -> int:
    return len(get_boxes(image_path))
