"""
yolo_utils.py
=============
Framework-agnostic helpers for the YOLO Vision Lab dashboard.

Kept deliberately free of Streamlit / PyAV imports so it stays easy to read,
reuse and unit-test. Everything here is plain Python + numpy/pandas +
ultralytics. The Streamlit UI lives in `app.py`.

To add a new model, just add a line to MODEL_REGISTRY below — the dropdowns
update automatically.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Friendly display name  ->  ultralytics weights file (auto-downloaded on first
# use). Grouped by task. Add / remove lines freely; the UI adapts.
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "Detection": {
        "YOLOv8n  ·  nano (fastest)": "yolov8n.pt",
        "YOLOv8s  ·  small": "yolov8s.pt",
        "YOLOv8m  ·  medium": "yolov8m.pt",
        "YOLOv8l  ·  large": "yolov8l.pt",
        "YOLOv9c  ·  compact": "yolov9c.pt",
        "YOLOv10n · nano": "yolov10n.pt",
        "YOLOv10s · small": "yolov10s.pt",
        "YOLO11n  ·  nano (newest)": "yolo11n.pt",
        "YOLO11s  ·  small": "yolo11s.pt",
        "YOLO11m  ·  medium": "yolo11m.pt",
    },
    "Segmentation": {
        "YOLOv8n-seg · nano": "yolov8n-seg.pt",
        "YOLO11n-seg · nano": "yolo11n-seg.pt",
    },
    "Pose": {
        "YOLOv8n-pose · nano": "yolov8n-pose.pt",
        "YOLO11n-pose · nano": "yolo11n-pose.pt",
    },
}


def models_for_task(task: str) -> dict[str, str]:
    """Return {display name: weights} for a given task group."""
    return MODEL_REGISTRY.get(task, {})


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def available_devices() -> list[str]:
    """Human-readable device choices based on what torch can see."""
    choices = ["Auto", "CPU"]
    try:
        import torch

        if torch.cuda.is_available():
            choices.append("GPU (CUDA)")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            choices.append("Apple GPU (MPS)")
    except Exception:
        pass
    return choices


def auto_device() -> str | int:
    """Pick the best available device for ultralytics."""
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_device(choice: str) -> str | int:
    """Map a UI label to an ultralytics device argument."""
    return {
        "Auto": auto_device(),
        "CPU": "cpu",
        "GPU (CUDA)": 0,
        "Apple GPU (MPS)": "mps",
    }.get(choice, "cpu")


# ---------------------------------------------------------------------------
# Inference + result parsing
# ---------------------------------------------------------------------------
def run_inference(
    model,
    image_bgr: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    classes: list[int] | None = None,
    device: str | int = "cpu",
    track: bool = False,
    persist: bool = True,
):
    """
    Run one forward pass and time it.

    Parameters
    ----------
    track : bool
        If True, run the ByteTrack tracker (`model.track`) so each object keeps a
        stable ID across frames. Only meaningful for a continuous stream/video.
    persist : bool
        Passed to `model.track`. Use False on the first frame of a new stream to
        reset the tracker, then True for every following frame.

    Returns
    -------
    result : ultralytics Results object (single image)
    latency_ms : float — inference wall-clock time in milliseconds
    """
    t0 = time.perf_counter()
    if track:
        results = model.track(
            image_bgr,
            persist=persist,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes if classes else None,
            device=device,
            verbose=False,
        )
    else:
        results = model.predict(
            image_bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes if classes else None,
            device=device,
            verbose=False,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return results[0], latency_ms


def detections_dataframe(result, names: dict[int, str]) -> pd.DataFrame:
    """Turn a Results object's boxes into a tidy DataFrame.

    Includes a leading ``id`` column when the result carries tracker IDs
    (i.e. it came from ``model.track``); otherwise that column is omitted.
    """
    cols = ["id", "class", "confidence", "x1", "y1", "x2", "y2"]
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return pd.DataFrame(columns=cols).drop(columns=["id"])

    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    rows = []
    for k, (c, cf, (x1, y1, x2, y2)) in enumerate(zip(cls, conf, xyxy)):
        rows.append(
            {
                "id": int(ids[k]) if ids is not None else None,
                "class": names.get(int(c), str(c)),
                "confidence": round(float(cf), 3),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }
        )
    df = pd.DataFrame(rows, columns=cols)
    # Only show the id column when we're actually tracking.
    if df["id"].isna().all():
        df = df.drop(columns=["id"])
    return df


def unique_track_ids(result) -> set[int]:
    """Return the set of tracker IDs present in a result (empty if not tracking)."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "id", None) is None:
        return set()
    return {int(i) for i in boxes.id.cpu().numpy().astype(int)}


def summarize(df: pd.DataFrame, latency_ms: float) -> dict:
    """Compact per-frame statistics used to drive metric cards and charts."""
    n = int(len(df))
    return {
        "objects": n,
        "classes": int(df["class"].nunique()) if n else 0,
        "latency_ms": float(latency_ms),
        "fps": (1000.0 / latency_ms) if latency_ms > 0 else 0.0,
        "avg_confidence": float(df["confidence"].mean()) if n else 0.0,
        "class_counts": Counter(df["class"]) if n else Counter(),
    }
