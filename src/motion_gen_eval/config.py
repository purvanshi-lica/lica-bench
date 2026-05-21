from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass
class TrackingConfig:
    """All tunables for the OBB motion-tracking pipeline.

    Note: temporal multi-object trackers (ByteTrack / BoT-SORT) have been
    removed. The ``yolo`` mode now performs per-frame YOLO detection with
    spatial polygon-IoU matching against the known layout polygons; the
    ``layout-init`` and ``contour`` modes are unchanged.
    """

    # -- model --
    weights: str = os.getenv(
        "OBB_MODEL_PATH", "yolo11x-obb.pt"
    )
    device: str = ""  # "" lets ultralytics auto-select; "cuda:0", "cpu", etc.

    # -- detection --
    imgsz: int = 1280
    conf: float = 0.01
    iou: float = 0.6
    max_det: int = 300
    target_classes: Optional[List[int]] = None  # None = keep all classes

    # -- output --
    output_json: bool = True
    output_dir: str = "output"

    # -- mode --
    mode: str = "layout-init"  # "layout-init", "yolo", or "contour"

    # -- layout / component tracking --
    layout_json: Optional[str] = None  # path to LICA layout JSON
    track_components: Optional[List[str]] = None  # e.g. ["IMAGE 0-1", "TEXT 0-5"]
    match_iou_thresh: float = 0.03  # min IoU to match a detection to a layout component
    cv_tracker_type: str = "DaSiamRPN"  # OpenCV tracker (layout-init mode)

    # -- contour mode (objects on uniform background) --
    bg_thresholds: Optional[List[int]] = None  # None = auto-discover from histogram
    morph_kernel: int = 5  # morphological cleanup kernel size (0 = off)

    # -- misc --
    verbose: bool = False
    start_frame: int = 0
    max_frames: int = 0  # 0 = process all

    @property
    def weights_path(self) -> Path:
        return Path(self.weights).resolve()

    @property
    def output_path(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
