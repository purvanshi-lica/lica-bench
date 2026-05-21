"""Layout-initialized motion tracker.

Instead of relying on YOLO to detect components (which requires a model
trained on this specific content), this module uses the layout JSON to know
exactly where each component is, then tracks the motion across frames using
OpenCV trackers or optical flow.

This is the right approach when:
- You know where components start (from layout metadata)
- The YOLO model wasn't trained on this type of content
- You want to track arbitrary design elements (text, images, shapes)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from motion_gen_eval.config import TrackingConfig
from motion_gen_eval.layout import Layout, LayoutComponent, parse_layout
from motion_gen_eval.video_io import get_video_info, iterate_frames, resolve_video_source


def _obb_to_aabb(polygon: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert a 4-corner polygon to an axis-aligned bounding box (x, y, w, h).

    Clips to non-negative values for OpenCV tracker init.
    """
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    x = max(0, int(xs.min()))
    y = max(0, int(ys.min()))
    w = max(1, int(xs.max()) - x)
    h = max(1, int(ys.max()) - y)
    return x, y, w, h


def _clip_bbox(bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def _bbox_to_polygon(x: int, y: int, w: int, h: int) -> List[float]:
    """Convert AABB to 8-float polygon for output consistency."""
    return [
        float(x), float(y),
        float(x + w), float(y),
        float(x + w), float(y + h),
        float(x), float(y + h),
    ]


def _create_tracker(tracker_type: str = "DaSiamRPN") -> cv2.Tracker:
    """Create an OpenCV tracker by name.

    Available trackers depend on your OpenCV build:
      - DaSiamRPN: Siamese network, accurate, handles scale/rotation (default)
      - Nano: lightweight CNN tracker
      - Vit: Vision Transformer tracker (may need model download)
      - MIL: basic, no deep learning needed
      - CSRT/KCF: require opencv-contrib-python
    """
    name = tracker_type.upper()
    factories = {}

    for attr, key in [
        ("TrackerDaSiamRPN_create", "DASIAMRPN"),
        ("TrackerNano_create", "NANO"),
        ("TrackerVit_create", "VIT"),
        ("TrackerMIL_create", "MIL"),
        ("TrackerGOTURN_create", "GOTURN"),
        ("TrackerCSRT_create", "CSRT"),
        ("TrackerKCF_create", "KCF"),
    ]:
        fn = getattr(cv2, attr, None)
        if fn is not None:
            factories[key] = fn

    # Also handle class-style API (OpenCV 4.5.1+)
    for cls_name, key in [
        ("TrackerDaSiamRPN", "DASIAMRPN"),
        ("TrackerNano", "NANO"),
        ("TrackerVit", "VIT"),
        ("TrackerMIL", "MIL"),
        ("TrackerGOTURN", "GOTURN"),
        ("TrackerCSRT", "CSRT"),
        ("TrackerKCF", "KCF"),
    ]:
        if key not in factories:
            cls = getattr(cv2, cls_name, None)
            if cls is not None and callable(getattr(cls, "create", None)):
                factories[key] = cls.create

    if name not in factories:
        raise ValueError(
            f"Tracker {tracker_type!r} not available. "
            f"Installed options: {sorted(factories.keys())}"
        )
    return factories[name]()


class ComponentTracker:
    """Tracks a single layout component across video frames."""

    def __init__(self, component: LayoutComponent, tracker_type: str = "CSRT"):
        self.component = component
        self.tracker_type = tracker_type
        self.tracker: Optional[cv2.Tracker] = None
        self.initialized = False
        self.lost = False
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        self.history: List[Dict[str, Any]] = []

    def init(self, frame: np.ndarray, bbox_override: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """Initialize tracker on a frame.

        Uses layout polygon to compute a canvas-clipped bounding box, or
        accepts an explicit bbox_override (x, y, w, h).
        """
        fh, fw = frame.shape[:2]

        if bbox_override is not None:
            bbox = _clip_bbox(bbox_override, fw, fh)
        else:
            poly = self.component.polygon()
            # Compute the visible portion: intersect OBB's AABB with the canvas
            raw = _obb_to_aabb(poly)
            bbox = _clip_bbox(raw, fw, fh)

        x, y, bw, bh = bbox
        # Ensure bbox doesn't exceed frame
        bw = min(bw, fw - x)
        bh = min(bh, fh - y)

        if bw < 10 or bh < 10:
            print(f"[layout-track] {self.component.id}: bbox too small ({bw}x{bh}), skipping")
            self.lost = True
            return False

        print(f"[layout-track] {self.component.id}: attempting init at ({x}, {y}, {bw}x{bh})")

        self.tracker = _create_tracker(self.tracker_type)
        try:
            result = self.tracker.init(frame, (x, y, bw, bh))
            # OpenCV 4.x: init() returns None on success (void method)
            # Older versions returned bool. Treat non-exception as success.
            self.initialized = True
            self.last_bbox = (x, y, bw, bh)
            print(f"[layout-track] {self.component.id}: initialized OK")
            return True
        except cv2.error as e:
            print(f"[layout-track] {self.component.id}: OpenCV error: {e}")
            self.lost = True
            return False

    def update(self, frame: np.ndarray, frame_idx: int) -> Optional[Dict[str, Any]]:
        """Update tracker with a new frame. Returns detection dict or None if lost."""
        if not self.initialized or self.lost or self.tracker is None:
            return None

        ok, bbox = self.tracker.update(frame)
        if not ok:
            self.lost = True
            return None

        x, y, w, h = [int(v) for v in bbox]
        self.last_bbox = (x, y, w, h)

        det = {
            "track_id": hash(self.component.id) % 10000,
            "component_id": self.component.id,
            "class_id": -1,
            "confidence": 1.0,
            "polygon": _bbox_to_polygon(x, y, w, h),
            "bbox": [x, y, w, h],
        }
        self.history.append({"frame_idx": frame_idx, "detection": det})
        return det


class LayoutInitTracker:
    """Track layout components through video using OpenCV trackers
    initialized from layout JSON positions.

    No YOLO model needed -- the layout metadata tells us where
    each component is, and OpenCV trackers follow the motion.
    """

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.cv_tracker_type = cfg.cv_tracker_type if hasattr(cfg, "cv_tracker_type") else "CSRT"

    def track_video(self, source: str) -> Dict[str, Any]:
        if not self.cfg.layout_json:
            raise ValueError("Layout JSON is required for layout-init tracking mode.")
        if not self.cfg.track_components:
            raise ValueError("--track component specs are required for layout-init mode.")

        layout = parse_layout(self.cfg.layout_json)
        print(f"[layout-track] Layout: {layout.width:.0f}x{layout.height:.0f}, "
              f"duration={layout.duration}s")

        video_path = resolve_video_source(source)
        info = get_video_info(video_path)
        fps = info["fps"] or 30.0
        v_w = float(info.get("width") or 0)
        v_h = float(info.get("height") or 0)
        if v_w and v_h and (
            abs(v_w - layout.width) > 1.0 or abs(v_h - layout.height) > 1.0
        ):
            print(
                f"[layout-track] Scaling layout {layout.width:.0f}x{layout.height:.0f}"
                f" -> video {v_w:.0f}x{v_h:.0f}"
            )
            layout = layout.scaled_to(v_w, v_h)

        targets = self._resolve_targets(layout)
        print(f"[layout-track] Tracking {len(targets)} component(s):")
        for t in targets:
            print(f"  {t}")

        comp_trackers = [
            ComponentTracker(t, self.cv_tracker_type) for t in targets
        ]

        all_results: List[dict] = []
        t0 = time.perf_counter()

        for frame_idx, bgr_frame in iterate_frames(
            video_path,
            start_frame=self.cfg.start_frame,
            max_frames=self.cfg.max_frames,
        ):
            if frame_idx == self.cfg.start_frame:
                for ct in comp_trackers:
                    ct.init(bgr_frame)

            detections = []
            for ct in comp_trackers:
                det = ct.update(bgr_frame, frame_idx)
                if det is not None:
                    detections.append(det)

            ts_ms = (frame_idx / fps) * 1000.0
            frame_result = {
                "frame_idx": frame_idx,
                "timestamp_ms": round(ts_ms, 2),
                "num_detections": len(detections),
                "detections": detections,
            }
            all_results.append(frame_result)

            if self.cfg.verbose and frame_idx % 50 == 0:
                elapsed = time.perf_counter() - t0
                n = frame_idx - self.cfg.start_frame + 1
                active = sum(1 for ct in comp_trackers if not ct.lost)
                print(
                    f"[layout-track] frame {frame_idx} | "
                    f"{len(detections)}/{len(comp_trackers)} active | "
                    f"{n / max(elapsed, 1e-6):.1f} fps"
                )

        elapsed = time.perf_counter() - t0

        component_tracks = {}
        for ct in comp_trackers:
            component_tracks[ct.component.id] = {
                "type": ct.component.type,
                "initialized": ct.initialized,
                "lost": ct.lost,
                "frames_tracked": len(ct.history),
                "trajectory": ct.history,
            }

        summary: Dict[str, Any] = {
            "mode": "layout-init",
            "video_source": str(source),
            "video_info": info,
            "total_frames_processed": len(all_results),
            "processing_time_s": round(elapsed, 2),
            "avg_fps": round(len(all_results) / max(elapsed, 1e-6), 2),
            "component_tracking": {
                "requested": [f"{t.type} {t.id}" for t in targets],
                "tracker_type": self.cv_tracker_type,
                "per_component": component_tracks,
            },
            "frames": all_results,
        }

        if self.cfg.output_json:
            json_path = self.cfg.output_path / (Path(video_path).stem + "_tracks.json")
            json_path.write_text(json.dumps(summary, indent=2))
            summary["json_output"] = str(json_path)
            print(f"[layout-track] JSON -> {json_path}")

        return summary

    def _resolve_targets(self, layout: Layout) -> List[LayoutComponent]:
        targets = []
        for spec in self.cfg.track_components:
            parts = spec.strip().split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid component spec {spec!r}. Use 'TYPE ID'.")
            comp_type, comp_id = parts[0].upper(), parts[1]
            comp = layout.find(comp_id)
            if comp is None:
                available = [f"{c.type} {c.id}" for c in layout.components]
                raise ValueError(f"Component {comp_id!r} not found. Available: {available}")
            if comp.type.upper() != comp_type:
                raise ValueError(
                    f"Component {comp_id!r} is type {comp.type!r}, not {comp_type!r}."
                )
            targets.append(comp)
        return targets
