"""Per-frame YOLO-OBB detection with spatial IoU matching to layout components.

This module replaces the previous ByteTrack-based temporal tracking. For
animation videos every component has a known ground-truth position in the
layout JSON, so we don't need to associate detections across frames with a
multi-object tracker -- we just run YOLO on every frame and match each
frame's detections to the layout components by polygon IoU (Hungarian
assignment).

Why per-frame matching beats ByteTrack here:
  * No identity drift / track fragmentation when components briefly leave
    the frame or briefly overlap.
  * No tracker confidence threshold suppressing valid low-confidence
    detections from initiating a stable track.
  * Detection at frame ``t`` is always labelled with the correct layout
    component_id, so ``motion_metrics._select_track`` finds the trajectory
    immediately.

The output dict mirrors the schema produced by the old ``OBBTracker`` (so
``motion_metrics.evaluate_sample`` works without changes), with fields:

  - ``frames``: list of {frame_idx, timestamp_ms, detections}
  - ``component_tracking.per_component``: {comp_id: [{frame_idx, detection}]}
  - ``component_tracking.matched`` / ``unmatched``
  - ``video_info`` and a ``config`` block describing the run
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import cv2
import numpy as np

from motion_gen_eval.layout import (
    Layout,
    LayoutComponent,
    match_detections_to_components,
    parse_layout,
)
from motion_gen_eval.video_io import get_video_info, resolve_video_source


YOLO_DETECTABLE_TYPES: Set[str] = {"IMAGE", "TEXT"}


def find_animated_components(layout_path: str | Path) -> List[Dict[str, str]]:
    """Walk a LICA layout JSON and produce one entry per animated component.

    Each entry is ``{"id": eval_id, "detect_id": child_id, "type": TYPE}``.

    * ``id`` is what motion_metrics expects (and may be a GROUP wrapper).
    * ``detect_id`` is the child IMAGE/TEXT component that YOLO actually
      sees (when the animation lives on a GROUP). When the animated
      component is itself an IMAGE/TEXT, ``detect_id == id``.

    Static IMAGE/TEXT layout components are also appended (with
    ``id == detect_id``) so spatial matching has every detectable target
    available, which improves disambiguation for the animated ones.
    """
    import json as _json

    layout_path = Path(layout_path)
    data = _json.loads(layout_path.read_text())
    lc = data.get("layout_config", data)
    layout = parse_layout(str(layout_path))

    def _walk_animated(node: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if isinstance(node, dict):
            if node.get("animations"):
                out.append(node)
            for v in node.values():
                out.extend(_walk_animated(v))
        elif isinstance(node, list):
            for v in node:
                out.extend(_walk_animated(v))
        return out

    def _find_detectable_child(node: Any) -> Optional[Dict[str, Any]]:
        if isinstance(node, dict):
            if (node.get("type") or "").upper() in YOLO_DETECTABLE_TYPES:
                return node
            for child in node.get("components", []) or []:
                hit = _find_detectable_child(child)
                if hit is not None:
                    return hit
        return None

    components: List[Dict[str, str]] = []
    seen_ids: Set[str] = set()

    for anim in _walk_animated(lc):
        anim_id = anim.get("id")
        if not anim_id or anim_id in seen_ids:
            continue
        seen_ids.add(anim_id)
        anim_type = (anim.get("type") or "").upper()
        if anim_type in YOLO_DETECTABLE_TYPES:
            components.append({"id": anim_id, "detect_id": anim_id, "type": anim_type})
        else:
            child = _find_detectable_child(anim)
            if child is not None:
                child_id = child.get("id", anim_id)
                child_type = (child.get("type") or "IMAGE").upper()
                components.append({
                    "id": anim_id,
                    "detect_id": child_id,
                    "type": child_type,
                })

    animated_ids = {c["id"] for c in components}
    detect_ids = {c["detect_id"] for c in components}
    for comp in layout.components:
        if (
            comp.type.upper() in YOLO_DETECTABLE_TYPES
            and comp.id not in animated_ids
            and comp.id not in detect_ids
        ):
            components.append({
                "id": comp.id,
                "detect_id": comp.id,
                "type": comp.type.upper(),
            })

    return components


def detect_components_per_frame(
    yolo_model,
    video_path: str | Path,
    layout_path: str | Path,
    components: Sequence[Mapping[str, str]],
    *,
    imgsz: int = 1280,
    conf: float = 0.01,
    nms_iou: float = 0.6,
    match_iou_thresh: float = 0.03,
    start_frame: int = 0,
    max_frames: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run YOLO on every frame and match detections to layout components.

    Parameters
    ----------
    yolo_model
        A loaded ``ultralytics.YOLO`` instance (caller controls device).
    video_path
        Path or URL to the input video.
    layout_path
        Path to the LICA layout JSON. Used to look up each component's
        ground-truth polygon for IoU matching.
    components
        Sequence of mappings, each with at least ``id`` (eval_id) and
        ``type`` keys; an optional ``detect_id`` is the layout component
        that YOLO actually sees (defaults to ``id``).
    imgsz, conf, nms_iou
        Standard YOLO predict knobs. Defaults are deliberately permissive
        (``conf=0.01``) because layout matching gives us a strong spatial
        prior -- a stray low-confidence box that overlaps a known target
        is almost always the right detection.
    match_iou_thresh
        Minimum polygon IoU for a detection-to-component match.
    start_frame, max_frames
        Optional frame slicing.

    Returns
    -------
    dict
        ``tracks_data`` payload compatible with
        ``motion_metrics.evaluate_sample``.
    """
    video_path = resolve_video_source(str(video_path))
    info = get_video_info(video_path)
    fps = info.get("fps") or 30.0
    v_w = float(info.get("width") or 0)
    v_h = float(info.get("height") or 0)

    layout = parse_layout(str(layout_path))
    if v_w and v_h and (
        abs(v_w - layout.width) > 1.0 or abs(v_h - layout.height) > 1.0
    ):
        layout = layout.scaled_to(v_w, v_h)

    targets: List[LayoutComponent] = []
    detect_to_eval: Dict[str, str] = {}
    for comp in components:
        eval_id = comp["id"]
        detect_id = comp.get("detect_id") or eval_id
        c = layout.find(detect_id)
        if c is None:
            if verbose:
                print(f"[frame_detector] WARN: component {detect_id!r} not in layout")
            continue
        targets.append(c)
        detect_to_eval[detect_id] = eval_id

    eval_ids: List[str] = sorted(set(detect_to_eval.values()))
    component_tracks: Dict[str, List[Dict[str, Any]]] = {eid: [] for eid in eval_ids}
    matched_ever: Set[str] = set()
    all_frames: List[Dict[str, Any]] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = total_frames if max_frames <= 0 else min(total_frames, start_frame + max_frames)

    t0 = time.perf_counter()
    try:
        for frame_idx in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break

            results = yolo_model.predict(
                frame,
                imgsz=imgsz,
                conf=conf,
                iou=nms_iou,
                verbose=False,
            )
            r = results[0] if results else None

            det_list: List[Dict[str, Any]] = []
            if (
                r is not None
                and getattr(r, "obb", None) is not None
                and r.obb.data is not None
                and getattr(r.obb, "xyxyxyxy", None) is not None
                and len(r.obb) > 0
            ):
                polys = r.obb.xyxyxyxy.cpu().numpy()
                confs = r.obb.conf.cpu().numpy()
                cls = r.obb.cls.cpu().numpy().astype(int)
                for di in range(len(r.obb)):
                    coords = polys[di].reshape(-1).tolist()
                    det_list.append({
                        "polygon": [round(float(v), 2) for v in coords],
                        "confidence": float(confs[di]),
                        "class_id": int(cls[di]),
                    })

            detections_this_frame: List[Dict[str, Any]] = []
            if det_list and targets:
                matches = match_detections_to_components(
                    det_list, targets, iou_thresh=match_iou_thresh,
                )
                for detect_id, det in matches.items():
                    eval_id = detect_to_eval.get(detect_id, detect_id)
                    matched_ever.add(eval_id)
                    det["component_id"] = eval_id
                    det["track_id"] = abs(hash(eval_id)) % 100000
                    detections_this_frame.append(det)
                    component_tracks[eval_id].append({
                        "frame_idx": frame_idx,
                        "detection": det,
                    })

            ts_ms = (frame_idx / fps) * 1000.0 if fps else 0.0
            all_frames.append({
                "frame_idx": frame_idx,
                "timestamp_ms": round(ts_ms, 2),
                "num_detections": len(detections_this_frame),
                "detections": detections_this_frame,
            })

            if verbose and frame_idx % 50 == 0 and frame_idx > start_frame:
                elapsed = time.perf_counter() - t0
                processed = frame_idx - start_frame + 1
                print(f"[frame_detector] frame {frame_idx} | "
                      f"{len(detections_this_frame)} matched | "
                      f"{processed / max(elapsed, 1e-6):.1f} fps")
    finally:
        cap.release()

    elapsed = time.perf_counter() - t0
    unmatched = [eid for eid in eval_ids if eid not in matched_ever]

    return {
        "video_source": str(video_path),
        "video_info": info,
        "config": {
            "mode": "yolo-per-frame",
            "imgsz": imgsz,
            "conf": conf,
            "nms_iou": nms_iou,
            "match_iou_thresh": match_iou_thresh,
        },
        "total_frames_processed": len(all_frames),
        "unique_tracks": len(matched_ever),
        "processing_time_s": round(elapsed, 2),
        "avg_fps": round(len(all_frames) / max(elapsed, 1e-6), 2),
        "frames": all_frames,
        "component_tracking": {
            "requested": [f"{t.type} {t.id}" for t in targets],
            "matched": {
                cid: {"frames_tracked": len(component_tracks[cid])}
                for cid in matched_ever
            },
            "unmatched": unmatched,
            "per_component": component_tracks,
        },
    }


class YoloFrameDetector:
    """Lazy-loaded YOLO model wrapper that we reuse across many videos.

    Loading the YOLO checkpoint costs several seconds, so the runner is
    persistent across an entire batch run. Use :meth:`run` once per video.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        imgsz: int = 1280,
        conf: float = 0.01,
        nms_iou: float = 0.6,
        match_iou_thresh: float = 0.03,
        device: str = "",
        verbose: bool = False,
    ):
        self.weights = Path(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.nms_iou = nms_iou
        self.match_iou_thresh = match_iou_thresh
        self.device = device
        self.verbose = verbose
        self._model = None
        self._chosen_device: Optional[str] = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        from ultralytics import YOLO

        self._model = YOLO(str(self.weights))
        device = self.device
        if not device:
            try:
                import torch

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        try:
            self._model.to(device)
        except Exception:
            pass
        self._chosen_device = device
        if self.verbose:
            print(f"[frame_detector] YOLO loaded on {device}")

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def detect_video(
        self,
        video_path: str | Path,
        layout_path: str | Path,
        components: Sequence[Mapping[str, str]],
        *,
        start_frame: int = 0,
        max_frames: int = 0,
    ) -> Dict[str, Any]:
        return detect_components_per_frame(
            self.model,
            video_path,
            layout_path,
            components,
            imgsz=self.imgsz,
            conf=self.conf,
            nms_iou=self.nms_iou,
            match_iou_thresh=self.match_iou_thresh,
            start_frame=start_frame,
            max_frames=max_frames,
            verbose=self.verbose,
        )
