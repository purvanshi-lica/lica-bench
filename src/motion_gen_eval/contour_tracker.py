"""Contour-based OBB tracker for objects on uniform backgrounds.

No ML model needed.  Works by:
1. Subtracting the known background colour from each frame
2. Auto-discovering threshold levels from the intensity histogram
   (each peak = a distinct visual element at a different contrast level)
3. Finding contours at each level and computing OBBs via cv2.minAreaRect

Handles any number of components automatically -- a subtle cream paper and
a bold golden medallion get separate OBBs without hardcoding thresholds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from motion_gen_eval.config import TrackingConfig
from motion_gen_eval.video_io import get_video_info, iterate_frames, resolve_video_source


# ---------------------------------------------------------------------------
# Background helpers
# ---------------------------------------------------------------------------

def _parse_rgb(color_str: str) -> Tuple[int, int, int]:
    s = color_str.strip()
    if s.startswith("rgb"):
        nums = s.replace("rgb(", "").replace(")", "").split(",")
        return tuple(int(float(n.strip())) for n in nums)[:3]
    if s.startswith("#") and len(s) == 7:
        return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))
    return (255, 255, 255)


def _detect_background_color(frame: np.ndarray) -> Tuple[int, int, int]:
    h, w = frame.shape[:2]
    corners = [frame[0, 0], frame[0, w - 1], frame[h - 1, 0], frame[h - 1, w - 1]]
    avg = np.mean(corners, axis=0).astype(int)
    return (int(avg[2]), int(avg[1]), int(avg[0]))


# ---------------------------------------------------------------------------
# Threshold auto-discovery
# ---------------------------------------------------------------------------

def _bg_diff_gray(frame_bgr: np.ndarray, bg_rgb: Tuple[int, int, int]) -> np.ndarray:
    bg_bgr = np.array([bg_rgb[2], bg_rgb[1], bg_rgb[0]], dtype=np.uint8)
    return cv2.cvtColor(cv2.absdiff(frame_bgr, bg_bgr), cv2.COLOR_BGR2GRAY)


def auto_thresholds(
    gray_diff: np.ndarray,
    min_gap_bins: int = 5,
    min_region_pixels: int = 500,
) -> List[int]:
    """Discover natural threshold levels from the intensity histogram.

    Instead of looking for peaks (which fails when one cluster is 300x
    larger than another), this finds **gaps** -- sustained runs of empty
    or near-empty bins that separate occupied regions.

    Each gap boundary becomes a threshold.  For a frame with a subtle cream
    paper (diff 1-5) and a bold medallion (diff 20-200), the gap from 6-19
    produces a threshold at ~6 that separates them.
    """
    nonzero = gray_diff[gray_diff > 0]
    if len(nonzero) < 100:
        return []

    hist = np.bincount(nonzero.ravel(), minlength=256)

    # Classify each bin as "occupied" or "empty".
    # A bin is occupied if it has more than a tiny fraction of pixels.
    noise_floor = max(5, len(nonzero) * 0.00005)

    # Walk through the histogram and find contiguous occupied regions
    # separated by gaps of empty bins.
    regions: List[Tuple[int, int, int]] = []  # (start, end, pixel_count)
    in_region = False
    region_start = 0
    region_pixels = 0

    for i in range(1, 256):
        occupied = hist[i] > noise_floor
        if occupied and not in_region:
            region_start = i
            region_pixels = int(hist[i])
            in_region = True
        elif occupied and in_region:
            region_pixels += int(hist[i])
        elif not occupied and in_region:
            # Check if this is a real gap or just a single empty bin
            gap_len = 0
            for j in range(i, min(i + min_gap_bins + 1, 256)):
                if hist[j] <= noise_floor:
                    gap_len += 1
                else:
                    break
            if gap_len >= min_gap_bins:
                # Real gap -- close this region
                if region_pixels >= min_region_pixels:
                    regions.append((region_start, i - 1, region_pixels))
                in_region = False
            else:
                region_pixels += int(hist[i])

    # Close final region
    if in_region and region_pixels >= min_region_pixels:
        end = max(i for i in range(1, 256) if hist[i] > noise_floor)
        regions.append((region_start, end, region_pixels))

    if len(regions) <= 1:
        return []

    # Place a threshold at the midpoint of each gap between consecutive regions
    thresholds: List[int] = []
    for (_, end_a, _), (start_b, _, _) in zip(regions[:-1], regions[1:]):
        thresholds.append((end_a + start_b) // 2)

    return sorted(set(thresholds))


def _build_threshold_levels(
    auto: List[int],
    explicit: Optional[List[int]],
) -> List[int]:
    """Combine auto-discovered valleys with any explicit overrides.

    Always includes 1 as the lowest level (catches the subtlest foreground).
    """
    if explicit and len(explicit) > 1:
        # User gave explicit multi-level thresholds -- honour them
        return sorted(set(explicit))

    levels = {1}  # always include a very low threshold
    for v in auto:
        if v > 1:
            levels.add(v)
    return sorted(levels)


# ---------------------------------------------------------------------------
# Contour -> OBB conversion
# ---------------------------------------------------------------------------

def _threshold_mask(gray: np.ndarray, threshold: int, morph_kernel: int) -> np.ndarray:
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    if morph_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def _contour_to_obb(
    contour: np.ndarray, track_id: int, min_area: float = 100,
) -> Optional[Dict[str, Any]]:
    area = cv2.contourArea(contour)
    if area < min_area:
        return None
    rect = cv2.minAreaRect(contour)
    box_pts = cv2.boxPoints(rect).astype(np.float32)
    cx, cy = rect[0]
    w, h = rect[1]
    angle = rect[2]
    return {
        "track_id": track_id,
        "class_id": 0,
        "confidence": 1.0,
        "polygon": [round(v, 2) for v in box_pts.reshape(-1).tolist()],
        "center": [round(cx, 2), round(cy, 2)],
        "obb_width": round(max(w, h), 2),
        "obb_height": round(min(w, h), 2),
        "angle": round(angle, 2),
        "area": round(area, 1),
    }


# ---------------------------------------------------------------------------
# Multi-level detection + deduplication
# ---------------------------------------------------------------------------

def _detect_multilevel(
    frame_bgr: np.ndarray,
    bg_rgb: Tuple[int, int, int],
    thresholds: List[int],
    morph_kernel: int,
    min_area: float = 100,
) -> List[Dict[str, Any]]:
    gray = _bg_diff_gray(frame_bgr, bg_rgb)

    level_obbs: List[Tuple[int, Dict[str, Any]]] = []
    for thresh in sorted(thresholds):
        mask = _threshold_mask(gray, thresh, morph_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            obb = _contour_to_obb(cnt, track_id=0, min_area=min_area)
            if obb is not None:
                level_obbs.append((thresh, obb))

    if not level_obbs:
        return []

    # Deduplicate: walk from highest to lowest threshold. Two OBBs are the
    # "same object" if their centres are within 30% of the smaller diagonal.
    # Keep the highest-threshold version (tightest fit).
    final: List[Dict[str, Any]] = []
    used = [False] * len(level_obbs)

    for i in range(len(level_obbs) - 1, -1, -1):
        if used[i]:
            continue
        thresh_i, obb_i = level_obbs[i]
        ci = obb_i["center"]

        for j in range(i):
            if used[j]:
                continue
            _, obb_j = level_obbs[j]
            cj = obb_j["center"]
            diag = min(
                (obb_i["obb_width"] ** 2 + obb_i["obb_height"] ** 2) ** 0.5,
                (obb_j["obb_width"] ** 2 + obb_j["obb_height"] ** 2) ** 0.5,
            )
            dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
            if dist < diag * 0.3:
                used[j] = True

        obb_i["threshold_level"] = thresh_i
        final.append(obb_i)
        used[i] = True

    final.sort(key=lambda d: (d["center"][1], d["center"][0]))
    for idx, det in enumerate(final):
        det["track_id"] = idx
    return final


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ContourTracker:
    """Track objects on a uniform background using contour detection.

    Automatically discovers the right threshold levels from the intensity
    histogram of the first visible frame.  No hardcoded thresholds needed.
    """

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.bg_rgb: Tuple[int, int, int] = (255, 255, 255)
        self.morph_kernel: int = cfg.morph_kernel
        self._thresholds: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Calibration pass: scan forward for a stable frame to discover
    # thresholds, without writing any output.
    # ------------------------------------------------------------------
    def _calibrate_from_video(
        self, video_path: str, info: Dict[str, Any],
    ) -> None:
        """Run a quick scan to find thresholds before the real pass."""
        if self.cfg.bg_thresholds:
            self._thresholds = _build_threshold_levels([], self.cfg.bg_thresholds)
            print(f"[contour] Using explicit thresholds: {self._thresholds}")
            return

        _MOTION_THRESH = 0.10
        prev_frame: Optional[np.ndarray] = None
        total = info.get("total_frames", 300)
        fallback_limit = max(60, total // 3)

        for frame_idx, bgr_frame in iterate_frames(
            video_path,
            start_frame=self.cfg.start_frame,
            max_frames=self.cfg.max_frames,
        ):
            if not self.cfg.layout_json:
                self.bg_rgb = _detect_background_color(bgr_frame)

            gray = _bg_diff_gray(bgr_frame, self.bg_rgb)
            fg_pct = 100 * np.count_nonzero(gray) / gray.size

            is_stable = False
            if prev_frame is not None and fg_pct > 5.0:
                motion = float(np.mean(cv2.absdiff(bgr_frame, prev_frame)))
                is_stable = motion < _MOTION_THRESH
            prev_frame = bgr_frame.copy()

            if is_stable:
                discovered = auto_thresholds(gray)
                if discovered:
                    self._thresholds = _build_threshold_levels(
                        discovered, self.cfg.bg_thresholds,
                    )
                    print(f"[contour] Background: rgb{self.bg_rgb}")
                    print(f"[contour] Auto-discovered gaps: {discovered}")
                    print(f"[contour] Using threshold levels: {self._thresholds}")
                    print(f"[contour] Calibrated on frame {frame_idx} "
                          f"({fg_pct:.1f}% foreground, stable)")
                    return

            if fg_pct > 5.0 and frame_idx - self.cfg.start_frame > fallback_limit:
                discovered = auto_thresholds(gray)
                self._thresholds = _build_threshold_levels(
                    discovered, self.cfg.bg_thresholds,
                )
                print(f"[contour] Background: rgb{self.bg_rgb}")
                print(f"[contour] Auto-discovered gaps: {discovered}")
                print(f"[contour] Using threshold levels: {self._thresholds}")
                print(f"[contour] Fallback-calibrated on frame {frame_idx}")
                return

        if self._thresholds is None:
            self._thresholds = [1]
            print("[contour] No foreground found during calibration; "
                  "defaulting to threshold [1]")

    # ------------------------------------------------------------------
    # Main tracking pass
    # ------------------------------------------------------------------
    def track_video(self, source: str) -> Dict[str, Any]:
        video_path = resolve_video_source(source)
        info = get_video_info(video_path)
        fps = info["fps"] or 30.0

        if self.cfg.layout_json:
            self._load_bg_from_layout(self.cfg.layout_json)

        # --- Pass 1: calibrate thresholds (fast, no output) ---
        self._calibrate_from_video(video_path, info)

        # --- Pass 2: detect from frame 0 ---
        all_results: List[dict] = []
        frames_with_object = 0
        max_objects_seen = 0
        t0 = time.perf_counter()

        for frame_idx, bgr_frame in iterate_frames(
            video_path,
            start_frame=self.cfg.start_frame,
            max_frames=self.cfg.max_frames,
        ):
            detections = _detect_multilevel(
                bgr_frame, self.bg_rgb, self._thresholds,
                self.morph_kernel,
            )

            if detections:
                frames_with_object += 1
                max_objects_seen = max(max_objects_seen, len(detections))

            ts_ms = (frame_idx / fps) * 1000.0
            all_results.append({
                "frame_idx": frame_idx,
                "timestamp_ms": round(ts_ms, 2),
                "num_detections": len(detections),
                "detections": detections,
            })

            if self.cfg.verbose and frame_idx % 50 == 0:
                elapsed = time.perf_counter() - t0
                n = frame_idx - self.cfg.start_frame + 1
                obj_str = ", ".join(
                    f"({d['center'][0]:.0f},{d['center'][1]:.0f})"
                    for d in detections
                ) or "none"
                print(
                    f"[contour] frame {frame_idx} | "
                    f"{len(detections)} objects: {obj_str} | "
                    f"{n / max(elapsed, 1e-6):.0f} fps"
                )

        elapsed = time.perf_counter() - t0

        summary: Dict[str, Any] = {
            "mode": "contour",
            "video_source": str(source),
            "video_info": info,
            "config": {
                "background_rgb": list(self.bg_rgb),
                "thresholds_used": self._thresholds,
                "morph_kernel": self.morph_kernel,
            },
            "total_frames_processed": len(all_results),
            "frames_with_object": frames_with_object,
            "max_objects_per_frame": max_objects_seen,
            "processing_time_s": round(elapsed, 2),
            "avg_fps": round(len(all_results) / max(elapsed, 1e-6), 2),
            "frames": all_results,
        }

        if self.cfg.output_json:
            json_path = self.cfg.output_path / (Path(video_path).stem + "_tracks.json")
            json_path.write_text(json.dumps(summary, indent=2))
            summary["json_output"] = str(json_path)
            print(f"[contour] JSON -> {json_path}")

        return summary

    def _load_bg_from_layout(self, path: str) -> None:
        try:
            data = json.loads(Path(path).read_text())
            if "layout_config" in data:
                style = data["layout_config"].get("style", {})
            else:
                style = data
            bg_str = style.get("background", "")
            if bg_str:
                self.bg_rgb = _parse_rgb(bg_str)
                print(f"[contour] Background from layout: rgb{self.bg_rgb}")
        except Exception as e:
            print(f"[contour] Could not read background from layout: {e}")
