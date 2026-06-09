#!/usr/bin/env python3
"""Evaluate tracking quality against layout ground truth.

Can be used standalone::

    python metrics.py output/foo_tracks.json --layout path/to/layout.json

Or integrated via ``--evaluate`` in ``main.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from motion_gen_eval.layout import Layout, LayoutComponent, parse_layout, polygon_iou


# ── Helpers ──────────────────────────────────────────────────────────


def _det_polygon(det: Dict[str, Any]) -> np.ndarray:
    """Extract the (4, 2) polygon array from a detection dict."""
    return np.array(det["polygon"], dtype=np.float64).reshape(4, 2)


def _det_center(det: Dict[str, Any]) -> Tuple[float, float]:
    """Get or compute the centroid of a detection."""
    if "center" in det:
        c = det["center"]
        return float(c[0]), float(c[1])
    poly = _det_polygon(det)
    return float(poly[:, 0].mean()), float(poly[:, 1].mean())


def _det_obb_params(det: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return (width, height, angle_deg) from a detection.

    Works for both contour detections (obb_width/obb_height/angle keys)
    and YOLO detections (computed from the polygon via minAreaRect).
    """
    if "obb_width" in det:
        return det["obb_width"], det["obb_height"], det.get("angle", 0.0)
    poly = _det_polygon(det)
    pts = poly.astype(np.float32)
    import cv2
    rect = cv2.minAreaRect(pts)
    (_, (w, h), angle) = rect
    return float(w), float(h), float(angle)


def _center_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_diff(a: float, b: float) -> float:
    """Smallest angle difference in degrees, accounting for 180-deg ambiguity."""
    d = abs(a - b) % 360
    if d > 180:
        d = 360 - d
    if d > 90:
        d = 180 - d
    return d


# ── 1. Component detection report ────────────────────────────────────


def _component_detection_report(
    frames: List[dict],
    layout: Layout,
    iou_thresh: float = 0.05,
    center_dist_thresh: float = 300.0,
) -> Dict[str, Any]:
    """For each layout component, determine if/when it was detected."""
    report: Dict[str, Dict[str, Any]] = {}

    for comp in layout.components:
        comp_poly = comp.polygon()
        comp_center = comp.center

        best_iou = 0.0
        best_center_dist = float("inf")
        first_frame: Optional[int] = None
        last_frame: Optional[int] = None
        frames_matched = 0

        for frame in frames:
            frame_idx = frame["frame_idx"]
            for det in frame.get("detections", []):
                det_poly = _det_polygon(det)
                iou = polygon_iou(comp_poly, det_poly)
                det_c = _det_center(det)
                dist = _center_distance(comp_center, det_c)

                matched = iou >= iou_thresh or dist < center_dist_thresh
                if matched:
                    if iou > best_iou:
                        best_iou = iou
                    if dist < best_center_dist:
                        best_center_dist = dist
                    if first_frame is None:
                        first_frame = frame_idx
                    last_frame = frame_idx
                    frames_matched += 1
                    break  # one match per frame is enough

        detected = frames_matched > 0
        report[comp.id] = {
            "type": comp.type,
            "label": comp.text or comp.src or "",
            "layout_center": [round(comp_center[0], 1), round(comp_center[1], 1)],
            "layout_size": [round(comp.width, 1), round(comp.height, 1)],
            "detected": detected,
            "best_iou": round(best_iou, 4),
            "best_center_dist_px": round(best_center_dist, 1) if detected else None,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "frames_matched": frames_matched,
        }

    n_detected = sum(1 for v in report.values() if v["detected"])
    n_total = len(report)

    return {
        "components_detected": n_detected,
        "components_total": n_total,
        "detection_rate": round(n_detected / max(n_total, 1), 4),
        "per_component": report,
    }


# ── 2. Settled-frame accuracy ────────────────────────────────────────


def _find_settled_frames(
    frames: List[dict],
    video_path: Optional[str] = None,
) -> List[int]:
    """Identify frames where the scene is stable (minimal motion).

    If video_path is available, uses inter-frame pixel diff.
    Otherwise falls back to detecting frames with consistent detections.
    """
    if video_path:
        try:
            import cv2
            from motion_gen_eval.video_io import iterate_frames
            prev = None
            settled = []
            frame_set = {f["frame_idx"] for f in frames}
            for fidx, bgr in iterate_frames(video_path):
                if fidx not in frame_set:
                    continue
                if prev is not None:
                    motion = float(np.mean(cv2.absdiff(bgr, prev)))
                    if motion < 0.1:
                        settled.append(fidx)
                prev = bgr.copy()
                if len(settled) >= 20:
                    break
            if settled:
                return settled
        except Exception:
            pass

    # Fallback: frames in the middle 50% of the video with stable detection count
    if not frames:
        return []
    n = len(frames)
    mid_start = n // 4
    mid_end = 3 * n // 4
    mid_frames = frames[mid_start:mid_end]

    if not mid_frames:
        return [f["frame_idx"] for f in frames]

    det_counts = [f["num_detections"] for f in mid_frames]
    if not det_counts:
        return []
    from statistics import mode as stat_mode
    try:
        common_count = stat_mode(det_counts)
    except Exception:
        common_count = det_counts[0]

    return [
        f["frame_idx"] for f in mid_frames
        if f["num_detections"] == common_count
    ][:20]


def _settled_frame_accuracy(
    frames: List[dict],
    layout: Layout,
    video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """On settled frames, compare each detection to the nearest layout component."""
    settled_idxs = set(_find_settled_frames(frames, video_path))
    if not settled_idxs:
        return {"settled_frames": 0, "note": "no settled frames found"}

    settled_frames = [f for f in frames if f["frame_idx"] in settled_idxs]

    per_component: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for frame in settled_frames:
        for det in frame.get("detections", []):
            det_poly = _det_polygon(det)
            det_c = _det_center(det)
            det_w, det_h, det_angle = _det_obb_params(det)

            best_comp: Optional[LayoutComponent] = None
            best_iou = -1.0
            best_dist = float("inf")

            for comp in layout.components:
                comp_poly = comp.polygon()
                iou = polygon_iou(comp_poly, det_poly)
                dist = _center_distance(comp.center, det_c)
                if iou > best_iou or (iou == best_iou and dist < best_dist):
                    best_iou = iou
                    best_dist = dist
                    best_comp = comp

            if best_comp is not None:
                size_ratio_w = det_w / max(best_comp.width, 1)
                size_ratio_h = det_h / max(best_comp.height, 1)
                angle_err = _angle_diff(det_angle, best_comp.rotation_deg)

                per_component[best_comp.id].append({
                    "iou": best_iou,
                    "center_offset_px": best_dist,
                    "size_ratio_w": size_ratio_w,
                    "size_ratio_h": size_ratio_h,
                    "angle_error_deg": angle_err,
                })

    summary: Dict[str, Any] = {}
    for comp_id, measurements in per_component.items():
        ious = [m["iou"] for m in measurements]
        offsets = [m["center_offset_px"] for m in measurements]
        summary[comp_id] = {
            "n_measurements": len(measurements),
            "mean_iou": round(float(np.mean(ious)), 4),
            "mean_center_offset_px": round(float(np.mean(offsets)), 1),
            "mean_size_ratio_w": round(float(np.mean([m["size_ratio_w"] for m in measurements])), 3),
            "mean_size_ratio_h": round(float(np.mean([m["size_ratio_h"] for m in measurements])), 3),
            "mean_angle_error_deg": round(float(np.mean([m["angle_error_deg"] for m in measurements])), 1),
        }

    return {
        "settled_frames_used": len(settled_frames),
        "per_component": summary,
    }


# ── 3. Temporal consistency ──────────────────────────────────────────


def _temporal_consistency(frames: List[dict]) -> Dict[str, Any]:
    """Per-track temporal metrics: coverage, fragmentation, smoothness."""
    tracks: Dict[int, List[Tuple[int, Dict]]] = defaultdict(list)
    for frame in frames:
        fidx = frame["frame_idx"]
        for det in frame.get("detections", []):
            tid = det.get("track_id", -1)
            if tid < 0:
                continue
            tracks[tid].append((fidx, det))

    per_track: Dict[str, Dict[str, Any]] = {}

    for tid, appearances in tracks.items():
        appearances.sort(key=lambda x: x[0])
        frame_idxs = sorted(set(a[0] for a in appearances))
        first, last = frame_idxs[0], frame_idxs[-1]
        span = last - first + 1
        coverage = len(frame_idxs) / max(span, 1)

        # Fragmentation: count gaps
        gaps = 0
        gap_lengths: List[int] = []
        for i in range(1, len(frame_idxs)):
            delta = frame_idxs[i] - frame_idxs[i - 1]
            if delta > 1:
                gaps += 1
                gap_lengths.append(delta - 1)

        # Trajectory smoothness
        centers = [_det_center(a[1]) for a in appearances]
        displacements = []
        for i in range(1, len(centers)):
            d = _center_distance(centers[i - 1], centers[i])
            displacements.append(d)

        mean_disp = float(np.mean(displacements)) if displacements else 0.0
        max_disp = float(np.max(displacements)) if displacements else 0.0
        jitter = float(np.std(displacements)) if len(displacements) > 1 else 0.0

        # OBB stability
        widths, heights, angles = [], [], []
        for _, det in appearances:
            w, h, a = _det_obb_params(det)
            widths.append(w)
            heights.append(h)
            angles.append(a)

        per_track[str(tid)] = {
            "first_frame": first,
            "last_frame": last,
            "span_frames": span,
            "frames_present": len(frame_idxs),
            "coverage": round(coverage, 4),
            "gaps": gaps,
            "max_gap_frames": max(gap_lengths) if gap_lengths else 0,
            "trajectory": {
                "mean_displacement_px": round(mean_disp, 2),
                "max_displacement_px": round(max_disp, 2),
                "jitter_px": round(jitter, 2),
            },
            "obb_stability": {
                "width_std": round(float(np.std(widths)), 2) if widths else 0,
                "height_std": round(float(np.std(heights)), 2) if heights else 0,
                "angle_std": round(float(np.std(angles)), 2) if angles else 0,
            },
        }

    return {"num_tracks": len(per_track), "per_track": per_track}


# ── 4. Summary statistics ────────────────────────────────────────────


def _summary_statistics(frames: List[dict], tracks_json: dict) -> Dict[str, Any]:
    """High-level aggregate stats."""
    total_frames = len(frames)
    all_det_counts = [f["num_detections"] for f in frames]
    frames_with_dets = sum(1 for c in all_det_counts if c > 0)
    total_detections = sum(all_det_counts)

    # Active range: first to last frame with detections
    active_frames = [f["frame_idx"] for f in frames if f["num_detections"] > 0]
    if active_frames:
        active_start = min(active_frames)
        active_end = max(active_frames)
        active_span = active_end - active_start + 1
        dead_frames_in_active = active_span - len(active_frames)
    else:
        active_start = active_end = active_span = dead_frames_in_active = 0

    confidences = []
    for f in frames:
        for d in f.get("detections", []):
            c = d.get("confidence")
            if c is not None and c < 1.0:  # 1.0 = contour mode synthetic
                confidences.append(c)

    unique_tracks = set()
    for f in frames:
        for d in f.get("detections", []):
            tid = d.get("track_id", -1)
            if tid >= 0:
                unique_tracks.add(tid)

    stats: Dict[str, Any] = {
        "total_frames": total_frames,
        "frames_with_detections": frames_with_dets,
        "total_detections": total_detections,
        "unique_track_ids": len(unique_tracks),
        "active_range": {
            "start_frame": active_start,
            "end_frame": active_end,
            "span_frames": active_span,
            "dead_frames_in_range": dead_frames_in_active,
        },
    }

    if confidences:
        stats["confidence"] = {
            "mean": round(float(np.mean(confidences)), 4),
            "min": round(float(np.min(confidences)), 4),
            "max": round(float(np.max(confidences)), 4),
        }

    mode = tracks_json.get("mode")
    if not mode:
        mode = "yolo" if "unique_tracks" in tracks_json else "unknown"
    stats["mode"] = mode

    return stats


# ── Main entry point ─────────────────────────────────────────────────


def evaluate_tracking(
    tracks_json: dict,
    layout_path: Optional[str] = None,
    video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all applicable metrics on a tracking result.

    Args:
        tracks_json: Loaded JSON output from any tracker mode.
        layout_path: Path to layout JSON for GT-based metrics (optional).
        video_path: Path to video for motion-based settled-frame detection (optional).

    Returns:
        Dict with all computed metrics.
    """
    frames = tracks_json.get("frames", [])
    results: Dict[str, Any] = {}

    results["summary"] = _summary_statistics(frames, tracks_json)

    if layout_path:
        layout = parse_layout(layout_path)
        info = tracks_json.get("video_info", {})
        v_w = float(info.get("width") or 0)
        v_h = float(info.get("height") or 0)
        if v_w and v_h and (
            abs(v_w - layout.width) > 1.0 or abs(v_h - layout.height) > 1.0
        ):
            layout = layout.scaled_to(v_w, v_h)
        results["component_detection"] = _component_detection_report(frames, layout)
        results["settled_frame_accuracy"] = _settled_frame_accuracy(
            frames, layout, video_path,
        )

    results["temporal_consistency"] = _temporal_consistency(frames)

    return results


# ── Pretty-print report ──────────────────────────────────────────────


def print_report(metrics: Dict[str, Any]) -> None:
    """Print a human-readable evaluation report to stdout."""
    s = metrics.get("summary", {})
    mode = s.get("mode", "?")

    print("=" * 65)
    print(f"  TRACKING EVALUATION REPORT  (mode: {mode})")
    print("=" * 65)

    print(f"\n{'Total frames:':<30} {s.get('total_frames', '?')}")
    print(f"{'Frames with detections:':<30} {s.get('frames_with_detections', '?')}")
    print(f"{'Total detections:':<30} {s.get('total_detections', '?')}")
    print(f"{'Unique track IDs:':<30} {s.get('unique_track_ids', '?')}")

    ar = s.get("active_range", {})
    if ar.get("span_frames"):
        print(f"{'Active range:':<30} frames {ar['start_frame']}-{ar['end_frame']} "
              f"({ar['span_frames']} frames, {ar['dead_frames_in_range']} gaps)")

    conf = s.get("confidence")
    if conf:
        print(f"{'Confidence (mean/min/max):':<30} "
              f"{conf['mean']:.3f} / {conf['min']:.3f} / {conf['max']:.3f}")

    # ── Component detection ──
    cd = metrics.get("component_detection")
    if cd:
        print(f"\n{'─' * 65}")
        print(f"  COMPONENT DETECTION: {cd['components_detected']}/{cd['components_total']} "
              f"({cd['detection_rate']:.0%})")
        print(f"{'─' * 65}")

        header = f"  {'Component':<25} {'Det?':<6} {'BestIoU':<9} {'CtrDist':<10} {'Frames':<8}"
        print(header)
        print(f"  {'─' * 60}")

        for comp_id, info in cd["per_component"].items():
            tag = f"{info['type']} {comp_id}"
            det_str = "YES" if info["detected"] else "NO"
            iou_str = f"{info['best_iou']:.3f}" if info["detected"] else "-"
            dist_str = f"{info['best_center_dist_px']:.0f}px" if info["detected"] else "-"
            frames_str = str(info["frames_matched"]) if info["detected"] else "-"
            print(f"  {tag:<25} {det_str:<6} {iou_str:<9} {dist_str:<10} {frames_str:<8}")

        missing = [
            f"{v['type']} {k}" for k, v in cd["per_component"].items()
            if not v["detected"]
        ]
        if missing:
            print(f"\n  MISSING: {', '.join(missing)}")

    # ── Settled-frame accuracy ──
    sa = metrics.get("settled_frame_accuracy")
    if sa and sa.get("per_component"):
        print(f"\n{'─' * 65}")
        print(f"  SETTLED-FRAME ACCURACY ({sa['settled_frames_used']} frames)")
        print(f"{'─' * 65}")

        header = (f"  {'Component':<20} {'IoU':<8} {'Offset':<10} "
                  f"{'SzW':<8} {'SzH':<8} {'AngErr':<8}")
        print(header)
        print(f"  {'─' * 60}")

        for comp_id, info in sa["per_component"].items():
            print(f"  {comp_id:<20} "
                  f"{info['mean_iou']:.3f}   "
                  f"{info['mean_center_offset_px']:>6.1f}px  "
                  f"{info['mean_size_ratio_w']:.2f}x   "
                  f"{info['mean_size_ratio_h']:.2f}x   "
                  f"{info['mean_angle_error_deg']:>5.1f}°")

    # ── Temporal consistency ──
    tc = metrics.get("temporal_consistency", {})
    tracks = tc.get("per_track", {})
    if tracks:
        print(f"\n{'─' * 65}")
        print(f"  TEMPORAL CONSISTENCY ({tc['num_tracks']} tracks)")
        print(f"{'─' * 65}")

        header = (f"  {'Track':<8} {'Span':<8} {'Cover':<8} {'Gaps':<6} "
                  f"{'MeanDisp':<10} {'Jitter':<9} {'OBBstd':<12}")
        print(header)
        print(f"  {'─' * 60}")

        for tid, info in tracks.items():
            obb = info["obb_stability"]
            traj = info["trajectory"]
            obb_str = f"w={obb['width_std']:.1f} h={obb['height_std']:.1f}"
            print(f"  {tid:<8} "
                  f"{info['span_frames']:<8} "
                  f"{info['coverage']:.1%}{'':>3} "
                  f"{info['gaps']:<6} "
                  f"{traj['mean_displacement_px']:>6.1f}px  "
                  f"{traj['jitter_px']:>6.1f}px  "
                  f"{obb_str}")

    print(f"\n{'=' * 65}")


# ── Standalone CLI ───────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Evaluate tracking results against layout ground truth",
    )
    p.add_argument("tracks_json", help="Path to *_tracks.json output file")
    p.add_argument("--layout", default=None, help="Path to layout JSON (ground truth)")
    p.add_argument("--video", default=None, help="Path to video (for settled-frame detection)")
    p.add_argument("--output", default=None, help="Write metrics JSON to this path")
    args = p.parse_args(argv)

    tracks = json.loads(Path(args.tracks_json).read_text())

    layout_path = args.layout
    video_path = args.video or tracks.get("video_source")

    results = evaluate_tracking(tracks, layout_path, video_path)
    print_report(results)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nMetrics JSON -> {out}")


if __name__ == "__main__":
    main()
