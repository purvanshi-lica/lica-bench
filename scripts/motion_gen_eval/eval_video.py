#!/usr/bin/env python3
"""CLI entry point for YOLO-OBB motion tracking.

Examples
--------
# Track a specific component using layout-initialized tracking (no YOLO needed)
python main.py video.mp4 \\
    --layout layout.json \\
    --track "IMAGE 0-1"

# Track multiple components
python main.py video.mp4 \\
    --layout layout.json \\
    --track "IMAGE 0-1" --track "TEXT 0-5"

# Use YOLO OBB detection mode (requires trained weights)
python main.py video.mp4 \\
    --mode yolo \\
    --weights yolo-obb.pt \\
    --layout layout.json \\
    --track "IMAGE 0-1"

# List all components in a layout
python main.py dummy --layout layout.json --list-components
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the ``motion_gen_eval`` package importable even when the package is
# not installed (``pip install -e .`` would also work).
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from motion_gen_eval.config import TrackingConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO-OBB motion tracking on video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("video", help="Path or URL to the input video")

    g = p.add_argument_group("mode")
    g.add_argument(
        "--mode", choices=["layout-init", "yolo", "contour"], default="layout-init",
        help=(
            "Tracking mode. 'contour' uses background subtraction for single "
            "objects on uniform backgrounds (no model needed). 'layout-init' "
            "uses layout JSON positions + OpenCV trackers. 'yolo' uses YOLO "
            "OBB detection. Default: layout-init"
        ),
    )

    g = p.add_argument_group("model (yolo mode only)")
    g.add_argument("--weights", default="yolo11x-obb.pt", help="Path to YOLO-OBB .pt checkpoint")
    g.add_argument("--device", default="", help='Device string ("cuda:0", "cpu", "" for auto)')

    g = p.add_argument_group("detection")
    g.add_argument("--imgsz", type=int, default=1280)
    g.add_argument("--conf", type=float, default=0.01,
                   help="Confidence threshold (default permissive; spatial "
                        "matching gives a strong prior in yolo mode)")
    g.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    g.add_argument("--max-det", type=int, default=300)
    g.add_argument(
        "--target-classes", type=int, nargs="*", default=None,
        help="Class IDs to keep (omit for all)",
    )

    g = p.add_argument_group("tracker")
    g.add_argument(
        "--cv-tracker", default="DaSiamRPN",
        help=(
            "OpenCV tracker type (layout-init mode). Options: "
            "DaSiamRPN (default, accurate), Nano, MIL, Vit, CSRT, KCF"
        ),
    )

    g = p.add_argument_group("component tracking")
    g.add_argument(
        "--layout", default=None,
        help="Path to LICA layout JSON (component metadata)",
    )
    g.add_argument(
        "--track", action="append", default=None, dest="track_components",
        help=(
            'Component to track, as "TYPE ID" (e.g. --track "IMAGE 0-1"). '
            "Can be repeated. Requires --layout."
        ),
    )
    g.add_argument(
        "--match-iou-thresh", type=float, default=0.03,
        help="Min polygon IoU to match a YOLO detection to a layout "
             "component (yolo mode, per-frame matching)",
    )
    g.add_argument(
        "--list-components", action="store_true",
        help="Print components from --layout and exit (no tracking)",
    )

    g = p.add_argument_group("contour mode")
    g.add_argument(
        "--bg-threshold", type=int, nargs="+", default=None,
        help=(
            "Pixel diff threshold(s) from background (contour mode). "
            "Default: auto-discover from the intensity histogram. "
            "Override with explicit values if needed, e.g. --bg-threshold 1 30"
        ),
    )
    g.add_argument(
        "--morph-kernel", type=int, default=5,
        help="Morphological cleanup kernel size, 0 to disable (contour mode)",
    )

    g = p.add_argument_group("output")
    g.add_argument("--output-dir", default="output", help="Directory for results")
    g.add_argument("--no-json", action="store_true", help="Skip JSON output")

    g = p.add_argument_group("evaluation")
    g.add_argument(
        "--evaluate", action="store_true",
        help="Run layout-fidelity / temporal metrics after tracking completes",
    )
    g.add_argument(
        "--eval-output", default=None,
        help="Write layout-fidelity metrics JSON to this path",
    )
    g.add_argument(
        "--motion-eval", action="store_true",
        help=(
            "Run motion-correctness metrics (motion type / direction / "
            "animation+component duration) from motion_metrics.evaluate_sample. "
            "Requires --layout."
        ),
    )
    g.add_argument(
        "--motion-eval-output", default=None,
        help="Write motion-correctness metrics JSON to this path",
    )
    g.add_argument(
        "--motion-eval-component", default=None,
        help=(
            "Component id to evaluate motion for (default: first animated "
            "component in the layout)"
        ),
    )

    g = p.add_argument_group("misc")
    g.add_argument("--start-frame", type=int, default=0)
    g.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    g.add_argument("--verbose", action="store_true")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # -- list-components shortcut --
    if args.list_components:
        if not args.layout:
            print("Error: --list-components requires --layout", file=sys.stderr)
            sys.exit(1)
        from motion_gen_eval.layout import parse_layout

        layout = parse_layout(args.layout)
        print(layout.summary())
        print()
        print("Use --track \"TYPE ID\" to track a specific component, e.g.:")
        for c in layout.components:
            print(f'  --track "{c.type} {c.id}"')
        return

    if args.track_components and not args.layout:
        print("Error: --track requires --layout", file=sys.stderr)
        sys.exit(1)

    cfg = TrackingConfig(
        mode=args.mode,
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        target_classes=args.target_classes,
        cv_tracker_type=args.cv_tracker,
        bg_thresholds=args.bg_threshold,
        morph_kernel=args.morph_kernel,
        output_json=not args.no_json,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        verbose=args.verbose,
        layout_json=args.layout,
        track_components=args.track_components,
        match_iou_thresh=args.match_iou_thresh,
    )

    if cfg.mode == "contour":
        from motion_gen_eval.contour_tracker import ContourTracker

        tracker = ContourTracker(cfg)
        summary = tracker.track_video(args.video)

    elif cfg.mode == "layout-init":
        from motion_gen_eval.layout_tracker import LayoutInitTracker

        if not cfg.layout_json or not cfg.track_components:
            print(
                "Error: layout-init mode requires --layout and --track",
                file=sys.stderr,
            )
            sys.exit(1)

        tracker = LayoutInitTracker(cfg)
        summary = tracker.track_video(args.video)

    elif cfg.mode == "yolo":
        # Per-frame YOLO + spatial polygon-IoU matching against the layout.
        # Replaces the old ByteTrack-based OBBTracker.
        if not cfg.layout_json:
            print(
                "Error: yolo mode requires --layout (layout JSON is used to "
                "match detections to components)",
                file=sys.stderr,
            )
            sys.exit(1)

        from motion_gen_eval.frame_detector import (
            YoloFrameDetector,
            find_animated_components,
        )
        from motion_gen_eval.layout import parse_layout

        if cfg.track_components:
            components: list[dict] = []
            for spec in cfg.track_components:
                parts = spec.strip().split(None, 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid component spec {spec!r}. Expected 'TYPE ID'"
                    )
                ctype, cid = parts[0].upper(), parts[1]
                components.append({"id": cid, "detect_id": cid, "type": ctype})
            extra = find_animated_components(cfg.layout_json)
            seen = {c["id"] for c in components}
            seen.update(c["detect_id"] for c in components)
            for c in extra:
                if c["id"] in seen or c["detect_id"] in seen:
                    continue
                components.append(c)
                seen.add(c["id"])
                seen.add(c["detect_id"])
        else:
            components = find_animated_components(cfg.layout_json)
            if not components:
                _layout = parse_layout(cfg.layout_json)
                components = [
                    {"id": c.id, "detect_id": c.id, "type": c.type.upper()}
                    for c in _layout.components
                    if c.type.upper() in {"IMAGE", "TEXT"}
                ]

        detector = YoloFrameDetector(
            weights=cfg.weights,
            imgsz=cfg.imgsz,
            conf=cfg.conf,
            nms_iou=cfg.iou,
            match_iou_thresh=cfg.match_iou_thresh,
            device=cfg.device,
            verbose=cfg.verbose,
        )
        detector.load()
        summary = detector.detect_video(
            video_path=args.video,
            layout_path=cfg.layout_json,
            components=components,
            start_frame=cfg.start_frame,
            max_frames=cfg.max_frames,
        )
        if cfg.output_json:
            out_path = cfg.output_path / (Path(args.video).stem + "_tracks.json")
            out_path.write_text(json.dumps(summary, indent=2, default=str))
            summary["json_output"] = str(out_path)
            print(f"[main] JSON results -> {out_path}")

    else:
        print(f"Error: unknown mode {cfg.mode!r}", file=sys.stderr)
        sys.exit(1)

    total = summary["total_frames_processed"]
    fps_avg = summary["avg_fps"]
    elapsed = summary["processing_time_s"]

    if cfg.mode == "contour":
        detected = summary.get("frames_with_object", 0)
        print(
            f"\n[done] {total} frames, object detected in {detected}, "
            f"{fps_avg:.1f} avg fps, {elapsed:.1f}s total"
        )
    else:
        print(
            f"\n[done] {total} frames, "
            f"{summary.get('unique_tracks', 'N/A')} unique tracks, "
            f"{fps_avg:.1f} avg fps, {elapsed:.1f}s total"
        )

    if "json_output" in summary:
        print(f"  json  -> {summary['json_output']}")

    if "component_tracking" in summary:
        ct = summary["component_tracking"]
        print("\nComponent tracking results:")
        per = ct.get("per_component", {})
        for comp_id, info in per.items():
            if isinstance(info, dict) and "frames_tracked" in info:
                status = "LOST" if info.get("lost") else "OK"
                print(f"  {comp_id}: {info['frames_tracked']} frames [{status}]")
        if ct.get("unmatched"):
            print(f"  Unmatched: {ct['unmatched']}")

    if args.evaluate:
        from motion_gen_eval.metrics import evaluate_tracking, print_report

        print()
        eval_results = evaluate_tracking(
            summary,
            layout_path=args.layout,
            video_path=args.video,
        )
        print_report(eval_results)

        if args.eval_output:
            out = Path(args.eval_output)
            out.write_text(json.dumps(eval_results, indent=2))
            print(f"\nMetrics JSON -> {out}")

    if args.motion_eval:
        if not args.layout:
            print("Error: --motion-eval requires --layout", file=sys.stderr)
            sys.exit(1)

        from dataclasses import asdict
        from motion_gen_eval.motion_metrics import (
            evaluate_sample,
            evaluation_to_dict,
        )

        tracks_path = summary.get("json_output")
        if not tracks_path:
            print(
                "Error: --motion-eval requires the tracker to produce a "
                "_tracks.json (do not pass --no-json).",
                file=sys.stderr,
            )
            sys.exit(1)

        ev = evaluate_sample(
            layout_path=args.layout,
            tracks_json_path=tracks_path,
            sample_id=Path(args.video).stem,
            component_id=args.motion_eval_component,
        )

        print()
        print("=" * 65)
        print("  MOTION-CORRECTNESS REPORT")
        print("=" * 65)
        gt = ev.ground_truth
        print(f"  GT motion type   : {gt.motion_type:<10} (raw={gt.motion_type_raw})")
        print(f"  GT direction     : {gt.direction}")
        print(f"  GT anim duration : {gt.animation_duration_s}")
        print(f"  GT comp duration : {ev.component_duration.gt_s}")
        print()
        print(f"  Pred motion type : {ev.motion_type.predicted}  "
              f"score={ev.motion_type.score:.2f}")
        print(f"  Pred direction   : {ev.direction.predicted}  "
              f"score={ev.direction.score}")
        print(f"  Pred anim dur    : {ev.animation_duration.predicted_s}s  "
              f"abs_err={ev.animation_duration.abs_error_s}s  "
              f"score={ev.animation_duration.score}")
        print(f"  Pred comp dur    : {ev.component_duration.predicted_s}s  "
              f"abs_err={ev.component_duration.abs_error_s}s  "
              f"score={ev.component_duration.score}")
        print(f"  Tracker presence : "
              f"{ev.tracking_quality.presence_frac:.0%} "
              f"(reliable={ev.tracking_quality.is_reliable})")

        if args.motion_eval_output:
            out = Path(args.motion_eval_output)
            out.write_text(json.dumps(evaluation_to_dict(ev), indent=2, default=str))
            print(f"\nMotion metrics JSON -> {out}")


if __name__ == "__main__":
    main()
