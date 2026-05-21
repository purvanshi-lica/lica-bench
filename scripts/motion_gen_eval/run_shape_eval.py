#!/usr/bin/env python3
"""Run the full shape-side evaluation over the dataset.

For every sample in ``data/shape/``:

  1. Locate the matching render in ``data/renders/``.
  2. Run ``ContourTracker`` on the render (no-video, JSON only) -> per-frame OBBs.
  3. Compute motion metrics with ``motion_metrics.evaluate_sample``.
  4. Append row to a CSV and an aggregate JSON.

Usage::

    python run_shape_eval.py                       # all shape samples
    python run_shape_eval.py --component-types shape image
    python run_shape_eval.py --skip-tracking       # reuse existing *_tracks.json
    python run_shape_eval.py --output-dir results/ # destination folder
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_SRC = REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from motion_gen_eval.config import TrackingConfig  # noqa: E402
from motion_gen_eval.contour_tracker import ContourTracker  # noqa: E402
from motion_gen_eval.motion_metrics import (  # noqa: E402
    SampleEvaluation,
    aggregate,
    evaluate_sample,
    evaluation_to_dict,
)


DATA_DIR = REPO_ROOT / "data" / "motion_gen_eval"
RENDERS_DIR = DATA_DIR / "renders"


# =====================================================================
#  Sample discovery
# =====================================================================


def discover_samples(component_types: Iterable[str]) -> List[Dict[str, Any]]:
    """Walk ``data/<component_type>/<param>/<sample>.json`` and pair with renders."""
    out: List[Dict[str, Any]] = []
    for ctype in component_types:
        cdir = DATA_DIR / ctype
        if not cdir.is_dir():
            continue
        for layout_path in sorted(cdir.glob("*/*.json")):
            sample_id = layout_path.stem
            render = RENDERS_DIR / f"{sample_id}.mp4"
            if not render.exists():
                print(f"[skip] no render for {sample_id}", file=sys.stderr)
                continue
            param = layout_path.parent.name
            out.append({
                "sample_id": sample_id,
                "component_type": ctype,
                "varying_parameter": param,
                "layout_path": layout_path,
                "render_path": render,
            })
    return out


# =====================================================================
#  Tracking
# =====================================================================


def run_tracker(sample: Dict[str, Any], output_dir: Path,
                skip_existing: bool = True, verbose: bool = False) -> Path:
    tracks_path = output_dir / f"{sample['sample_id']}_tracks.json"
    if skip_existing and tracks_path.exists():
        return tracks_path

    cfg = TrackingConfig(
        mode="contour",
        layout_json=str(sample["layout_path"]),
        output_json=True,
        output_dir=str(output_dir),
        verbose=verbose,
    )
    tracker = ContourTracker(cfg)
    tracker.track_video(str(sample["render_path"]))
    if not tracks_path.exists():
        # ContourTracker writes <video_stem>_tracks.json; rename if needed.
        produced = output_dir / f"{sample['render_path'].stem}_tracks.json"
        if produced.exists() and produced != tracks_path:
            produced.rename(tracks_path)
    return tracks_path


# =====================================================================
#  CSV writer
# =====================================================================


CSV_FIELDS = [
    "sample_id", "component_type", "varying_parameter",
    "n_frames_tracked",
    "gt_motion_type", "pred_motion_type", "motion_type_score",
    "gt_direction", "pred_direction", "direction_score",
    "gt_anim_duration_s", "pred_anim_duration_s",
    "anim_duration_abs_err_s", "anim_duration_score",
    "gt_comp_duration_s", "pred_comp_duration_s",
    "comp_duration_abs_err_s", "comp_duration_score",
    "has_animation",
    "error",
]


def evaluation_to_row(sample: Dict[str, Any],
                      ev: Optional[SampleEvaluation],
                      err: Optional[str]) -> Dict[str, Any]:
    if ev is None:
        return {
            "sample_id": sample["sample_id"],
            "component_type": sample["component_type"],
            "varying_parameter": sample["varying_parameter"],
            "error": err or "unknown",
        }
    gt = ev.ground_truth
    return {
        "sample_id": sample["sample_id"],
        "component_type": sample["component_type"],
        "varying_parameter": sample["varying_parameter"],
        "n_frames_tracked": ev.n_frames_tracked,
        "gt_motion_type": gt.motion_type,
        "pred_motion_type": ev.motion_type.predicted,
        "motion_type_score": ev.motion_type.score,
        "gt_direction": gt.direction,
        "pred_direction": ev.direction.predicted,
        "direction_score": ev.direction.score,
        "gt_anim_duration_s": gt.animation_duration_s,
        "pred_anim_duration_s": ev.animation_duration.predicted_s,
        "anim_duration_abs_err_s": ev.animation_duration.abs_error_s,
        "anim_duration_score": ev.animation_duration.score,
        "gt_comp_duration_s": ev.component_duration.gt_s,
        "pred_comp_duration_s": ev.component_duration.predicted_s,
        "comp_duration_abs_err_s": ev.component_duration.abs_error_s,
        "comp_duration_score": ev.component_duration.score,
        "has_animation": gt.has_animation,
        "error": "",
    }


# =====================================================================
#  Main loop
# =====================================================================


def run(component_types: List[str], output_dir: Path,
        skip_tracking: bool, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(component_types)
    print(f"[run] discovered {len(samples)} samples across {component_types}")

    evaluations: List[SampleEvaluation] = []
    rows: List[Dict[str, Any]] = []
    per_sample_dump: List[Dict[str, Any]] = []

    for sample in samples:
        sid = sample["sample_id"]
        print(f"\n[{sid}] tracking + scoring...")
        try:
            tracks_path = run_tracker(
                sample, output_dir,
                skip_existing=skip_tracking,
                verbose=verbose,
            )
            ev = evaluate_sample(
                layout_path=sample["layout_path"],
                tracks_json_path=tracks_path,
                sample_id=sid,
            )
            evaluations.append(ev)
            rows.append(evaluation_to_row(sample, ev, None))
            per_sample_dump.append(evaluation_to_dict(ev))
            print(
                f"  GT:   type={ev.ground_truth.motion_type:<10} "
                f"dir={ev.ground_truth.direction:<10} "
                f"anim={ev.ground_truth.animation_duration_s} "
                f"vis={ev.component_duration.gt_s}\n"
                f"  PRED: type={ev.motion_type.predicted:<10} "
                f"dir={ev.direction.predicted:<10} "
                f"anim={ev.animation_duration.predicted_s} "
                f"vis={ev.component_duration.predicted_s}\n"
                f"  scores  type={ev.motion_type.score} dir={ev.direction.score} "
                f"anim={ev.animation_duration.score} comp={ev.component_duration.score}"
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[{sid}] ERROR: {e}", file=sys.stderr)
            print(tb, file=sys.stderr)
            rows.append(evaluation_to_row(sample, None, str(e)))

    # Per-sample CSV
    csv_path = output_dir / "shape_eval_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"\n[run] CSV -> {csv_path}")

    # Per-sample JSON dump (full features)
    dump_path = output_dir / "shape_eval_per_sample.json"
    dump_path.write_text(json.dumps(per_sample_dump, indent=2, default=str))
    print(f"[run] per-sample JSON -> {dump_path}")

    # Aggregate
    summary = aggregate(evaluations)
    summary_path = output_dir / "shape_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[run] summary -> {summary_path}")

    block = summary.get("all", {})
    print("\n=" * 1, "FINAL AGGREGATE", "=" * 30)
    print(f"  N = {block.get('n', 0)}  (tracker_unreliable = {summary.get('tracker_unreliable_n', 0)})")
    if block:
        mt = block.get("motion_type", {})
        dr = block.get("direction", {})
        ad = block.get("animation_duration", {})
        cd = block.get("component_duration", {})
        print(f"  Motion type accuracy   : {mt.get('accuracy', float('nan')):.3f}")
        print(f"  Direction accuracy     : {dr.get('accuracy', float('nan')):.3f}")
        print(f"  Anim duration MAE (s)  : {ad.get('mae_s', float('nan')):.3f}  (score={ad.get('score', float('nan')):.3f})")
        print(f"  Comp duration MAE (s)  : {cd.get('mae_s', float('nan')):.3f}  (score={cd.get('score', float('nan')):.3f})")
        print("  Confusion (gt -> pred):")
        for gt, row in mt.get("confusion", {}).items():
            print(f"    {gt:<10} -> " + ", ".join(f"{p}={c}" for p, c in row.items()))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--component-types", nargs="+",
        default=["shape"],
        choices=["shape", "image", "text"],
        help="Which component categories to evaluate (default: shape)",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "results" / "motion_gen_eval" / "shape",
        help="Where to write tracker JSONs, the metrics CSV, and the summary",
    )
    p.add_argument(
        "--skip-tracking", action="store_true",
        help="Reuse existing *_tracks.json instead of re-running the tracker",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(
        component_types=args.component_types,
        output_dir=args.output_dir,
        skip_tracking=args.skip_tracking,
        verbose=args.verbose,
    )
