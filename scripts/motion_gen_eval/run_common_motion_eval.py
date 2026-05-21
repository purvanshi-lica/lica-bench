#!/usr/bin/env python3
"""Run motion-correctness evaluation across all generated videos.

Iterates ``<data-root>/generated_videos/<model>/<dataset>/`` and the GT
renders in ``<data-root>/{all_full_layout,all_single_components}/renders/``,
and produces:

* per-video tracker JSONs in ``<output-dir>/<model>/<dataset>/tracks/``
* per-video motion-metric JSONs (one per evaluated component) in
  ``<output-dir>/<model>/<dataset>/metrics/``
* a flat CSV ``<dataset>_results.csv``
* an aggregate ``<dataset>_summary.json``

For ``single_components`` we use the no-YOLO :class:`ContourTracker` (the
videos are rendered on a known background, single object). For
``full_layout`` we use the YOLO-OBB checkpoint at
``ckpt/yolo11xOBB-obb80_best_*.pt`` and run per-frame YOLO ``predict``
with spatial polygon-IoU matching against the layout components listed
in ``<data-root>/all_full_layout/manifest.jsonl``. There is no temporal
multi-object tracker (ByteTrack/BoT-SORT are no longer used).

Usage::

    # Everything (all discovered models + GT, both datasets)
    python run_common_motion_eval.py --data-root /path/to/motion_gen_eval

    # Only the Sora 2 single-component videos
    python run_common_motion_eval.py --models sora2 --datasets single_components

    # Force re-tracking even if a *_tracks.json already exists
    python run_common_motion_eval.py --no-skip-tracking

    # Limit to a handful of samples for a smoke run
    python run_common_motion_eval.py --max-per-dataset 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_SRC = REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))


def _json_safe(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with ``None`` so the output is
    strictly-valid JSON (RFC 8259 forbids NaN / Infinity)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _dump_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(_json_safe(obj), indent=2, default=str,
                               allow_nan=False))

# All eval-data paths derive from DATA_ROOT and are re-bound in `main()`
# whenever the user passes --data-root. The defaults match the original
# in-repo layout (`<repo>/data/motion_gen_eval/...`), but the data is
# typically held outside the repo and pointed at via --data-root.
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "motion_gen_eval"
DATA_ROOT = DEFAULT_DATA_ROOT
DEFAULT_VIDEOS_ROOT = DATA_ROOT / "generated_videos"
VIDEOS_ROOT = DEFAULT_VIDEOS_ROOT  # overridden by --videos-root at CLI parse time
SINGLE_LAYOUTS = DATA_ROOT / "all_single_components" / "layouts"
SINGLE_MANIFEST = DATA_ROOT / "all_single_components" / "manifest.jsonl"
SINGLE_RENDERS = DATA_ROOT / "all_single_components" / "renders"
FULL_LAYOUTS = DATA_ROOT / "all_full_layout" / "layouts"
FULL_MANIFEST = DATA_ROOT / "all_full_layout" / "manifest.jsonl"
FULL_RENDERS = DATA_ROOT / "all_full_layout" / "renders"
DEFAULT_YOLO_CKPT = REPO_ROOT / "ckpt" / "yolo11xOBB-obb80_best.pt"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "motion_gen_eval" / "common"

MODELS = ("sora2", "veo3.1", "gt")
DATASETS = ("single_components", "full_layout")


def _apply_data_root(data_root: Path) -> None:
    """Rebind all DATA_ROOT-derived module globals.

    Called from ``main()`` after argument parsing so that --data-root,
    --videos-root, etc. are reflected in the discovery helpers below.
    """
    global DATA_ROOT, DEFAULT_VIDEOS_ROOT, VIDEOS_ROOT
    global SINGLE_LAYOUTS, SINGLE_MANIFEST, SINGLE_RENDERS
    global FULL_LAYOUTS, FULL_MANIFEST, FULL_RENDERS
    DATA_ROOT = data_root
    DEFAULT_VIDEOS_ROOT = DATA_ROOT / "generated_videos"
    VIDEOS_ROOT = DEFAULT_VIDEOS_ROOT
    SINGLE_LAYOUTS = DATA_ROOT / "all_single_components" / "layouts"
    SINGLE_MANIFEST = DATA_ROOT / "all_single_components" / "manifest.jsonl"
    SINGLE_RENDERS = DATA_ROOT / "all_single_components" / "renders"
    FULL_LAYOUTS = DATA_ROOT / "all_full_layout" / "layouts"
    FULL_MANIFEST = DATA_ROOT / "all_full_layout" / "manifest.jsonl"
    FULL_RENDERS = DATA_ROOT / "all_full_layout" / "renders"

# Only these layout types are detectable by the trained YOLO checkpoint
# (which has classes {0: text, 1: image}). GROUP boxes are typically
# wrappers around nested components and are skipped for full-layout mode
# but still kept in the per-video evaluation summary so we can report
# coverage.
YOLO_DETECTABLE_TYPES = {"IMAGE", "TEXT"}


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a JSONL manifest into a {sample_id: entry} dict."""
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = entry.get("sample_id")
        if sid:
            out[sid] = entry
    return out


# ---------------------------------------------------------------------------
# Job discovery
# ---------------------------------------------------------------------------


def _video_sample_id(path: Path) -> str:
    """Sample id is just the file stem (matches the layout JSON stem)."""
    return path.stem


def _resolve_gt_video(entry: Dict[str, Any], gt_videos_root: Optional[Path],
                      default_renders_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve the ground-truth render video for a manifest entry.

    Checks, in order:
    1. ``gt_videos_root / <sample_id>.mp4`` if gt_videos_root is provided
    2. ``default_renders_dir / <sample_id>.mp4`` (typically
       ``<data_root>/<dataset>/renders/``)
    3. The ``render_path`` field relative to ``REPO_ROOT``
    4. The ``render_path`` field relative to ``DATA_ROOT``
    5. The ``render_path`` field as-is (absolute)
    """
    sid = entry.get("sample_id", "")
    render_rel = entry.get("render_path", "")

    if gt_videos_root:
        candidate = gt_videos_root / f"{sid}.mp4"
        if candidate.exists():
            return candidate

    if default_renders_dir:
        candidate = default_renders_dir / f"{sid}.mp4"
        if candidate.exists():
            return candidate

    if render_rel:
        for base in (REPO_ROOT, DATA_ROOT):
            candidate = base / render_rel
            if candidate.exists():
                return candidate
        candidate = Path(render_rel)
        if candidate.exists():
            return candidate

    return None


def discover_single_jobs(model: str, max_per_dataset: int = 0,
                         gt_videos_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    manifest = _load_manifest(SINGLE_MANIFEST)

    if model == "gt":
        jobs: List[Dict[str, Any]] = []
        for sid, entry in sorted(manifest.items()):
            layout = SINGLE_LAYOUTS / f"{sid}.json"
            if not layout.exists():
                print(f"[discover] gt/single_components: missing layout for {sid}",
                      file=sys.stderr)
                continue
            video = _resolve_gt_video(entry, gt_videos_root,
                                      default_renders_dir=SINGLE_RENDERS)
            if video is None:
                print(f"[discover] gt/single_components: missing render for {sid}",
                      file=sys.stderr)
                continue
            component_id = entry.get("source_component_id")
            component_family = (entry.get("component_family") or
                                entry.get("component_type") or "").upper()
            jobs.append({
                "model": "gt",
                "dataset": "single_components",
                "sample_id": sid,
                "video_path": video,
                "layout_path": layout,
                "components": [{"id": component_id, "type": component_family}],
                "manifest": entry,
            })
            if max_per_dataset and len(jobs) >= max_per_dataset:
                break
        return jobs

    video_dir = VIDEOS_ROOT / model / "single_components"
    if not video_dir.is_dir():
        return []
    jobs = []
    for video in sorted(video_dir.glob("*.mp4")):
        sid = _video_sample_id(video)
        layout = SINGLE_LAYOUTS / f"{sid}.json"
        if not layout.exists():
            print(f"[discover] {model}/single_components: missing layout for {sid}",
                  file=sys.stderr)
            continue
        entry = manifest.get(sid, {})
        component_id = entry.get("source_component_id")
        component_family = (entry.get("component_family") or
                            entry.get("component_type") or "").upper()
        jobs.append({
            "model": model,
            "dataset": "single_components",
            "sample_id": sid,
            "video_path": video,
            "layout_path": layout,
            "components": [{
                "id": component_id,
                "type": component_family,
            }],
            "manifest": entry,
        })
        if max_per_dataset and len(jobs) >= max_per_dataset:
            break
    return jobs


def discover_full_jobs(model: str, max_per_dataset: int = 0,
                       gt_videos_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    manifest = _load_manifest(FULL_MANIFEST)

    if model == "gt":
        jobs: List[Dict[str, Any]] = []
        for sid, entry in sorted(manifest.items()):
            layout = FULL_LAYOUTS / f"{sid}.json"
            if not layout.exists():
                print(f"[discover] gt/full_layout: missing layout for {sid}",
                      file=sys.stderr)
                continue
            video = _resolve_gt_video(entry, gt_videos_root,
                                      default_renders_dir=FULL_RENDERS)
            if video is None:
                print(f"[discover] gt/full_layout: missing render for {sid}",
                      file=sys.stderr)
                continue
            comp_ids: List[str] = list(entry.get("source_component_ids") or [])
            comp_types: List[str] = list(entry.get("animated_component_types") or [])
            track_specs: List[str] = []
            eval_components: List[Dict[str, str]] = []
            for cid, ctype in zip(comp_ids, comp_types):
                entry_dict = {"id": cid, "type": ctype}
                eval_components.append(entry_dict)
                if ctype.upper() in YOLO_DETECTABLE_TYPES:
                    track_specs.append(f"{ctype.upper()} {cid}")
            if not eval_components:
                print(f"[discover] gt/full_layout: no animated components for {sid}",
                      file=sys.stderr)
            jobs.append({
                "model": "gt",
                "dataset": "full_layout",
                "sample_id": sid,
                "video_path": video,
                "layout_path": layout,
                "components": eval_components,
                "track_specs": track_specs,
                "manifest": entry,
            })
            if max_per_dataset and len(jobs) >= max_per_dataset:
                break
        return jobs

    video_dir = VIDEOS_ROOT / model / "full_layout"
    if not video_dir.is_dir():
        return []
    jobs = []
    for video in sorted(video_dir.glob("*.mp4")):
        sid = _video_sample_id(video)
        layout = FULL_LAYOUTS / f"{sid}.json"
        if not layout.exists():
            print(f"[discover] {model}/full_layout: missing layout for {sid}",
                  file=sys.stderr)
            continue
        entry = manifest.get(sid, {})
        comp_ids: List[str] = list(entry.get("source_component_ids") or [])
        comp_types: List[str] = list(entry.get("animated_component_types") or [])
        track_specs: List[str] = []
        eval_components: List[Dict[str, str]] = []
        for cid, ctype in zip(comp_ids, comp_types):
            entry_dict = {"id": cid, "type": ctype}
            eval_components.append(entry_dict)
            if ctype.upper() in YOLO_DETECTABLE_TYPES:
                track_specs.append(f"{ctype.upper()} {cid}")
        if not eval_components:
            print(f"[discover] {model}/full_layout: no animated components for {sid}",
                  file=sys.stderr)
        jobs.append({
            "model": model,
            "dataset": "full_layout",
            "sample_id": sid,
            "video_path": video,
            "layout_path": layout,
            "components": eval_components,
            "track_specs": track_specs,
            "manifest": entry,
        })
        if max_per_dataset and len(jobs) >= max_per_dataset:
            break
    return jobs


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def run_contour(job: Dict[str, Any], tracks_dir: Path,
                skip_existing: bool = True, verbose: bool = False) -> Path:
    """Run the no-YOLO contour tracker for a single-component video."""
    from motion_gen_eval.config import TrackingConfig
    from motion_gen_eval.contour_tracker import ContourTracker

    sid = job["sample_id"]
    out_path = tracks_dir / f"{sid}_tracks.json"
    if skip_existing and out_path.exists():
        return out_path

    cfg = TrackingConfig(
        mode="contour",
        layout_json=str(job["layout_path"]),
        output_json=True,
        output_dir=str(tracks_dir),
        verbose=verbose,
    )
    tracker = ContourTracker(cfg)
    tracker.track_video(str(job["video_path"]))

    # The tracker writes <video_stem>_tracks.json. Sample id == video stem
    # in our naming so the file is already at out_path.
    if not out_path.exists():
        produced = tracks_dir / f"{job['video_path'].stem}_tracks.json"
        if produced.exists() and produced != out_path:
            produced.rename(out_path)
    return out_path


class _YOLOFrameRunner:
    """Lazy-loaded per-frame YOLO + spatial-IoU runner reused across jobs.

    Replaces the previous ByteTrack-based ``_YOLORunner``. For each video,
    we run YOLO ``predict()`` on every frame and match each frame's
    detections to layout components by polygon IoU. Detections are
    labelled with the layout component_id so ``motion_metrics`` can
    look up the trajectory directly.
    """

    def __init__(self, weights: str, *, imgsz: int = 1280,
                 conf: float, iou: float,
                 match_iou_thresh: float,
                 device: str = "", verbose: bool = False):
        from motion_gen_eval.frame_detector import YoloFrameDetector

        self._detector = YoloFrameDetector(
            weights=weights,
            imgsz=imgsz,
            conf=conf,
            nms_iou=iou,
            match_iou_thresh=match_iou_thresh,
            device=device,
            verbose=verbose,
        )

    def load(self) -> None:
        self._detector.load()

    def run(self, job: Dict[str, Any], tracks_dir: Path,
            skip_existing: bool = True) -> Path:
        from motion_gen_eval.frame_detector import find_animated_components

        sid = job["sample_id"]
        out_path = tracks_dir / f"{sid}_tracks.json"
        if skip_existing and out_path.exists():
            return out_path

        # Build the (possibly augmented) component list to use for spatial
        # matching. We start with the manifest-listed eval components and
        # then merge in every other detectable IMAGE/TEXT in the layout
        # (so non-target detections are absorbed by the right component
        # rather than competing for a target slot).
        manifest_components = job.get("components") or []
        manifest_detectable = [
            {"id": c["id"], "detect_id": c["id"], "type": (c.get("type") or "").upper()}
            for c in manifest_components
            if isinstance(c, dict) and (c.get("type") or "").upper() in YOLO_DETECTABLE_TYPES
        ]
        if not manifest_detectable:
            print(f"[yolo] {sid}: no IMAGE/TEXT components to track, skipping")
            return out_path  # may not exist

        try:
            extra = find_animated_components(job["layout_path"])
        except Exception:
            extra = []
        seen_ids = {c["id"] for c in manifest_detectable}
        seen_ids.update(c["detect_id"] for c in manifest_detectable)
        targets = list(manifest_detectable)
        for c in extra:
            if c["id"] in seen_ids or c["detect_id"] in seen_ids:
                continue
            targets.append(c)
            seen_ids.add(c["id"])
            seen_ids.add(c["detect_id"])

        tracks_data = self._detector.detect_video(
            video_path=job["video_path"],
            layout_path=job["layout_path"],
            components=targets,
        )

        out_path.write_text(json.dumps(_json_safe(tracks_data), indent=2,
                                       default=str, allow_nan=False))
        return out_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _row(model: str, dataset: str, sid: str, comp: Optional[Dict[str, Any]],
         ev=None, err: Optional[str] = None) -> Dict[str, Any]:
    if ev is None:
        return {
            "model": model,
            "dataset": dataset,
            "sample_id": sid,
            "component_id": comp.get("id") if comp else "",
            "component_type": comp.get("type") if comp else "",
            "error": err or "",
        }
    gt = ev.ground_truth
    return {
        "model": model,
        "dataset": dataset,
        "sample_id": sid,
        "component_id": ev.component_id or (comp.get("id") if comp else ""),
        "component_type": comp.get("type") if comp else "",
        "n_frames_total": ev.tracking_quality.n_frames_total,
        "n_frames_tracked": ev.tracking_quality.n_frames_tracked,
        "presence_frac": ev.tracking_quality.presence_frac,
        "tracker_reliable": ev.tracking_quality.is_reliable,
        "gt_motion_type_raw": gt.motion_type_raw,
        "gt_motion_type": gt.motion_type,
        "pred_motion_type": ev.motion_type.predicted,
        "motion_type_score": ev.motion_type.score,
        "gt_direction_raw": gt.direction_raw,
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


CSV_FIELDS = [
    "model", "dataset", "sample_id", "component_id", "component_type",
    "n_frames_total", "n_frames_tracked", "presence_frac", "tracker_reliable",
    "gt_motion_type_raw", "gt_motion_type", "pred_motion_type", "motion_type_score",
    "gt_direction_raw", "gt_direction", "pred_direction", "direction_score",
    "gt_anim_duration_s", "pred_anim_duration_s", "anim_duration_abs_err_s",
    "anim_duration_score",
    "gt_comp_duration_s", "pred_comp_duration_s", "comp_duration_abs_err_s",
    "comp_duration_score",
    "has_animation",
    "error",
]


def evaluate_job(job: Dict[str, Any], tracks_path: Path, metrics_dir: Path,
                 ) -> Tuple[List[Dict[str, Any]], List[Any], List[str]]:
    """Run motion_metrics on each evaluation component for one video.

    Returns (rows, evaluations, ev_component_types). The latter two lists
    are parallel and contain only the successfully evaluated components
    (one entry per call to ``evaluate_sample`` that didn't raise).
    """
    from motion_gen_eval.motion_metrics import evaluate_sample, evaluation_to_dict

    sid = job["sample_id"]
    rows: List[Dict[str, Any]] = []
    evaluations: List[Any] = []
    ev_types: List[str] = []

    if not tracks_path.exists():
        rows.append(_row(job["model"], job["dataset"], sid, None,
                         err="no_tracks_json"))
        return rows, evaluations, ev_types

    components = job.get("components") or [None]

    for comp in components:
        if isinstance(comp, dict):
            cid = comp.get("id")
            ctype = comp.get("type") or ""
        else:
            cid = comp
            ctype = ""
            comp = {"id": cid, "type": ctype}

        try:
            ev = evaluate_sample(
                layout_path=str(job["layout_path"]),
                tracks_json_path=str(tracks_path),
                sample_id=sid,
                component_id=cid,
            )
            rows.append(_row(job["model"], job["dataset"], sid, comp, ev=ev))
            evaluations.append(ev)
            ev_types.append(ctype)
            comp_tag = (cid or "auto").replace("/", "_")
            out = metrics_dir / f"{sid}__{comp_tag}_motion.json"
            _dump_json(evaluation_to_dict(ev), out)
        except Exception as e:  # pragma: no cover - defensive
            tb = traceback.format_exc()
            print(f"[eval] {sid} cid={cid}: ERROR {e}\n{tb}", file=sys.stderr)
            rows.append(_row(job["model"], job["dataset"], sid, comp,
                             err=str(e)))
    return rows, evaluations, ev_types


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_block(evaluations: List[Any]) -> Dict[str, Any]:
    """Per-axis aggregate matching ``motion_metrics.aggregate`` behaviour."""
    import numpy as np

    if not evaluations:
        return {"n": 0}

    def _mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(xs)) if xs else float("nan")

    motion_scores = [e.motion_type.score for e in evaluations]
    dir_scores = [e.direction.score for e in evaluations]
    anim_scores = [e.animation_duration.score for e in evaluations]
    comp_scores = [e.component_duration.score for e in evaluations]
    anim_errs = [e.animation_duration.abs_error_s for e in evaluations
                 if e.animation_duration.abs_error_s is not None]
    comp_errs = [e.component_duration.abs_error_s for e in evaluations
                 if e.component_duration.abs_error_s is not None]
    presence = [e.tracking_quality.presence_frac for e in evaluations]

    confusion: Dict[str, Dict[str, int]] = {}
    for e in evaluations:
        gt = e.ground_truth.motion_type or "unknown"
        pr = e.motion_type.predicted
        confusion.setdefault(gt, {}).setdefault(pr, 0)
        confusion[gt][pr] += 1

    return {
        "n": len(evaluations),
        "tracker": {
            "mean_presence_frac": _mean(presence),
            "n_reliable": sum(1 for e in evaluations if e.tracking_quality.is_reliable),
        },
        "motion_type": {
            "accuracy": _mean(motion_scores),
            "confusion": confusion,
        },
        "direction": {"accuracy": _mean(dir_scores)},
        "animation_duration": {
            "score": _mean(anim_scores),
            "mae_s": _mean(anim_errs) if anim_errs else float("nan"),
        },
        "component_duration": {
            "score": _mean(comp_scores),
            "mae_s": _mean(comp_errs) if comp_errs else float("nan"),
        },
    }


def write_summary(model: str, dataset: str, evaluations: List[Any],
                  ev_component_types: List[str],
                  rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    """Write the per-(model,dataset) aggregate summary.

    ``evaluations`` and ``ev_component_types`` are parallel lists; only
    successful evaluations contribute to aggregates.
    """
    by_type: Dict[str, Dict[str, Any]] = {}
    for ct in sorted({c for c in ev_component_types if c}):
        evs = [
            e for e, t in zip(evaluations, ev_component_types) if t == ct
        ]
        by_type[ct] = aggregate_block(evs)

    summary = {
        "model": model,
        "dataset": dataset,
        "n_videos": len({r["sample_id"] for r in rows}),
        "n_evaluations": len(evaluations),
        "n_evaluation_errors": sum(1 for r in rows if r.get("error")),
        "all": aggregate_block(evaluations),
        "tracker_reliable": aggregate_block(
            [e for e in evaluations if e.tracking_quality.is_reliable]
        ),
        "by_component_type": by_type,
    }
    out = out_dir / f"{dataset}_summary.json"
    _dump_json(summary, out)
    return out


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_dataset(model: str, dataset: str, out_root: Path, *,
                yolo_runner: Optional["_YOLOFrameRunner"] = None,
                skip_tracking: bool = True,
                retrack_ids: Optional[set[str]] = None,
                max_per_dataset: int = 0,
                gt_videos_root: Optional[Path] = None,
                verbose: bool = False) -> None:
    if dataset == "single_components":
        jobs = discover_single_jobs(model, max_per_dataset=max_per_dataset,
                                    gt_videos_root=gt_videos_root)
    else:
        jobs = discover_full_jobs(model, max_per_dataset=max_per_dataset,
                                  gt_videos_root=gt_videos_root)

    print(f"[run] {model}/{dataset}: {len(jobs)} videos")
    if not jobs:
        return

    tracks_dir = out_root / model / dataset / "tracks"
    metrics_dir = out_root / model / dataset / "metrics"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if retrack_ids:
        n_retrack = sum(1 for j in jobs if j["sample_id"] in retrack_ids)
        print(f"[run] {model}/{dataset}: re-tracking {n_retrack} sample(s) "
              f"listed in --retrack-ids-file (others use cached tracks)")

    all_rows: List[Dict[str, Any]] = []
    all_evals: List[Any] = []
    all_ev_types: List[str] = []

    t0 = time.perf_counter()
    for i, job in enumerate(jobs, 1):
        sid = job["sample_id"]
        # Per-job tracking-cache override: if the user supplied an
        # ``--retrack-ids-file``, the listed samples ignore the cached
        # ``*_tracks.json`` and re-run the tracker; everyone else still
        # honours the caller-level ``skip_tracking`` flag.
        skip_for_this_job = skip_tracking and not (retrack_ids and sid in retrack_ids)
        try:
            if dataset == "single_components":
                tracks_path = run_contour(
                    job, tracks_dir,
                    skip_existing=skip_for_this_job, verbose=verbose,
                )
            else:
                if yolo_runner is None:
                    raise RuntimeError("YOLO runner not initialised")
                tracks_path = yolo_runner.run(
                    job, tracks_dir, skip_existing=skip_for_this_job,
                )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[track] {sid}: ERROR {e}\n{tb}", file=sys.stderr)
            all_rows.append(_row(model, dataset, sid, None, err=str(e)))
            continue

        rows, evs, ev_types = evaluate_job(job, tracks_path, metrics_dir)
        all_rows.extend(rows)
        all_evals.extend(evs)
        all_ev_types.extend(ev_types)

        elapsed = time.perf_counter() - t0
        eta = elapsed / max(i, 1) * (len(jobs) - i)
        print(f"[run] {model}/{dataset} {i}/{len(jobs)} {sid} | "
              f"+{len(rows)} eval rows | "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    csv_path = out_root / model / dataset / f"{dataset}_results.csv"
    write_csv(all_rows, csv_path)
    summary_path = write_summary(model, dataset, all_evals, all_ev_types,
                                 all_rows, out_root / model / dataset)
    print(f"[run] {model}/{dataset} CSV     -> {csv_path}")
    print(f"[run] {model}/{dataset} SUMMARY -> {summary_path}")

    s = aggregate_block(all_evals)
    print(f"\n=== {model}/{dataset} aggregate ({s['n']} evals, "
          f"{s['tracker']['n_reliable']} tracker-reliable) ===")
    print(f"  motion type accuracy : {s['motion_type']['accuracy']:.3f}")
    print(f"  direction accuracy   : {s['direction']['accuracy']:.3f}")
    print(f"  anim duration score  : {s['animation_duration']['score']:.3f}  "
          f"MAE={s['animation_duration']['mae_s']:.3f}s")
    print(f"  comp duration score  : {s['component_duration']['score']:.3f}  "
          f"MAE={s['component_duration']['mae_s']:.3f}s")
    print(f"  mean presence frac   : {s['tracker']['mean_presence_frac']:.3f}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["all"],
                   choices=["all"] + list(MODELS),
                   help="Generator outputs to evaluate (default: all)")
    p.add_argument("--datasets", nargs="+", default=["all"],
                   choices=["all"] + list(DATASETS),
                   help="Dataset(s) to process (default: all)")
    p.add_argument("--yolo-weights", default=str(DEFAULT_YOLO_CKPT),
                   help="Path to the YOLO-OBB checkpoint for full_layout")
    p.add_argument("--device", default="",
                   help='YOLO device string ("cuda:0", "cpu", "" for auto)')
    p.add_argument("--imgsz", type=int, default=1280,
                   help="YOLO inference image size (default: 1280)")
    p.add_argument("--conf", type=float, default=0.01,
                   help=("YOLO detection confidence threshold (default: 0.01). "
                         "Spatial matching against the layout makes a strong "
                         "prior, so we keep the threshold low to avoid "
                         "dropping faint components."))
    p.add_argument("--iou", type=float, default=0.6,
                   help="YOLO NMS IoU threshold")
    p.add_argument("--match-iou-thresh", type=float, default=0.03,
                   help=("Min polygon IoU to match a YOLO detection to a "
                         "layout component (full_layout, per-frame mode)"))
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                   help=("Root directory of the eval dataset. Expected layout: "
                         "<data-root>/{all_full_layout,all_single_components}/"
                         "{layouts/,renders/,manifest.jsonl} and "
                         "<data-root>/generated_videos/<model>/<dataset>/. "
                         f"Default: {DEFAULT_DATA_ROOT}"))
    p.add_argument("--videos-root", type=Path, default=None,
                   help=("Root directory containing <model>/<dataset>/*.mp4. "
                         "Defaults to <data-root>/generated_videos."))
    p.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                   help="Where to write tracks / metrics / summaries")
    p.add_argument("--no-skip-tracking", action="store_true",
                   help="Re-run trackers even if a *_tracks.json exists")
    p.add_argument("--retrack-ids-file", type=Path, default=None,
                   help=("Path to a text file with one sample_id per line. "
                         "Listed samples will have their *_tracks.json "
                         "regenerated; everyone else still uses the cached "
                         "tracker output. Pair with the same file passed to "
                         "generate_videos.py --sample-ids-file."))
    p.add_argument("--gt-videos-root", type=Path, default=None,
                   help=("Root directory containing GT render videos. "
                         "When --models includes 'gt', videos are resolved "
                         "from this directory (as <sample_id>.mp4), then "
                         "from <data-root>/<dataset>/renders/, and finally "
                         "from the manifest render_path field."))
    p.add_argument("--max-per-dataset", type=int, default=0,
                   help="If >0, limit each (model,dataset) to N videos "
                        "(useful for smoke-testing)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    _apply_data_root(args.data_root.resolve())
    print(f"[run] data root  : {DATA_ROOT}")
    if args.videos_root is None:
        args.videos_root = DEFAULT_VIDEOS_ROOT

    models = list(MODELS) if "all" in args.models else list(dict.fromkeys(args.models))
    datasets = list(DATASETS) if "all" in args.datasets else list(dict.fromkeys(args.datasets))

    yolo_runner: Optional[_YOLOFrameRunner] = None
    if "full_layout" in datasets:
        if not Path(args.yolo_weights).exists():
            print(f"Error: YOLO checkpoint not found: {args.yolo_weights}",
                  file=sys.stderr)
            sys.exit(1)
        yolo_runner = _YOLOFrameRunner(
            weights=args.yolo_weights,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            match_iou_thresh=args.match_iou_thresh,
            device=args.device,
            verbose=args.verbose,
        )
        yolo_runner.load()

    global VIDEOS_ROOT
    VIDEOS_ROOT = args.videos_root.resolve()
    print(f"[run] videos root: {VIDEOS_ROOT}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    retrack_ids: Optional[set[str]] = None
    if args.retrack_ids_file is not None:
        if not args.retrack_ids_file.exists():
            print(f"Error: --retrack-ids-file not found: {args.retrack_ids_file}",
                  file=sys.stderr)
            sys.exit(1)
        retrack_ids = {
            line.strip()
            for line in args.retrack_ids_file.read_text().splitlines()
            if line.strip()
        }
        print(f"[run] re-track filter: {len(retrack_ids)} sample id(s) from "
              f"{args.retrack_ids_file}")

    gt_videos_root: Optional[Path] = None
    if args.gt_videos_root is not None:
        gt_videos_root = args.gt_videos_root.resolve()
        print(f"[run] GT videos root: {gt_videos_root}")

    GT_RENDER_DIRS = {
        "single_components": "all_single_components/renders",
        "full_layout": "all_full_layout/renders",
    }

    for model in models:
        for dataset in datasets:
            ds_gt_root = None
            if model == "gt" and gt_videos_root:
                ds_gt_root = gt_videos_root / GT_RENDER_DIRS.get(dataset, dataset)
                if not ds_gt_root.is_dir():
                    ds_gt_root = gt_videos_root / dataset
                if not ds_gt_root.is_dir():
                    ds_gt_root = gt_videos_root
            run_dataset(
                model, dataset, args.output_dir,
                yolo_runner=yolo_runner,
                skip_tracking=not args.no_skip_tracking,
                retrack_ids=retrack_ids,
                max_per_dataset=args.max_per_dataset,
                gt_videos_root=ds_gt_root,
                verbose=args.verbose,
            )


if __name__ == "__main__":
    main()
