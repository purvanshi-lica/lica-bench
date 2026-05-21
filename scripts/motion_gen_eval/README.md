# motion-gen-eval

Tracker + motion-correctness evaluation for animated graphic-design videos.
Given a LICA layout JSON (the ground-truth animation spec) and a rendered
video, this package classifies each animated component along four axes —
**motion type**, **motion direction**, **animation duration**, and
**component duration** — and reports a per-axis score plus an aggregate.

The pipeline is used to benchmark video-generation models (e.g. Veo,
Sora) against the LICA reference renderer.

## Layout

```
src/motion_gen_eval/                # importable Python package
├── __init__.py
├── motion_metrics.py               # per-sample metric computation (the core)
├── layout.py                       # LICA layout JSON parser + polygon IoU + matching
├── frame_detector.py               # per-frame YOLO-OBB + spatial polygon-IoU matching
├── contour_tracker.py              # background-subtraction tracker (no model needed)
├── layout_tracker.py               # OpenCV trackers initialised from layout positions
├── video_io.py                     # video reading / URL fetching helpers
├── metrics.py                      # extra layout-fidelity metrics
└── config.py                       # TrackingConfig dataclass

scripts/motion_gen_eval/            # CLI runners (this directory)
├── README.md                       # this file
├── eval_video.py                   # single-video CLI (--motion-eval / --evaluate)
├── run_common_motion_eval.py       # full sweep across <model>/<dataset>
└── run_shape_eval.py               # contour-tracker sweep over shape data
```

## Install

From the repository root:

```bash
pip install -e ".[motion-gen-eval]"
```

This installs `numpy`, `opencv-python`, `ultralytics`, `shapely`, `scipy`,
and `requests`.

If you cannot install `shapely` or `scipy`, the package still works:
`motion_gen_eval.layout.polygon_iou` falls back to axis-aligned bounding-box
IoU and `match_detections_to_components` falls back to greedy matching.

The runner scripts also work without an editable install — each script
prepends `<repo_root>/src` to `sys.path` so the `motion_gen_eval` package
is importable as long as the repository is checked out locally.

## Download the YOLO-OBB checkpoint

The `full_layout` evaluator needs a YOLO11x-OBB checkpoint trained on
LICA component categories (`IMAGE`, `TEXT`).

Download URL: [yolo_ckpt.pt](https://storage.googleapis.com/lica-assets/websites/blog/yolo11xOBB-obb80_best_f881f849-5985-4a45-bb26-6d6247118262.pt)

Default expected path:

```
ckpt/yolo11xOBB-obb80_best.pt        # at the repo root
```

## Download the evaluation data

Download URL: [data.zip](https://storage.googleapis.com/lica-assets/websites/blog/motion_gen_eval_data.zip)

Unzip somewhere convenient — by default the runner looks in
`<repo>/data/motion_gen_eval/`, but you can point at any location with
`--data-root /abs/path/to/motion_gen_eval`. Inside that root we expect:

```
<data-root>/
├── all_full_layout/
│   ├── manifest.jsonl                 # one JSON line per sample
│   ├── layouts/<sample_id>.json       # LICA layout JSON
│   └── renders/<sample_id>.mp4        # ground-truth render
│
├── all_single_components/
│   ├── manifest.jsonl
│   ├── layouts/<sample_id>.json
│   └── renders/<sample_id>.mp4
│
└── generated_videos/                  # model outputs you want to evaluate
    └── <model>/                       # e.g. sora2, veo3.1
        ├── single_components/<sample_id>.mp4
        └── full_layout/<sample_id>.mp4
```

`data/` is gitignored at the repo root, so if you unzip in-tree the
contents will not land in version control.

### Manifest format

`all_full_layout/manifest.jsonl` — one JSON object per line. Fields read
by `discover_full_jobs`:

```json
{
  "sample_id": "0av452xBKWsVWrPWZVM5",
  "source_component_ids": ["0-2", "0-3", "0-4", "0-5", "0-6", "0-7"],
  "animated_component_types": ["TEXT", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT"],
  "canvas": [1920, 1080],
  "total_duration": 5.0,
  "num_leaves": 9
}
```

| Field | Required | Notes |
|---|---|---|
| `sample_id` | yes | Join key with `layouts/<sid>.json` and `renders/<sid>.mp4`. |
| `source_component_ids` | yes | Component IDs to evaluate (must exist in the layout JSON). |
| `animated_component_types` | yes | Parallel list of `IMAGE` / `TEXT` / etc. Only `IMAGE` and `TEXT` are detectable by the YOLO checkpoint; other types are kept in the summary but skipped during tracking. |
| `render_path` | optional | If absent, GT renders auto-resolve from `<data-root>/all_full_layout/renders/<sample_id>.mp4`. When present, interpreted relative to `--data-root`, then to the repo root, then absolute. |
| `canvas`, `total_duration`, `num_leaves` | optional | Informational only. |

`all_single_components/manifest.jsonl` — fields read by
`discover_single_jobs`:

| Field | Required | Notes |
|---|---|---|
| `sample_id` | yes | Join key. |
| `source_component_id` | yes | The single animated component. |
| `component_family` (or `component_type`) | optional | `IMAGE`, `TEXT`, `GROUP`, … — informational only; the contour tracker doesn't need it. |
| `render_path` | optional | Same auto-resolution as above, falling back to `<data-root>/all_single_components/renders/<sample_id>.mp4`. |

(Other fields like `layout_json`, `final_frame_path`,
`track1_prompt_path` etc. are ignored by the runner.)

### Layout JSON schema

`parse_layout` (in `src/motion_gen_eval/layout.py`) accepts two shapes:

**Flat form**:

```json
{
  "width": 1080,
  "height": 1920,
  "duration": 5.0,
  "background": "#ffffff",
  "components": [
    {
      "type": "TEXT",
      "id": "0-3",
      "left": 120, "top": 240, "width": 800, "height": 160,
      "transform": "rotate(-12.5deg)",
      "opacity": 1.0,
      "text": "Hello"
    }
  ]
}
```

**Nested LICA form** (with `layout_metadata` and CSS-style `style` dicts):

```json
{
  "layout_metadata": { "width": "1920px", "height": "1080px" },
  "layout_config": {
    "duration": 5.0,
    "style": { "background": "#000" },
    "components": [
      {
        "type": "GROUP", "id": "0",
        "style": { "left": "0px", "top": "0px", "width": "1920px", "height": "1080px" },
        "components": [
          {
            "type": "TEXT", "id": "0-3",
            "style": {
              "left": "120px", "top": "240px", "width": "800px", "height": "160px",
              "transform": "rotate(-12.5deg)", "opacity": 1.0
            },
            "text": "Hello"
          }
        ]
      }
    ]
  }
}
```

Rules enforced by the parser:

* `id` (string) must match the IDs you list in the manifest's
  `source_component_id(s)`. Convention: `"<group>-<index>"`.
* `type`: `IMAGE` or `TEXT` to be tracked by YOLO; `GROUP` /
  `TEXT_NEW` / etc. are parsed and flattened (children inherit parent
  offsets via `_collect_components`).
* Geometry (`left`, `top`, `width`, `height`) accepts numbers or
  `"<n>px"` strings.
* `transform` only honours `rotate(<deg>deg)` and
  `translate(<x>px, <y>px)` — anything else is ignored (no scale/skew).
* `duration` (seconds) is the GT animation duration scored by
  `animation_duration.score`, so it needs to be accurate.
* Canvas size (`width`/`height`, or `layout_metadata.width`/`height`)
  only needs to be **proportional** to the rendered video — components
  are auto-rescaled to the actual video resolution via
  `Layout.scaled_to`.

### Prompts (optional)

The eval pipeline only consumes layouts + videos + manifests; **prompts
are not required** to run the eval. If you want the directory to be a
fully reproducible benchmark release (so somebody else can regenerate
the sora2 / veo3.1 outputs from the same prompts), drop the per-sample
text prompts alongside, e.g.:

```
<data-root>/all_full_layout/prompts/<sample_id>.txt          # optional
<data-root>/all_single_components/prompts/<sample_id>.txt    # optional
```

These files are not read by `run_common_motion_eval.py`.

## Quick start

### Single-video evaluation

```bash
python scripts/motion_gen_eval/eval_video.py path/to/video.mp4 \
    --layout path/to/layout.json \
    --motion-eval \
    --motion-eval-output out_motion.json
```

`eval_video.py` supports three tracking modes (`--mode`):

* `layout-init` (default) — uses layout JSON positions to initialise an
  OpenCV tracker (DaSiamRPN by default). No model needed.
* `contour` — background-subtraction tracker for single-component
  videos on a uniform background. No model needed.
* `yolo` — per-frame YOLO-OBB + spatial polygon-IoU matching to the
  layout components. Needs the checkpoint above.

### Common-set sweep (multi-model, multi-dataset)

```bash
# Everything (sora2, veo3.1, gt × single_components, full_layout)
python scripts/motion_gen_eval/run_common_motion_eval.py \
    --data-root /abs/path/to/motion_gen_eval

# Just one slice
python scripts/motion_gen_eval/run_common_motion_eval.py \
    --data-root /abs/path/to/motion_gen_eval \
    --models sora2 --datasets full_layout

# Force re-tracking
python scripts/motion_gen_eval/run_common_motion_eval.py --no-skip-tracking

# Smoke run
python scripts/motion_gen_eval/run_common_motion_eval.py --max-per-dataset 3
```

Outputs land under `results/motion_gen_eval/common/<model>/<dataset>/`.
For ground-truth (`--models gt`) the renders are auto-resolved from
`<data-root>/{all_full_layout,all_single_components}/renders/<sample_id>.mp4`;
override with `--gt-videos-root <path>` if you keep them elsewhere.

### Shape-side sweep

```bash
python scripts/motion_gen_eval/run_shape_eval.py
python scripts/motion_gen_eval/run_shape_eval.py --skip-tracking
```

Uses the contour tracker; expects shape layouts under
`data/motion_gen_eval/<type>/<param>/<sample>.json` and renders under
`data/motion_gen_eval/renders/<sample>.mp4`.

## Python API

```python
from motion_gen_eval.motion_metrics import (
    evaluate_sample,
    evaluation_to_dict,
)

ev = evaluate_sample(
    layout_path="data/motion_gen_eval/all_full_layout/layouts/0av452xBKWsVWrPWZVM5.json",
    tracks_json_path="results/motion_gen_eval/common/veo3.1/full_layout/tracks/0av452xBKWsVWrPWZVM5_tracks.json",
    component_id="0-3",
)
print(evaluation_to_dict(ev))
```

`SampleEvaluation` exposes:

| Field | Type | Notes |
|---|---|---|
| `motion_type.predicted` | `str` | one of `scrapbook / fade / pop / wiggle / breathe / rotate / pan / static / unknown` |
| `motion_type.score` | `float` | exact-match plus a partial-credit table for related transitions |
| `direction.predicted` | `str` | 8-way compass + `clockwise` / `anticlockwise` / `none` |
| `direction.score` | `float` | LICA-canonical: only `pan`-family (left/right) and `rotate`-family (CW/CCW) carry directional GT |
| `animation_duration.score` | `float` | `1 - min(rel_error, 1)` (or hard-tolerance match for short animations) |
| `component_duration.score` | `float` | same scoring rule applied to longest contiguous tracked interval |
| `tracking_quality.presence_frac` | `float` | fraction of total video frames with a detection |
| `tracking_quality.is_reliable` | `bool` | heuristic: `presence_frac >= 0.3` |

## How the metrics are computed (high level)

`motion_gen_eval.motion_metrics.classify_motion_type` runs a hand-tuned
decision tree over per-frame OBB centroids, scale, rotation, and (proxy)
opacity:

* `presence < 5 %` → degenerate static.
* fade detector (contour-mode and YOLO-mode variants).
* "endpoints hot, middle cold" transient gate → `scrapbook` / `pop` /
  `rotate` (tumble/roll entries).
* Sustained signatures: `rotate` (large angular drift), `wiggle`
  (oscillating centre), `breathe` (oscillating scale), `pan` (sustained
  monotonic translation).
* Presence-based rescue and sustained-motion fallback for chains the
  rule chain otherwise drops to `static`.

`classify_direction` derives the dominant direction from the entry-third
displacement (transients) or the net displacement (pan), and from the
sign of the period-90 unwrapped OBB angle (rotate). The output is mapped
to `_direction_label(angle)` (8-way compass) or `clockwise` /
`anticlockwise`.

`estimate_animation_duration` extracts the time from animation onset to
settling from the smoothed motion-energy curve. `estimate_component_duration`
returns the longest contiguous run of presence frames.

See the docstrings in `src/motion_gen_eval/motion_metrics.py` for the
full, heavily-commented rationale (including the LICA-grammar mapping
table that explains why many raw labels collapse onto `scrapbook`).

## Reproducing the paper numbers

```bash
# 1. tracker output for every (model, dataset) pair
python scripts/motion_gen_eval/run_common_motion_eval.py

# 2. (optional) shape-side eval
python scripts/motion_gen_eval/run_shape_eval.py
```

Aggregates land at `results/motion_gen_eval/common/<model>/<dataset>/<dataset>_summary.json`
and `results/motion_gen_eval/common/<model>/<dataset>/<dataset>_results.csv`.
