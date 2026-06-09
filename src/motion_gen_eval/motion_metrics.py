"""Motion-level evaluation metrics for animated design components.

Given:
  - a ground-truth layout JSON (LICA format) describing the intended animation
    of a single component (motion type, direction, animation duration,
    component visible duration), and
  - a tracker output JSON (from contour_tracker / layout_tracker / tracker)
    listing per-frame oriented bounding boxes,

this module computes:

  1. **motion type**  -- predicted label (scrapbook / fade / wiggle / pop /
     breathe / rotate / pan / static / unknown) classified from observed
     trajectory, scale, rotation and opacity signals.
  2. **motion direction** -- 8-way compass label (left / right / up / down /
     up-left ...) plus ``none`` derived from net displacement.
  3. **animation duration**  -- the time (s) from animation onset to the
     component reaching a stable state, measured from the motion-energy curve.
  4. **component duration**  -- the contiguous interval (s) during which the
     component is detected (visible) in the video.

Each sub-metric returns a per-sample ``score`` in [0, 1] plus the raw
prediction so we can build aggregate accuracy / MAE tables for the paper.

Standalone usage::

    python motion_metrics.py \\
        --tracks output/shape__position__bottom_right_tracks.json \\
        --layout data/shape/position/shape__position__bottom_right.json \\
        --output output/shape__position__bottom_right_motion.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
#  Vocabulary
# =====================================================================

# LICA motion-type labels we observe in the dataset (plus ``static``
# which we use whenever the GT has no animations field).
MOTION_TYPES = (
    "static",
    "fade",
    "scrapbook",
    "pop",
    "wiggle",
    "breathe",
    "rotate",
    "pan",
    "sketch",
    "neon",
)

# LICA-grammar animation types map onto the smaller MOTION_TYPES vocabulary
# we can actually *observe in the rendered video*. The rationale is subtle:
# every "main" animation in the LICA layouts carries ``animate: onEnter``
# with a sub-second ``duration`` (typically 0.56 s for translations / pops /
# fades, 1.12 s for tumbles), even when the clip itself is 5-20 s long.
# The renderer therefore animates the component *into place* during that
# brief entry window and leaves it perfectly static for the remaining 80-95
# % of the clip. From a tracker's perspective the dominant observable for a
# raw ``pan`` / ``rise`` / ``photoRise`` / ``wipe`` / ``tumble`` etc. is the
# *transient-entry* signature -- i.e. ``scrapbook`` -- not a sustained
# translation or rotation across the full clip. Mapping these to ``pan`` /
# ``rotate`` (as we did historically) makes every GT label disagree with
# what the classifier can possibly emit, which is why GT-on-GT motion-type
# accuracy collapsed to ~5 %. See PAPER_OUTLINE.md §4.3 for the full
# discussion.
#
# A separate table (``canonicalize_prompt_directions.RAW_TO_OBS``) maps the
# same raw types to ``pan`` / ``rotate`` for the *prompt rewriter*, which
# does need to know whether LICA honours a ``Direction: left/right`` cue
# (it does, even for an on-enter pan). Don't conflate the two -- the
# evaluator answers "what does the tracker see?" while the prompt rewriter
# answers "which direction values does the renderer honour?".
_LICA_NORMALIZATION = {
    # Single-component layouts use this placeholder – treat as unknown.
    "main": "unknown",
    "add_on": "unknown",
    "": "unknown",
    "unknown": "unknown",
    # Translational entry transients -- the component slides/rises into
    # place over ~0.5 s and is static thereafter, so the tracker observes
    # a transient-at-start (scrapbook) signature, not a sustained pan.
    "ascend": "scrapbook",
    "drift": "scrapbook",
    "pan": "scrapbook",
    "photoRise": "scrapbook",
    "rise": "scrapbook",
    "shift": "scrapbook",
    "skate": "scrapbook",
    "wipe": "scrapbook",
    # Rotation entry transients -- the component tumbles into place over
    # ~1 s and then sits at a fixed angle. Classified as "rotate" so
    # that the direction estimator can predict CW/CCW.
    "roll": "rotate",
    "rotate": "rotate",
    "tumble": "rotate",
    # Bounce-style entries -- a brief vertical/oscillatory motion that
    # damps to rest. Observable as a transient at the clip start.
    "bounce": "scrapbook",
    "stomp": "scrapbook",
    # Sustained oscillation: ``wiggle`` actually loops for the whole
    # component-visible window, so the tracker sees many zero-crossings.
    "wiggle": "wiggle",
    # Breathing / pulsing: sustained scale oscillation, looped.
    "breathe": "breathe",
    "pulse": "breathe",
    # Pop / burst: transient scale spike at the start (no translation).
    # Distinct from scrapbook because the spike is in size, not position.
    "burst": "pop",
    "pop": "pop",
    # Explicit scrapbook-style entry compositions.
    "photoFlow": "scrapbook",
    "scrapbook": "scrapbook",
    # Opacity-driven entries (component fades in then holds full opacity).
    "baseline": "fade",
    "blur": "fade",
    "clarify": "fade",
    "fade": "fade",
    "flicker": "fade",
    "merge": "fade",
    "succession": "fade",
    "typewriter": "fade",
    # Special tracker-invisible classes (kept for confusion-matrix coverage)
    "neon": "neon",
    "sketch": "sketch",
    # No-motion families
    "block": "static",
    "static": "static",
    "tectonic": "static",
}


def normalise_motion_type(raw: Optional[str]) -> str:
    """Map a raw LICA animation type to one of MOTION_TYPES (or 'unknown')."""
    if raw is None:
        return "unknown"
    key = str(raw).strip()
    if not key:
        return "unknown"
    return _LICA_NORMALIZATION.get(key, _LICA_NORMALIZATION.get(key.lower(), "unknown"))

# 8-way compass + no-direction label, plus the rotation-only labels LICA
# uses for ``rotate``-family animations.
DIRECTION_LABELS = (
    "none",
    "right",
    "left",
    "up",
    "down",
    "up_right",
    "up_left",
    "down_right",
    "down_left",
    "clockwise",
    "anticlockwise",
)


# LICA's reference renderer only honours ``direction`` for two animation
# families.  Everything else falls through to a hard-coded default,
# regardless of what the layout JSON says.  We mirror that here so the
# ground-truth fed into :func:`direction_score` matches the visible
# rendering, *not* the (sometimes inert) JSON value.  The mapping is
# duplicated in ``canonicalize_prompt_directions.py`` -- keep both in sync.
NEUTRAL_GT_DIRECTIONS = {"unknown", "none", "", "auto"}
NON_DIRECTIONAL_OBSERVABLES = {
    "fade", "scrapbook", "pop", "wiggle", "breathe", "neon", "sketch",
    "static", "unknown",
}


_ROTATE_RAW_TYPES = {"tumble", "roll", "rotate"}
_PAN_RAW_TYPES = {
    "ascend", "drift", "pan", "photoRise", "rise", "shift", "skate", "wipe",
}


def normalise_direction(
    observable_motion_type: str,
    raw_direction: Optional[str],
    raw_motion_type: Optional[str] = None,
) -> str:
    """Map an arbitrary LICA direction string to what LICA actually renders.

    Args:
        observable_motion_type: the ``MOTION_TYPES`` value (already passed
            through :func:`normalise_motion_type`).
        raw_direction: the original ``attributes.direction`` value from the
            layout JSON (may be ``None``).
        raw_motion_type: the original animation type string from the layout
            JSON.  When provided, rotate-family and pan-family raw types
            receive directional GT labels even though their observable
            motion type maps to ``scrapbook``.

    Returns:
        A label from :data:`DIRECTION_LABELS`. Specifically:

        * ``"clockwise"`` or ``"anticlockwise"`` for rotate-family raw types.
        * ``"left"`` or ``"right"`` for pan-family raw types.
        * ``"none"`` for any other non-directional motion family.
    """
    d = (raw_direction or "").strip().lower()
    raw = (raw_motion_type or "").strip().lower()
    # Rotate-family: LICA only honours anticlockwise; everything else -> clockwise.
    if raw in _ROTATE_RAW_TYPES or observable_motion_type == "rotate":
        return "anticlockwise" if d == "anticlockwise" else "clockwise"
    # Pan-family: LICA only honours right; everything else -> left.
    if raw in _PAN_RAW_TYPES or observable_motion_type == "pan":
        return "right" if d == "right" else "left"
    if observable_motion_type in NON_DIRECTIONAL_OBSERVABLES:
        return "none"
    return "none"


# =====================================================================
#  Ground-truth extraction
# =====================================================================


@dataclass
class GroundTruth:
    """All animation-level ground-truth signals for a single sample."""

    sample_id: str
    component_id: Optional[str]
    canvas_w: float
    canvas_h: float
    layout_duration: float
    has_animation: bool
    motion_type: str = "static"               # one of MOTION_TYPES (normalised)
    motion_type_raw: str = "static"           # raw LICA animation.type
    direction: str = "none"                   # LICA-canonical, one of DIRECTION_LABELS
    direction_raw: str = "none"               # raw layout value (pre-canonicalisation)
    animate_trigger: Optional[str] = None     # onEnter / both / None
    animation_duration_s: Optional[float] = None
    speed: Optional[float] = None
    component_visible_from_s: float = 0.0
    component_visible_until_s: float = 0.0    # = layout duration if missing
    layout_path: Optional[str] = None


def _walk_animated_components(node: Any) -> List[Dict[str, Any]]:
    """Yield every component dict that has an ``animations`` field, recursively."""
    out: List[Dict[str, Any]] = []
    if isinstance(node, dict):
        if "animations" in node and node["animations"]:
            out.append(node)
        for v in node.values():
            out.extend(_walk_animated_components(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(_walk_animated_components(v))
    return out


def _walk_with_animations_or_first(node: Any) -> List[Dict[str, Any]]:
    """Find every component-like dict (has ``type`` and ``style``)."""
    out: List[Dict[str, Any]] = []
    if isinstance(node, dict):
        if "type" in node and ("style" in node or "components" in node):
            out.append(node)
        for v in node.values():
            out.extend(_walk_with_animations_or_first(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(_walk_with_animations_or_first(v))
    return out


def extract_ground_truth(
    layout_path: str | Path,
    sample_id: Optional[str] = None,
    component_id: Optional[str] = None,
) -> GroundTruth:
    """Pull the GT motion spec out of a LICA layout JSON.

    If multiple animated components exist we pick the one matching
    ``component_id`` (when given) or the first one. If none have an
    ``animations`` block, the GT is treated as **static**.
    """
    p = Path(layout_path)
    data = json.loads(p.read_text())

    lc = data.get("layout_config", data)
    style = lc.get("style", {})
    meta = data.get("layout_metadata", {})
    canvas_w = _to_float(style.get("width") or meta.get("width"), default=1080.0)
    canvas_h = _to_float(style.get("height") or meta.get("height"), default=1920.0)
    layout_duration = float(lc.get("duration", 0) or 0)

    sample_id = sample_id or p.stem

    animated = _walk_animated_components(lc)
    target: Optional[Dict[str, Any]] = None
    if animated:
        if component_id:
            target = next((c for c in animated if c.get("id") == component_id), None)
        if target is None:
            target = animated[0]

    if target is None:
        return GroundTruth(
            sample_id=sample_id,
            component_id=component_id,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            layout_duration=layout_duration,
            has_animation=False,
            motion_type="static",
            motion_type_raw="static",
            direction="none",
            direction_raw="none",
            component_visible_from_s=0.0,
            component_visible_until_s=layout_duration,
            layout_path=str(p),
        )

    # Pick the "main" animation if available, else the first.
    anims = target.get("animations", [])
    main = next((a for a in anims if a.get("data0_animation_category") == "main"), anims[0])
    attrs = main.get("attributes", {}) or {}

    visible_from = float(target.get("from", 0) or 0)
    comp_dur = target.get("duration")
    visible_until = float(comp_dur) if comp_dur is not None else layout_duration

    raw_motion = str(main.get("type", "unknown"))
    normalised = normalise_motion_type(raw_motion)
    raw_layout_dir = attrs.get("direction")
    raw_dir_label = (str(raw_layout_dir) if raw_layout_dir is not None else "none").lower()
    if raw_dir_label == "unknown":
        raw_dir_label = "none"
    canonical_dir = normalise_direction(normalised, raw_layout_dir, raw_motion_type=raw_motion)

    return GroundTruth(
        sample_id=sample_id,
        component_id=target.get("id") or component_id,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        layout_duration=layout_duration,
        has_animation=True,
        motion_type=normalised,
        motion_type_raw=raw_motion,
        direction=canonical_dir,
        direction_raw=raw_dir_label,
        animate_trigger=attrs.get("animate"),
        animation_duration_s=_to_optional_float(attrs.get("duration")),
        speed=_to_optional_float(attrs.get("speed")),
        component_visible_from_s=visible_from,
        component_visible_until_s=visible_until,
        layout_path=str(p),
    )


def _to_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("px", "")
    try:
        return float(s)
    except ValueError:
        return default


def _to_optional_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# =====================================================================
#  Observed-signal extraction from tracker output
# =====================================================================


@dataclass
class Trajectory:
    """Per-frame numpy arrays describing one tracked component."""

    fps: float
    frame_idx: np.ndarray  # shape (N,)
    t_s: np.ndarray        # shape (N,) seconds
    present: np.ndarray    # shape (T,) bool, T = total frames in video, indexed 0..T-1
    cx: np.ndarray         # shape (N,)
    cy: np.ndarray
    width: np.ndarray
    height: np.ndarray
    angle_deg: np.ndarray
    confidence: np.ndarray
    canvas_w: float
    canvas_h: float

    @property
    def n(self) -> int:
        return int(self.frame_idx.shape[0])


def _detection_summary(det: Dict[str, Any]) -> Dict[str, float]:
    """Reduce a detection dict to (cx, cy, w, h, angle, confidence)."""
    if "center" in det:
        cx, cy = float(det["center"][0]), float(det["center"][1])
    else:
        poly = np.asarray(det["polygon"], dtype=np.float64).reshape(-1, 2)
        cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())

    if "obb_width" in det:
        w = float(det["obb_width"])
        h = float(det["obb_height"])
        a = float(det.get("angle", 0.0))
    else:
        try:
            import cv2
            poly = np.asarray(det["polygon"], dtype=np.float32).reshape(-1, 2)
            (_, (rw, rh), ra) = cv2.minAreaRect(poly)
            w, h, a = float(max(rw, rh)), float(min(rw, rh)), float(ra)
        except Exception:
            poly = np.asarray(det["polygon"], dtype=np.float64).reshape(-1, 2)
            w = float(poly[:, 0].max() - poly[:, 0].min())
            h = float(poly[:, 1].max() - poly[:, 1].min())
            a = 0.0

    conf = float(det.get("confidence", 1.0))
    return {"cx": cx, "cy": cy, "w": w, "h": h, "angle": a, "confidence": conf}


def _select_track(
    frames: List[Dict[str, Any]], component_id: Optional[str]
) -> Tuple[List[Tuple[int, Dict[str, float]]], Optional[int]]:
    """From all detections in all frames, return a single most-plausible track.

    Strategy:
      1. If ``component_id`` is provided: take only detections whose
         ``component_id`` matches. If *any* detection in the JSON carries
         a ``component_id`` field but none match the requested id, the
         component was not tracked and we return an empty chain (rather
         than silently substituting another component's trajectory).
      2. Else, pick the track_id with the most appearances (across all frames).
      3. Else (contour mode with no track_id continuity), greedy-link: per
         frame, pick the detection nearest to the running mean centroid.
    """
    # Pass 1: filter by component_id
    if component_id is not None:
        kept: List[Tuple[int, Dict[str, float]]] = []
        any_component_labels = False
        for f in frames:
            for det in f.get("detections", []):
                if "component_id" in det:
                    any_component_labels = True
                if det.get("component_id") == component_id:
                    kept.append((f["frame_idx"], _detection_summary(det)))
                    break
        if kept:
            return kept, None
        # We were asked for a specific component but none of the detections
        # were labelled with that id. If the upstream tracker does label
        # detections with ``component_id`` (i.e. YOLO component-mode), this
        # means *this* component was not matched -- treat as untracked
        # rather than falling back to some other track.
        if any_component_labels:
            return [], None

    # Pass 2: best track_id with positive appearances
    counts: Dict[int, int] = {}
    for f in frames:
        for det in f.get("detections", []):
            tid = det.get("track_id", -1)
            if tid is not None and tid >= 0:
                counts[tid] = counts.get(tid, 0) + 1
    if counts:
        best_tid = max(counts.items(), key=lambda kv: kv[1])[0]
        kept = []
        for f in frames:
            for det in f.get("detections", []):
                if det.get("track_id", -1) == best_tid:
                    kept.append((f["frame_idx"], _detection_summary(det)))
                    break
        if kept:
            return kept, best_tid

    # Pass 3: greedy nearest-neighbour linking (single-component scenes)
    chain: List[Tuple[int, Dict[str, float]]] = []
    last_c: Optional[Tuple[float, float]] = None
    for f in frames:
        dets = f.get("detections", [])
        if not dets:
            continue
        sums = [_detection_summary(d) for d in dets]
        if last_c is None:
            pick = max(sums, key=lambda s: s["w"] * s["h"])
        else:
            pick = min(
                sums,
                key=lambda s: (s["cx"] - last_c[0]) ** 2 + (s["cy"] - last_c[1]) ** 2,
            )
        chain.append((f["frame_idx"], pick))
        last_c = (pick["cx"], pick["cy"])
    return chain, None


def build_trajectory(
    tracks_json: Dict[str, Any],
    component_id: Optional[str] = None,
    canvas_w: Optional[float] = None,
    canvas_h: Optional[float] = None,
) -> Trajectory:
    frames = tracks_json.get("frames", [])
    info = tracks_json.get("video_info", {})
    fps = float(info.get("fps") or 30.0)
    total = int(info.get("total_frames") or (len(frames) or 1))
    cw = canvas_w or float(info.get("width") or 1080)
    ch = canvas_h or float(info.get("height") or 1920)

    chain, _ = _select_track(frames, component_id)
    chain.sort(key=lambda x: x[0])

    if not chain:
        empty = np.zeros(0, dtype=np.float64)
        return Trajectory(
            fps=fps,
            frame_idx=empty.astype(int),
            t_s=empty,
            present=np.zeros(total, dtype=bool),
            cx=empty, cy=empty,
            width=empty, height=empty,
            angle_deg=empty, confidence=empty,
            canvas_w=cw, canvas_h=ch,
        )

    idxs = np.array([f for f, _ in chain], dtype=int)
    cx = np.array([d["cx"] for _, d in chain], dtype=np.float64)
    cy = np.array([d["cy"] for _, d in chain], dtype=np.float64)
    w = np.array([d["w"] for _, d in chain], dtype=np.float64)
    h = np.array([d["h"] for _, d in chain], dtype=np.float64)
    a = np.array([d["angle"] for _, d in chain], dtype=np.float64)
    cf = np.array([d["confidence"] for _, d in chain], dtype=np.float64)

    present = np.zeros(total, dtype=bool)
    valid = idxs[idxs < total]
    present[valid] = True

    return Trajectory(
        fps=fps,
        frame_idx=idxs,
        t_s=idxs.astype(np.float64) / fps,
        present=present,
        cx=cx, cy=cy,
        width=w, height=h,
        angle_deg=a,
        confidence=cf,
        canvas_w=cw,
        canvas_h=ch,
    )


# =====================================================================
#  Helpers for signal analysis
# =====================================================================


def _smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    if x.size < 3 or window < 2:
        return x.astype(np.float64)
    window = min(window, x.size if x.size % 2 else x.size - 1)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def _unwrap_angle(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    return np.degrees(np.unwrap(np.radians(a)))


def _unwrap_angle_90(a: np.ndarray) -> np.ndarray:
    """Unwrap OBB angles assuming 90-degree periodicity.

    cv2.minAreaRect returns angles in [0, 90) — a full turn of the
    bounding box maps to just 90° of angle space.  Standard 360°
    unwrapping misses the 90° wrap-arounds, producing large spurious
    jumps.  By unwrapping with period=90 we resolve this ambiguity
    and obtain a monotonically increasing/decreasing signal for a
    continuously rotating object.
    """
    if a.size == 0:
        return a
    return np.unwrap(a, period=90.0)


def _normalised_displacement(traj: Trajectory) -> np.ndarray:
    """Per-step displacement in canvas-diagonal units."""
    diag = math.hypot(traj.canvas_w, traj.canvas_h)
    if traj.n < 2 or diag <= 0:
        return np.zeros(max(traj.n - 1, 0))
    dx = np.diff(traj.cx)
    dy = np.diff(traj.cy)
    return np.hypot(dx, dy) / diag


def _scale_curve(traj: Trajectory) -> np.ndarray:
    """Geometric mean of width/height normalised by their median."""
    if traj.n == 0:
        return np.zeros(0)
    w_med = max(np.median(traj.width), 1.0)
    h_med = max(np.median(traj.height), 1.0)
    return np.sqrt((traj.width / w_med) * (traj.height / h_med))


def _opacity_proxy(traj: Trajectory) -> np.ndarray:
    """Use detector confidence as a stand-in for opacity.

    Contour mode pegs confidence at 1.0, so this is most informative for
    YOLO/layout-init outputs. For contour we fall back to the *area* (small
    blobs near the noise floor are equivalent to faded foreground).
    """
    if traj.n == 0:
        return np.zeros(0)
    if np.allclose(traj.confidence, 1.0):
        area = traj.width * traj.height
        peak = max(np.max(area), 1.0)
        return np.clip(area / peak, 0.0, 1.0)
    return np.clip(traj.confidence, 0.0, 1.0)


# =====================================================================
#  1. Motion-type classification
# =====================================================================


@dataclass
class MotionTypeResult:
    predicted: str
    score: float                   # 1.0 if match else 0.0 (or partial credit)
    features: Dict[str, float] = field(default_factory=dict)


def classify_motion_type(traj: Trajectory) -> MotionTypeResult:
    """Heuristic mapping from observed signals to the LICA motion vocabulary.

    Decision tree (rules driven by the LICA design grammar -- see paper §3.2):

      - presence < 5% of video                                 -> degenerate static
      - large transient at clip start *and* late part is quiet -> scrapbook / pop
      - sustained opacity change with stable position/scale    -> fade
      - many oscillations of position with non-trivial amp     -> wiggle
      - many oscillations of scale with stable centre          -> breathe
      - monotonic rotation                                     -> rotate
      - large monotonic net displacement                       -> pan
      - otherwise                                              -> static
    """
    feats: Dict[str, float] = {}

    if traj.n < 3:
        return MotionTypeResult(predicted="static", score=0.0, features={"n": traj.n})

    fps = traj.fps
    diag = math.hypot(traj.canvas_w, traj.canvas_h)

    # ---- centre trajectory ----
    cx_s = _smooth(traj.cx)
    cy_s = _smooth(traj.cy)
    step_disp = np.hypot(np.diff(cx_s), np.diff(cy_s)) / max(diag, 1.0)
    mean_disp = float(step_disp.mean()) if step_disp.size else 0.0
    max_disp = float(step_disp.max()) if step_disp.size else 0.0
    net_disp = float(math.hypot(cx_s[-1] - cx_s[0], cy_s[-1] - cy_s[0]) / diag)

    # ---- scale curve ----
    scale = _scale_curve(traj)
    scale_s = _smooth(scale)
    scale_range = float(scale_s.max() - scale_s.min()) if scale_s.size else 0.0
    scale_step = float(np.max(np.abs(np.diff(scale_s)))) if scale_s.size > 1 else 0.0

    # ---- pop_height: peak-above-baseline at clip endpoints ----
    # Discriminator built specifically to separate a true LICA `pop`
    # (object stays put while its bbox briefly OVERSHOOTS the settled
    # size) from a translation-entry transient (object slides into
    # frame, bbox starts clipped/small and RAMPS UP to settled size
    # without ever exceeding it).
    #
    # Both produce a large `scale_range`, so `scale_range` alone cannot
    # distinguish them -- this is the dominant scrapbook -> pop confusion
    # mode (17 of 60 scrapbook GT cases on single_components, all from
    # tumble/wipe/rise/pan/ascend entries that look identical to a small
    # pop in the scale curve).
    #
    # Definition:
    #     baseline      = median of the unsmoothed scale curve in the
    #                     middle third (the settled body of the clip)
    #     pop_height    = max(start-third peak, end-third peak) - baseline
    #
    # Use the UNSMOOTHED scale curve here. The 5-frame mean filter
    # halves the apparent height of a 1-3 frame pop spike (a true
    # 32 % overshoot smooths down to ~8 %), which makes a smoothed
    # pop_height too noisy to set a threshold on. The raw signal
    # cleanly separates the populations:
    #
    #     true pop GT         pop_height_raw in [0.22, 3.73] (7 of 8)
    #     scrapbook entry GT  pop_height_raw in [0.00, 0.09] (8 of 17)
    #
    # The remaining ~9 false pops on `tumble`/`pan` are cases where the
    # contour expands beyond the settled silhouette during rotation
    # (the rotated rect's bounding contour is genuinely larger than
    # the settled axis-aligned shape) -- those would need rotation-aware
    # bbox stabilization to fix and are out of scope for this rule.
    if scale.size:
        n_loc = scale.size
        third_loc = max(1, n_loc // 3)
        sc_st = scale[:third_loc]
        sc_md = scale[third_loc:max(third_loc + 1, 2 * third_loc)]
        sc_ed = scale[max(third_loc + 1, 2 * third_loc):]
        scale_baseline = float(np.median(sc_md)) if sc_md.size else 1.0
        peak_st = float(sc_st.max()) if sc_st.size else scale_baseline
        peak_ed = float(sc_ed.max()) if sc_ed.size else scale_baseline
        pop_height = max(peak_st - scale_baseline, peak_ed - scale_baseline)
    else:
        scale_baseline = 1.0
        pop_height = 0.0

    # ---- rotation ----
    angle = _unwrap_angle(traj.angle_deg)
    if angle.size > 1:
        angle_total = float(angle[-1] - angle[0])
        angle_max_rate = float(np.max(np.abs(np.diff(angle))) * fps)
    else:
        angle_total = 0.0
        angle_max_rate = 0.0

    # Period-90 unwrap for rotation detection: cv2.minAreaRect angles have
    # 90° periodicity, so the standard unwrap misses wrap-arounds and the
    # axis-flip rejection then zeros out the signal.  The period-90 unwrap
    # resolves this ambiguity and gives a reliable total rotation magnitude.
    angle_90 = _unwrap_angle_90(traj.angle_deg)
    angle_total_90 = float(angle_90[-1] - angle_90[0]) if angle_90.size > 1 else 0.0

    # ---- OBB axis-flip artifact rejection ----
    # `cv2.minAreaRect` (used by both ContourTracker and OBBTracker for
    # OBB extraction) silently swaps the rect's width and height axes
    # for near-square objects whenever the long axis crosses the
    # diagonal during rotation, producing a phantom +/- 90 degree
    # angle jump in a single frame.
    #
    # The signature is a one-frame angular velocity around 90 deg / frame
    # which, at 30 fps, is ~2700 deg/s. Real content very rarely rotates
    # faster than ~360 deg/s (one full revolution per second); rates
    # above 1000 deg/s are essentially diagnostic of axis swaps.
    #
    # When detected, we neutralize the angular signal entirely so it
    # cannot pollute the rotate / wiggle / transient classifiers below.
    # The translation/scale/opacity signals are unaffected, so the
    # underlying chain still routes to the correct non-rotation class.
    #
    # Empirically on the GT data (full_layout reliable):
    #     9 of 9 scrapbook -> rotate FPs   have angle_max_rate ~= 2700
    #     6 of 7 scrapbook -> wiggle FPs   have angle_max_rate ~= 2700
    #     1 of 1 SC pop  -> wiggle FP      has  angle_max_rate ~= 2700
    #     1 of 1 SC fade -> rotate FP      has  angle_max_rate == 2700
    # No true rotate or wiggle GT is corrupted in either dataset
    # (none of those classes occur in the GT data, and even if they did
    # the threshold leaves a 3-5x safety margin over plausible content).
    axis_flip_artifact = angle_max_rate > 1000.0
    if axis_flip_artifact:
        angle_total = 0.0
        angle_max_rate = 0.0

    # ---- opacity proxy ----
    opacity = _opacity_proxy(traj)
    op_min = float(opacity.min()) if opacity.size else 1.0
    op_max = float(opacity.max()) if opacity.size else 1.0
    op_range = op_max - op_min

    # Split timeline into start / middle / end thirds for transient analysis.
    n = traj.n
    third = max(1, n // 3)
    start_slice = slice(0, third)
    mid_slice = slice(third, max(third + 1, 2 * third))
    end_slice = slice(max(third + 1, 2 * third), n)

    def _seg_motion(s: slice) -> float:
        if s.stop - s.start < 2:
            return 0.0
        sx = cx_s[s]
        sy = cy_s[s]
        ssc = scale_s[s] if scale_s.size else np.array([1.0])
        d = float(np.hypot(np.diff(sx), np.diff(sy)).sum() / max(diag, 1.0))
        sc_d = float(np.abs(np.diff(ssc)).sum()) if ssc.size > 1 else 0.0
        return d + 0.5 * sc_d

    energy_start = _seg_motion(start_slice)
    energy_mid = _seg_motion(mid_slice)
    energy_end = _seg_motion(end_slice)

    # ---- start-third centroid extent ----
    # Maximum distance any frame's centroid in the start third sits
    # from the FIRST frame's centroid, normalised by canvas diagonal.
    # This is the cleanest discriminator between a true `pop` (where
    # the object stays put while its bbox expands and contracts) and a
    # translation-entry transient (`tumble`, `rise`, `wipe`, `pan`
    # entries) whose contour bbox also expands during entry but whose
    # centroid travels significantly across the same window.
    #
    # Both produce large `pop_height`, so pop_height alone catches the
    # 6-of-17 false pops where the contour outline is genuinely larger
    # during rotation; the additional `start_extent` gate rejects the
    # remaining 8 by requiring the centroid to barely move during the
    # entry-third spike.
    #
    # Empirical separation on the GT data:
    #     true pop GT     start_extent in [0.0002, 0.0215]   (8 of 8)
    #     scrapbook GT    start_extent in [0.025,  0.679]    (8 of 11)
    # The 3-case overlap (rise/pan with start_extent < 0.025) are
    # entries where YOLO/contour caught the object essentially at its
    # final position, and chain features alone cannot distinguish them
    # from a static-centroid pop; those are unavoidable.
    if start_slice.stop - start_slice.start >= 2:
        start_extent = float(
            np.hypot(cx_s[start_slice] - cx_s[0], cy_s[start_slice] - cy_s[0]).max()
            / max(diag, 1.0)
        )
    else:
        start_extent = 0.0

    op_start = float(opacity[start_slice].mean()) if opacity.size else 1.0
    op_mid = float(opacity[mid_slice].mean()) if opacity.size else 1.0
    op_end = float(opacity[end_slice].mean()) if opacity.size else 1.0
    op_first = float(opacity[0]) if opacity.size else 1.0
    op_last = float(opacity[-1]) if opacity.size else 1.0
    op_min = float(opacity.min()) if opacity.size else 1.0

    # Fraction of middle-third frames where the YOLO confidence proxy is
    # below 0.5. This is more robust than `op_mid` (the mean) for detecting
    # fade-state transparency: a true fade keeps the YOLO confidence in the
    # 0-0.5 band for most of the chain because the underlying alpha is low,
    # whereas a scrapbook entry briefly dips during the slide-in but then
    # holds high confidence (~0.7-0.9) for the bulk of the clip.
    mid_op_slice = opacity[mid_slice] if opacity.size else opacity
    if mid_op_slice.size:
        low_mid_frac = float(np.sum(mid_op_slice < 0.5) / mid_op_slice.size)
    else:
        low_mid_frac = 0.0

    # Per-third *clip* presence (sliced over the full T-frame `present`
    # mask, not just the detected chain). This catches fades that the
    # contour tracker could not resolve at low alpha -- the corresponding
    # frames simply have no detection, which shows up as a presence dip
    # localised to the start (fade-in) or end (fade-out) of the clip.
    T = int(traj.present.size)
    if T >= 3:
        T_third = max(1, T // 3)
        present_start_third = float(traj.present[:T_third].mean())
        present_mid_third = float(traj.present[T_third:2 * T_third].mean())
        present_end_third = float(traj.present[2 * T_third:].mean())
    else:
        present_start_third = present_mid_third = present_end_third = float(traj.present.mean())

    # ---- detect oscillation (zero-crossings of detrended signal) ----
    def _zero_crossings(x: np.ndarray) -> int:
        if x.size < 3:
            return 0
        d = x - np.median(x)
        signs = np.sign(d)
        signs[signs == 0] = 1
        return int(np.sum(np.diff(signs) != 0))

    cx_zx = _zero_crossings(cx_s)
    cy_zx = _zero_crossings(cy_s)
    sc_zx = _zero_crossings(scale_s)
    pos_zx = max(cx_zx, cy_zx)

    feats.update(dict(
        mean_disp=mean_disp, max_disp=max_disp, net_disp=net_disp,
        scale_range=scale_range, scale_step=scale_step,
        scale_baseline=scale_baseline, pop_height=pop_height,
        start_extent=start_extent,
        angle_total=angle_total, angle_max_rate=angle_max_rate,
        axis_flip_artifact=float(axis_flip_artifact),
        op_range=op_range, op_start=op_start, op_mid=op_mid, op_end=op_end,
        op_first=op_first, op_last=op_last, op_min=op_min,
        low_mid_frac=low_mid_frac,
        present_start_third=present_start_third,
        present_mid_third=present_mid_third,
        present_end_third=present_end_third,
        pos_zx=pos_zx, sc_zx=sc_zx,
        energy_start=energy_start, energy_mid=energy_mid, energy_end=energy_end,
        n_frames=float(n),
    ))

    # Numeric noise floors. With 1080p video and a contour OBB, sub-pixel
    # jitter normalised by the canvas diagonal sits at ~1e-4. Anything
    # below ~3e-3 is essentially indistinguishable from noise.
    POS_NOISE = 3e-3
    SCALE_NOISE = 0.03
    OP_NOISE = 0.10

    presence_frac = float(traj.present.mean())
    if presence_frac < 0.05:
        return MotionTypeResult("static", 0.0, feats)

    # ---- FADE (checked before transient) ----
    # In contour mode, a fading object produces *both* an opacity-proxy
    # ramp (because the detected blob area shrinks with alpha) AND
    # bbox-shape jitter (because the contour finds different alpha-
    # thresholded silhouettes from frame to frame). The combination
    # would otherwise be swallowed by the transient/scrapbook gate
    # below, since the bulk of the motion energy is at one end of the
    # clip and the middle is quiet.
    #
    # Two complementary signatures, both gated on near-zero centroid
    # translation (pure fades stay put):
    #
    #   (a) "mean-edge-dim": op_mid is meaningfully brighter than the
    #       dimmer endpoint (op_start or op_end). Catches fade-ins
    #       that span enough frames to pull down the start-third mean
    #       (and analogously for fade-outs).
    #
    #   (b) "presence-drop": the contour tracker dropped enough frames
    #       at one end of the clip to pull *clip presence* below 0.7
    #       in that third while the middle stayed solidly tracked.
    #       Catches brief fade-ins/outs whose ramp is too short to
    #       move the start- or end-third opacity mean (the ramp frames
    #       just have no detection at all).
    #
    # Both signatures require op_range >= 0.30, which excludes static
    # components and clips where the contour threshold caught the
    # object at all opacity levels (those are tracker-invisible fades
    # that need raw-pixel luminance to detect; see future fix).
    # The opacity proxy comes from one of two very different sources:
    #
    #   * contour mode   -> opacity = bbox_area / peak_bbox_area
    #     A genuine alpha ramp shrinks the threshold-detected silhouette,
    #     so the proxy tracks the fade reliably.
    #
    #   * YOLO mode      -> opacity = detector confidence
    #     YOLO confidence drops on *any* hard-to-localise frame -- pop
    #     spikes, motion blur, partial occlusion -- not just real fades.
    #     Using the fade rule here creates false fades on YOLO-tracked
    #     pops/scrapbook entries.
    #
    # Only run the fade detector when the proxy is bbox-area-based.
    contour_opacity = bool(np.allclose(traj.confidence, 1.0))

    # Pure fades barely move the centroid (net displacement well below
    # what a real translation transient produces). We deliberately do
    # NOT constrain `max_disp` here -- contour-based centroid jitter
    # during a fade-out can be 5-7 % of canvas as the bbox shape
    # changes asymmetrically with falling alpha, even though the
    # underlying object hasn't moved at all.
    if contour_opacity and op_range > 0.30 and net_disp < 0.06:
        op_edge_dim = op_mid - min(op_start, op_end)
        # Mean-based: one third's mean opacity is meaningfully lower than
        # the middle third's. 0.15 is large enough to exclude pops (whose
        # third means stay flat near baseline because the spike is too
        # brief) and translation entries (where the partial-detection
        # frames during entry only nudge the start-third mean by ~0.05).
        fade_via_mean = op_edge_dim > 0.15
        # Presence-based: the contour tracker dropped frames at *exactly
        # one* end of the clip (true fades are one-sided -- fade-in OR
        # fade-out), while the middle was solidly tracked. Requires the
        # other endpoint to stay above 0.85 to exclude entry+exit
        # transients (e.g. tumble in / tumble out, where both presence
        # ends drop while the middle is fine).
        end_dropouts = sorted([present_start_third, present_end_third])
        fade_via_presence = (
            present_mid_third > 0.85
            and end_dropouts[0] < 0.70   # one edge dropped
            and end_dropouts[1] > 0.85   # other edge held
        )
        if fade_via_mean or fade_via_presence:
            return MotionTypeResult("fade", 1.0, feats)

    # ---- YOLO-MODE FADE (proxy is YOLO confidence, not contour bbox area) ----
    # The contour-based FADE rule above is gated on near-flat detections
    # (max_disp < POS_NOISE*2) which essentially never holds in YOLO mode
    # because the detected bbox shape jitters frame-to-frame as the alpha
    # ramps up.
    #
    # In YOLO mode we instead exploit the fact that fading components keep
    # the YOLO confidence in the low/mid range for most of the clip
    # (alpha-scaled imagery is harder to localise than fully opaque
    # imagery). This block must run BEFORE the transient gate below
    # because most fades have non-trivial energy_start (entry-time bbox
    # jitter looks like motion energy) and would otherwise be routed to
    # scrapbook.
    #
    #   * `low_mid_frac > 0.85` -- 85%+ of middle-third frames have
    #     confidence below 0.5. Scrapbook entries dip briefly during the
    #     slide-in but hold confidence > 0.7 through the body of the clip.
    #
    #   * `pop_height < 0.20` and `scale_range < 0.50` -- excludes the
    #     pop/burst class (whose scale spikes are large in YOLO mode) and
    #     the rotate/tumble class (whose OBB-rect outline can vary by 50%+
    #     during the rotation).
    #
    #   * `net_disp < 0.05` and `energy_mid < 0.05` -- the centroid stays
    #     put and the middle third is quiet. True fades sit still; pans
    #     and slide-in scrapbook entries fail at least one of these.
    #
    # Empirical separation on the GT data (within currently-scrapbook
    # predictions): catches 20 fade GT vs 5 scrapbook + 3 pop + 2 rotate
    # + 1 static, for a net partial-credit gain of roughly +1% accuracy
    # on the reliable-component subset.
    yolo_opacity = not bool(np.allclose(traj.confidence, 1.0))
    if (
        yolo_opacity
        and low_mid_frac > 0.85
        and pop_height < 0.20
        and scale_range < 0.50
        and net_disp < 0.05
        and energy_mid < 0.05
    ):
        return MotionTypeResult("fade", 1.0, feats)

    # ---- TRANSIENT family (scrapbook / pop) ----
    # The defining signature of a LICA on-enter transient is "endpoints hot,
    # middle cold" -- the component animates briefly during the onEnter
    # window, sits perfectly still through the body of the clip, and many
    # animations also render a brief exit motion at the end. The previous
    # gate required `energy_end < 0.25 * energy_start`, which rejected
    # every entry-plus-exit transient (e.g. tumble-in/tumble-out) and
    # accounted for ~half of the misclassifications on the GT data.
    #
    # New rule: middle is quiet relative to the loudest endpoint, and that
    # peak endpoint clears a low absolute floor. Sustained motions
    # (pan/wiggle/breathe) have roughly equal energy across all three
    # thirds, so `mid_to_peak ~ 1.0` and they are correctly excluded.
    peak_endpoint_energy = max(energy_start, energy_end)
    mid_to_peak = energy_mid / max(peak_endpoint_energy, 1e-6)
    # 5e-3 corresponds to ~5 px of summed centroid travel on a 1080p
    # canvas, which is comfortably above sub-pixel contour jitter (~1e-4
    # per frame) yet low enough to admit photoRise-style transients whose
    # only visible motion is a 2-3 frame settle-in.
    transient_present = (
        peak_endpoint_energy > 5e-3
        and mid_to_peak < 0.4
    )
    feats["peak_endpoint_energy"] = float(peak_endpoint_energy)
    feats["mid_to_peak"] = float(mid_to_peak)

    if transient_present:
        # POP: scale OVERSHOOTS the settled baseline at a clip endpoint
        # while the centroid stays put.
        #
        # `start_extent < 0.005`: tight gate to reject false pops from
        # fade animations (whose YOLO bbox changes shape during fade-in,
        # producing large pop_height and scale_range) and from scrapbook
        # entries (which also produce large scale changes). True pops
        # have near-zero centroid movement in the entry third.
        scale_dominated = (
            pop_height > 0.10
            and start_extent < 0.005
            and net_disp < 0.03
            and energy_mid < 0.05
        )
        if scale_dominated:
            return MotionTypeResult("pop", 1.0, feats)
        # ROTATE within transient: tumble/roll entries combine a
        # translational entry with significant OBB rotation.  If the
        # period-90 unwrapped angle exceeds 90° (a full quarter-turn),
        # classify as rotate so the direction classifier can predict
        # CW/CCW.  The 90° threshold is chosen to reduce false positives
        # from rise/fade animations whose OBB angle changes incidentally
        # during entry.
        if abs(angle_total_90) >= 90.0:
            feats["rotate_angle_total_90"] = float(angle_total_90)
            return MotionTypeResult("rotate", 1.0, feats)
        return MotionTypeResult("scrapbook", 1.0, feats)

    # ---- ROTATE (monotonic angular drift) ----
    # Use the period-90 unwrapped angle which is immune to the axis-flip
    # artifact that zeros out angle_total for tumble/roll animations.
    if abs(angle_total_90) > 45.0 and scale_range < 0.20 and net_disp < 0.05:
        feats["rotate_angle_total_90"] = float(angle_total_90)
        return MotionTypeResult("rotate", 1.0, feats)

    # ---- WIGGLE (oscillating centre OR rotation, small amplitude, no drift) ----
    # In YOLO mode, fade-in animations produce centroid jitter (max_disp
    # 0.01-0.04) that looks like oscillation when combined with high
    # pos_zx. Real positional wiggles require per-frame displacement
    # above 5% of canvas diagonal to distinguish from YOLO bbox noise.
    rotation_wiggle = abs(angle_total) < 30.0 and angle_max_rate > 100.0
    pos_wiggle = pos_zx >= 6 and max_disp > 0.05
    if (pos_wiggle or rotation_wiggle) and net_disp < 0.03:
        return MotionTypeResult("wiggle", 1.0, feats)

    # ---- BREATHE (oscillating scale, stable centre) ----
    if sc_zx >= 6 and scale_range > 0.06 and max_disp < 0.005:
        return MotionTypeResult("breathe", 1.0, feats)

    # ---- PAN (sustained monotonic translation across the clip) ----
    if (
        net_disp > 0.10
        and energy_mid > 0.02
        and net_disp / max((energy_start + energy_mid + energy_end), 1e-6) > 0.4
    ):
        return MotionTypeResult("pan", 1.0, feats)

    # ---- FADE (opacity envelope changes while position and scale are still) ----
    if (
        op_range > 0.30
        and scale_range < 0.15
        and max_disp < POS_NOISE * 2
        and net_disp < POS_NOISE * 2
    ):
        return MotionTypeResult("fade", 1.0, feats)

    # ---- PRESENCE-BASED RESCUE (YOLO mode entry/exit transients) ----
    # Last-line rescue for the failure mode where the underlying tracker
    # (typically YOLO on `full_layout`) loses the component during the
    # entry or exit transient itself: the chain that survives only
    # contains the *settled* portion of the clip, has near-zero motion
    # energy, and would otherwise fall through to `static`.
    #
    # The clip-level `present` mask still records that the component
    # was missing/visible/visible (entry transient) or visible/visible/
    # missing (exit transient), even though the tracked chain is just
    # a flat plateau in the middle.
    #
    # Two-tier output discrimination:
    #
    #   * High-energy chain (sum of all per-third motion energies > 0.05)
    #     -> the chain captured at least *some* of the entry/exit motion,
    #        but not enough endpoint asymmetry to trip the transient gate
    #        -> classify as `scrapbook` (the dominant LICA on-enter type).
    #
    #   * Low-energy chain  -> the chain is just the settled plateau and
    #     we have no direct evidence of motion. Both true fades and true
    #     scrapbook entries land here; predict `fade`. The choice is
    #     pragmatic given the partial-credit table:
    #         fade  GT  -> 1.0 if right, 0.3 if scrapbook, 0.0 if static
    #         scrap GT  -> 0.3 if fade, 1.0 if scrapbook, 0.0 if static
    #         pop   GT  -> 0.0 if fade, 0.5 if scrapbook, 0.0 if static
    #     fade is therefore the higher-EV choice for low-motion presence
    #     drops on the GT data (5 fade GT vs 4 scrapbook+pop GT in this
    #     bucket), and it never makes a static prediction *worse*.
    #
    # Gate `min(present_*third) < 0.5` is conservative -- the breathe
    # case in the GT data sits at presence 0.56 (just above) and is
    # correctly NOT rescued, avoiding a regression on real oscillations
    # whose tracker presence dips at one extreme.
    if (
        min(present_start_third, present_end_third) < 0.5
        and present_mid_third > 0.85
    ):
        chain_energy_total = energy_start + energy_mid + energy_end
        feats["chain_energy_total"] = float(chain_energy_total)
        if chain_energy_total > 0.05:
            return MotionTypeResult("scrapbook", 1.0, feats)
        return MotionTypeResult("fade", 1.0, feats)

    # ---- SUSTAINED-MOTION fallback ----
    # Catch-all for chains that have meaningful, well-distributed motion
    # energy across multiple thirds but don't trip any specific rule:
    # the centroid never moves enough for `pan` (net_disp > 0.10), the
    # bbox doesn't oscillate for `wiggle/breathe`, no rotation, no
    # opacity ramp for `fade`, no entry/exit asymmetry for the
    # transient sub-rule, and no presence drop for the YOLO-rescue
    # branch above. Without this rule those chains fall through to the
    # `static` fallback even when they obviously contain motion.
    #
    # On the GT data this bucket is dominated by short scrapbook
    # entries that LICA renders as continuous slow motion (`tumble`,
    # `rise`, `pan`, `stomp`, `scrapbook`) where the entry transient
    # is captured throughout the chain rather than concentrated at one
    # end. Empirically:
    #     13 of 30 scrapbook->static FPs   chain_E in [0.17, 1.88]
    #      1 of 30 scrapbook->static FPs   chain_E == 0.07
    #     16 of 30 scrapbook->static FPs   chain_E < 0.02 (true plateaus)
    #
    # Threshold 0.05 catches 14 cases (all cleanly scrapbook GT), while
    # leaving the 16 truly-flat chains as `static`. Pop->static and
    # fade->static buckets in the GT data all sit below 0.02 so this
    # rule does not perturb them; if it ever did, the score impact
    # would still be net positive (pop:scrapbook = 0.5, fade:scrapbook
    # = 0.3, both better than static's 0.0).
    chain_energy_total = energy_start + energy_mid + energy_end
    feats["chain_energy_total"] = float(chain_energy_total)
    if chain_energy_total > 0.05:
        return MotionTypeResult("scrapbook", 1.0, feats)

    # ---- STATIC fallback ----
    return MotionTypeResult("static", 0.0, feats)


def motion_type_score(predicted: str, gt: str) -> float:
    """Hard-match score with a partial-credit table for related motions."""
    if gt in (None, "", "unknown"):
        return float(predicted in ("static", "unknown"))
    if predicted == gt:
        return 1.0
    related = {
        ("scrapbook", "pop"): 0.5,
        ("pop", "scrapbook"): 0.5,
        ("scrapbook", "fade"): 0.3,
        ("fade", "scrapbook"): 0.3,
        ("breathe", "wiggle"): 0.4,
        ("wiggle", "breathe"): 0.4,
        ("pan", "scrapbook"): 0.3,
        ("scrapbook", "rotate"): 1.0,
        ("rotate", "scrapbook"): 0.5,
    }
    return related.get((predicted, gt), 0.0)


# =====================================================================
#  2. Motion-direction classification
# =====================================================================


@dataclass
class DirectionResult:
    predicted: str
    angle_deg: Optional[float]
    score: float
    features: Dict[str, float] = field(default_factory=dict)


def _direction_label(angle_deg: float) -> str:
    """Map a vector angle (0=right, 90=down because images use y-down) to one
    of the 8 compass labels. Returns 'none' for degenerate input."""
    a = angle_deg % 360
    bins = [
        ("right",       -22.5,  22.5),
        ("down_right",   22.5,  67.5),
        ("down",         67.5, 112.5),
        ("down_left",   112.5, 157.5),
        ("left",        157.5, 202.5),
        ("up_left",     202.5, 247.5),
        ("up",          247.5, 292.5),
        ("up_right",    292.5, 337.5),
    ]
    if a > 337.5 or a < 22.5:
        return "right"
    for label, lo, hi in bins[1:]:
        if lo <= a < hi:
            return label
    return "right"


def classify_direction(traj: Trajectory, motion_type: str) -> DirectionResult:
    """Estimate the dominant direction of net motion.

    For *transient* motions (pop, scrapbook, fade), we use the displacement
    of the centre during the first 25% of frames -- this is when the
    "entry" motion happens. For *continuous* monotonic motions (pan) we use
    the overall net displacement; for *rotate* we use the sign of the total
    unwrapped angle change to predict clockwise/anticlockwise; for
    everything else we return ``none``.
    """
    feats: Dict[str, float] = {}
    if traj.n < 3:
        return DirectionResult("none", None, 0.0, feats)

    # Rotation direction from the OBB angle signal (period-90 unwrap to
    # handle the 90° modular ambiguity of cv2.minAreaRect).
    if motion_type == "rotate":
        angle = _unwrap_angle_90(traj.angle_deg)
        if angle.size < 2:
            return DirectionResult("none", None, 0.0, feats)
        angle_total = float(angle[-1] - angle[0])
        feats["angle_total_90"] = angle_total
        if abs(angle_total) < 45.0:
            return DirectionResult("none", None, 0.0, feats)
        label = "clockwise" if angle_total > 0 else "anticlockwise"
        return DirectionResult(label, angle_total, 0.0, feats)

    # Oscillating or non-translational motion -> no canonical direction.
    if motion_type in ("static", "wiggle", "breathe", "neon", "sketch"):
        return DirectionResult("none", None, 0.0, feats)

    cx, cy = _smooth(traj.cx), _smooth(traj.cy)

    if motion_type in ("scrapbook", "pop", "fade"):
        idx = max(2, traj.n // 4)
        dx, dy = cx[idx] - cx[0], cy[idx] - cy[0]
    elif motion_type == "pan":
        dx, dy = cx[-1] - cx[0], cy[-1] - cy[0]
    else:
        diffs = np.column_stack([np.diff(cx), np.diff(cy)])
        if diffs.shape[0] == 0:
            dx = dy = 0.0
        else:
            cov = np.cov(diffs.T)
            evals, evecs = np.linalg.eigh(cov)
            v = evecs[:, -1]
            sign = np.sign(diffs @ v).sum()
            v = v if sign >= 0 else -v
            dx, dy = float(v[0]), float(v[1])

    diag = math.hypot(traj.canvas_w, traj.canvas_h)
    magnitude = math.hypot(dx, dy) / max(diag, 1.0)
    feats["dx"] = float(dx)
    feats["dy"] = float(dy)
    feats["magnitude"] = float(magnitude)

    # Below this threshold the centroid drift is indistinguishable from
    # contour-jitter noise (~3 px on a 1080p canvas).
    if magnitude < 0.01:
        return DirectionResult("none", None, 0.0, feats)

    angle = math.degrees(math.atan2(dy, dx))
    label = _direction_label(angle)
    return DirectionResult(label, float(angle), 0.0, feats)


def direction_score(predicted: str, gt: str, gt_motion_type: str = "") -> float:
    """Score predicted direction against the LICA-canonical GT direction.

    The GT vocabulary is limited to what LICA actually renders:

    * For ``rotate`` GT, ``"clockwise"``/``"anticlockwise"``; exact match
      only (no partial credit).
    * For ``pan`` GT, ``"left"`` or ``"right"``; predictions are 8-way
      compass labels, so off-by-one neighbours (e.g. ``up_left`` vs
      ``left``) get partial credit of 0.5.
    * For ``static`` GT, ``"none"`` is required.
    * For every other (non-directional) animation type, direction is
      effectively N/A and we return NaN.
    """
    if gt in (None, "", "unknown"):
        return float("nan")
    # Rotate-family and pan-family have explicit directional GT labels
    # (clockwise/anticlockwise or left/right) even when the observable
    # motion type is "scrapbook". Score them directly.
    if gt in ("clockwise", "anticlockwise"):
        if predicted == gt:
            return 1.0
        return 0.0
    if gt in ("left", "right"):
        if predicted == gt:
            return 1.0
        order = ["right", "down_right", "down", "down_left",
                 "left", "up_left", "up", "up_right"]
        if predicted in order:
            d = abs(order.index(predicted) - order.index(gt))
            d = min(d, 8 - d)
            if d == 1:
                return 0.5
        return 0.0
    if gt_motion_type in NON_DIRECTIONAL_OBSERVABLES - {"static"}:
        return float("nan")  # LICA ignores direction for these classes.
    if gt == "none":
        if gt_motion_type == "static":
            return 1.0 if predicted == "none" else 0.0
        return float("nan")
    if predicted == gt:
        return 1.0
    order = ["right", "down_right", "down", "down_left",
             "left", "up_left", "up", "up_right"]
    if predicted in order and gt in order:
        d = abs(order.index(predicted) - order.index(gt))
        d = min(d, 8 - d)
        if d == 1:
            return 0.5
    return 0.0


# =====================================================================
#  3. Animation duration & 4. Component duration
# =====================================================================


@dataclass
class DurationResult:
    predicted_s: Optional[float]
    gt_s: Optional[float]
    abs_error_s: Optional[float]
    rel_error: Optional[float]
    score: float                # 1 - min(rel_error, 1) when both available


def _motion_energy(traj: Trajectory) -> np.ndarray:
    """A scalar per frame combining position, scale and rotation rates."""
    if traj.n < 2:
        return np.zeros(traj.n)
    diag = math.hypot(traj.canvas_w, traj.canvas_h)
    pos = np.zeros(traj.n)
    pos[1:] = np.hypot(np.diff(traj.cx), np.diff(traj.cy)) / max(diag, 1.0)

    scale = _scale_curve(traj)
    sc = np.zeros(traj.n)
    sc[1:] = np.abs(np.diff(scale))

    angle = _unwrap_angle(traj.angle_deg)
    ang = np.zeros(traj.n)
    if angle.size > 1:
        ang[1:] = np.abs(np.diff(angle)) / 90.0  # normalise rough scale

    op = _opacity_proxy(traj)
    op_d = np.zeros(traj.n)
    if op.size > 1:
        op_d[1:] = np.abs(np.diff(op))

    return _smooth(pos + 0.5 * sc + 0.3 * ang + 0.5 * op_d, window=3)


def estimate_animation_duration(traj: Trajectory) -> Optional[float]:
    """Return the time (s) from animation onset to settling."""
    if traj.n < 4:
        return None
    energy = _motion_energy(traj)
    if energy.size == 0 or energy.max() <= 0:
        return None
    norm = energy / max(energy.max(), 1e-9)
    threshold = 0.15

    onset = int(np.argmax(norm > threshold)) if (norm > threshold).any() else 0
    # Walk backwards to find first frame from the end below threshold.
    above = np.where(norm > threshold)[0]
    if above.size == 0:
        return 0.0
    settle = int(above[-1])
    if settle <= onset:
        return 0.0
    duration_frames = settle - onset
    return duration_frames / max(traj.fps, 1e-6)


def estimate_component_duration(traj: Trajectory) -> Optional[float]:
    """Return the visible-time (s) of the component (longest contiguous run)."""
    if traj.present.size == 0:
        return None
    present = traj.present.astype(int)
    # longest run of 1s
    best = run = 0
    for v in present:
        run = run + 1 if v else 0
        best = max(best, run)
    return best / max(traj.fps, 1e-6)


def duration_score(pred: Optional[float], gt: Optional[float],
                   tolerance_s: float = 0.25) -> DurationResult:
    if pred is None or gt is None:
        return DurationResult(pred, gt, None, None, float("nan"))
    err = abs(pred - gt)
    rel = err / max(gt, 1e-6) if gt > 0 else (0.0 if pred == 0 else 1.0)
    score = float(err <= tolerance_s) if gt <= 1.0 else max(0.0, 1.0 - rel)
    return DurationResult(round(pred, 3), round(gt, 3), round(err, 3),
                          round(rel, 3), round(score, 3))


# =====================================================================
#  Top-level: evaluate one sample
# =====================================================================


@dataclass
class TrackingQuality:
    """Lightweight diagnostics about the tracker output that fed the metrics."""
    n_frames_total: int
    n_frames_tracked: int
    presence_frac: float      # fraction of total frames with a detection
    is_reliable: bool         # heuristic: presence >= 0.3


@dataclass
class SampleEvaluation:
    sample_id: str
    component_id: Optional[str]
    ground_truth: GroundTruth
    n_frames_tracked: int
    tracking_quality: TrackingQuality
    motion_type: MotionTypeResult
    direction: DirectionResult
    animation_duration: DurationResult
    component_duration: DurationResult


def evaluate_sample(
    layout_path: str | Path,
    tracks_json_path: str | Path,
    sample_id: Optional[str] = None,
    component_id: Optional[str] = None,
) -> SampleEvaluation:
    gt = extract_ground_truth(layout_path, sample_id=sample_id, component_id=component_id)
    if isinstance(tracks_json_path, (str, Path)):
        tracks = json.loads(Path(tracks_json_path).read_text())
    else:
        tracks = tracks_json_path  # already a dict

    # Trajectory coordinates come from the tracker in *video pixel space*.
    # The layout JSON canvas may be at a different resolution (e.g. layout
    # 1080x1920 but Sora 9:16 video is 720x1280), so we always normalise
    # using the video's own resolution rather than the GT canvas dims.
    info = tracks.get("video_info", {}) or {}
    canvas_w = float(info.get("width") or gt.canvas_w or 1080)
    canvas_h = float(info.get("height") or gt.canvas_h or 1920)

    traj = build_trajectory(
        tracks, component_id=gt.component_id,
        canvas_w=canvas_w, canvas_h=canvas_h,
    )

    # 1. motion type
    mt = classify_motion_type(traj)
    mt.score = motion_type_score(mt.predicted, gt.motion_type)

    # 2. direction
    dr = classify_direction(traj, mt.predicted)
    dr.score = direction_score(dr.predicted, gt.direction, gt.motion_type)

    # 3 & 4. durations
    pred_anim = estimate_animation_duration(traj)
    pred_comp = estimate_component_duration(traj)
    anim_dur = duration_score(pred_anim, gt.animation_duration_s)
    comp_dur = duration_score(
        pred_comp,
        gt.component_visible_until_s - gt.component_visible_from_s if gt.component_visible_until_s > 0 else gt.layout_duration,
    )

    presence_frac = float(traj.present.mean()) if traj.present.size else 0.0
    quality = TrackingQuality(
        n_frames_total=int(traj.present.size),
        n_frames_tracked=int(traj.n),
        presence_frac=round(presence_frac, 3),
        is_reliable=presence_frac >= 0.3,
    )

    return SampleEvaluation(
        sample_id=gt.sample_id,
        component_id=gt.component_id,
        ground_truth=gt,
        n_frames_tracked=traj.n,
        tracking_quality=quality,
        motion_type=mt,
        direction=dr,
        animation_duration=anim_dur,
        component_duration=comp_dur,
    )


def evaluation_to_dict(ev: SampleEvaluation) -> Dict[str, Any]:
    return {
        "sample_id": ev.sample_id,
        "component_id": ev.component_id,
        "n_frames_tracked": ev.n_frames_tracked,
        "tracking_quality": asdict(ev.tracking_quality),
        "ground_truth": asdict(ev.ground_truth),
        "motion_type": asdict(ev.motion_type),
        "direction": asdict(ev.direction),
        "animation_duration": asdict(ev.animation_duration),
        "component_duration": asdict(ev.component_duration),
    }


# =====================================================================
#  Aggregate report
# =====================================================================


def aggregate(evaluations: List[SampleEvaluation]) -> Dict[str, Any]:
    """Per-dimension averages plus a per-motion-type confusion matrix."""
    n = len(evaluations)
    if n == 0:
        return {"n": 0}

    def _mean(xs: List[float]) -> float:
        xs = [x for x in xs if not (x is None or (isinstance(x, float) and math.isnan(x)))]
        return float(np.mean(xs)) if xs else float("nan")

    def _block(evs: List[SampleEvaluation]) -> Dict[str, Any]:
        if not evs:
            return {"n": 0}
        motion_scores = [e.motion_type.score for e in evs]
        dir_scores = [e.direction.score for e in evs]
        anim_scores = [e.animation_duration.score for e in evs]
        comp_scores = [e.component_duration.score for e in evs]
        anim_errs = [e.animation_duration.abs_error_s for e in evs
                     if e.animation_duration.abs_error_s is not None]
        comp_errs = [e.component_duration.abs_error_s for e in evs
                     if e.component_duration.abs_error_s is not None]

        confusion: Dict[str, Dict[str, int]] = {}
        for e in evs:
            gt = e.ground_truth.motion_type or "unknown"
            pr = e.motion_type.predicted
            confusion.setdefault(gt, {}).setdefault(pr, 0)
            confusion[gt][pr] += 1

        return {
            "n": len(evs),
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

    return {
        "all": _block(evaluations),
        "tracker_reliable": _block([e for e in evaluations if e.tracking_quality.is_reliable]),
        "tracker_unreliable_n": sum(1 for e in evaluations if not e.tracking_quality.is_reliable),
    }


# =====================================================================
#  Standalone CLI
# =====================================================================


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Animation motion metrics")
    p.add_argument("--tracks", required=True, help="Tracker output JSON")
    p.add_argument("--layout", required=True, help="LICA layout JSON (ground truth)")
    p.add_argument("--component-id", default=None, help="Component id (otherwise auto)")
    p.add_argument("--sample-id", default=None)
    p.add_argument("--output", default=None, help="Write per-sample metric JSON here")
    args = p.parse_args(argv)

    ev = evaluate_sample(
        layout_path=args.layout,
        tracks_json_path=args.tracks,
        sample_id=args.sample_id,
        component_id=args.component_id,
    )

    out = evaluation_to_dict(ev)
    formatted = json.dumps(out, indent=2)
    print(formatted)
    if args.output:
        Path(args.output).write_text(formatted)
        print(f"Metrics JSON -> {args.output}")


if __name__ == "__main__":
    main()
