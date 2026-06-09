"""Parse LICA layout JSON and convert components into oriented bounding-box polygons.

A layout JSON has a top-level ``components`` list where each entry has at
minimum ``type``, ``id``, and size fields.  Position (``left``, ``top``) and
``transform`` (rotation) may be absent for full-canvas items like backgrounds.

This module converts each component into a 4-corner polygon in pixel space so
we can match YOLO OBB detections against known layout elements.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LayoutComponent:
    """A single component extracted from a layout JSON."""

    type: str  # IMAGE, TEXT, GROUP, TEXT_NEW, …
    id: str  # e.g. "0-1", "0-5"
    left: float = 0.0  # px
    top: float = 0.0  # px
    width: float = 0.0  # px
    height: float = 0.0  # px
    rotation_deg: float = 0.0
    opacity: float = 1.0
    src: Optional[str] = None
    text: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.left + self.width / 2, self.top + self.height / 2)

    def polygon(self) -> np.ndarray:
        """Return the 4-corner OBB polygon as shape (4, 2) float64.

        Corners are in image-pixel coordinates, accounting for rotation about
        the component center.
        """
        cx, cy = self.center
        hw, hh = self.width / 2, self.height / 2
        corners = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh],
        ], dtype=np.float64)

        rad = math.radians(self.rotation_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = corners @ rot.T
        return rotated + np.array([cx, cy])

    def __str__(self) -> str:
        label = self.text or self.src or ""
        if len(label) > 40:
            label = label[:37] + "..."
        return f"{self.type} id={self.id} ({self.width:.0f}x{self.height:.0f}) {label}"


@dataclass
class Layout:
    """Parsed representation of a full layout JSON."""

    components: List[LayoutComponent]
    width: float  # canvas width in px
    height: float  # canvas height in px
    duration: float  # seconds
    background: Optional[str] = None

    def find(self, component_id: str) -> Optional[LayoutComponent]:
        for c in self.components:
            if c.id == component_id:
                return c
        return None

    def find_by_type(self, comp_type: str) -> List[LayoutComponent]:
        t = comp_type.upper()
        return [c for c in self.components if c.type.upper() == t]

    def summary(self) -> str:
        lines = [f"Canvas {self.width:.0f}x{self.height:.0f}  duration={self.duration}s"]
        for c in self.components:
            lines.append(f"  {c}")
        return "\n".join(lines)

    def scaled_to(self, target_w: float, target_h: float) -> "Layout":
        """Return a new Layout whose components are scaled to a target canvas.

        The LICA layout JSON uses design-canvas coordinates (e.g. 1080x1920),
        but generated videos may be rendered at a different resolution
        (e.g. Sora 9:16 = 720x1280). To match tracker detections (in video
        pixel space) against layout polygons, we proportionally scale every
        component's position and size.

        ``rotation`` and ``opacity`` are unaffected. If ``target_w/target_h``
        already equal the layout canvas, returns a structurally-equivalent
        copy.
        """
        if self.width <= 0 or self.height <= 0:
            return self
        sx = target_w / self.width
        sy = target_h / self.height
        if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
            return self
        scaled: List[LayoutComponent] = []
        for c in self.components:
            scaled.append(LayoutComponent(
                type=c.type,
                id=c.id,
                left=c.left * sx,
                top=c.top * sy,
                width=c.width * sx,
                height=c.height * sy,
                rotation_deg=c.rotation_deg,
                opacity=c.opacity,
                src=c.src,
                text=c.text,
                raw=c.raw,
            ))
        return Layout(
            components=scaled,
            width=target_w,
            height=target_h,
            duration=self.duration,
            background=self.background,
        )


def _parse_px(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace("px", "")
    try:
        return float(s)
    except ValueError:
        return default


def _parse_rotation(transform: Any) -> float:
    """Extract rotation in degrees from a CSS transform string like 'rotate(-12.5028deg)'."""
    if not transform or transform == "none":
        return 0.0
    m = re.search(r"rotate\(\s*([-\d.]+)\s*deg\s*\)", str(transform))
    return float(m.group(1)) if m else 0.0


def _parse_transform_translate(transform: str) -> Tuple[float, float]:
    """Extract translate(Xpx, Ypx) offsets from a CSS transform string."""
    m = re.search(r"translate\(\s*([-\d.e]+)px\s*,\s*([-\d.e]+)px\s*\)", transform or "")
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


def _component_from_raw(raw: Dict[str, Any]) -> LayoutComponent:
    """Build a LayoutComponent from a flat or style-based raw dict."""
    style = raw.get("style", {})
    has_style = bool(style)

    left = _parse_px(style.get("left") if has_style else raw.get("left"))
    top = _parse_px(style.get("top") if has_style else raw.get("top"))
    width = _parse_px(style.get("width") if has_style else raw.get("width"))
    height = _parse_px(style.get("height") if has_style else raw.get("height"))

    transform = style.get("transform") if has_style else raw.get("transform")
    rotation_deg = _parse_rotation(transform)
    tx, ty = _parse_transform_translate(str(transform) if transform else "")
    left += tx
    top += ty

    opacity_val = style.get("opacity") if has_style else raw.get("opacity")
    opacity = float(opacity_val) if opacity_val is not None else 1.0

    return LayoutComponent(
        type=raw.get("type", "UNKNOWN"),
        id=raw.get("id", ""),
        left=left,
        top=top,
        width=width,
        height=height,
        rotation_deg=rotation_deg,
        opacity=opacity,
        src=raw.get("src"),
        text=raw.get("text"),
        raw=raw,
    )


def _collect_components(
    nodes: List[Dict[str, Any]],
    parent_left: float = 0.0,
    parent_top: float = 0.0,
) -> List[LayoutComponent]:
    """Recursively collect leaf components, accumulating parent offsets."""
    results: List[LayoutComponent] = []
    for raw in nodes:
        comp = _component_from_raw(raw)
        abs_left = parent_left + comp.left
        abs_top = parent_top + comp.top
        children = raw.get("components", [])
        if children:
            results.extend(_collect_components(children, abs_left, abs_top))
        else:
            comp.left = abs_left
            comp.top = abs_top
            results.append(comp)
    return results


def parse_layout(path: str | Path) -> Layout:
    """Read a layout JSON file and return a ``Layout`` object.

    Handles both flat LICA layouts (top-level ``components``) and nested
    ``layout_config`` structures with ``style`` dicts.
    """
    data = json.loads(Path(path).read_text())

    # Detect nested layout_config format
    if "layout_config" in data:
        lc = data["layout_config"]
        meta = data.get("layout_metadata", {})
        canvas_w = _parse_px(meta.get("width"), 1920)
        canvas_h = _parse_px(meta.get("height"), 1080)
        duration = float(lc.get("duration", 0))
        style = lc.get("style", {})
        background = style.get("background")
        raw_components = lc.get("components", [])
        components = _collect_components(raw_components)
    else:
        canvas_w = _parse_px(data.get("width"), 1080)
        canvas_h = _parse_px(data.get("height"), 1920)
        duration = float(data.get("duration", 0))
        background = data.get("background")
        components = []
        for raw in data.get("components", []):
            components.append(_component_from_raw(raw))

    return Layout(
        components=components,
        width=canvas_w,
        height=canvas_h,
        duration=duration,
        background=background,
    )


def polygon_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Approximate IoU between two convex 4-corner polygons using Shapely if
    available, otherwise fall back to axis-aligned bounding-box IoU.
    """
    try:
        from shapely.geometry import Polygon as ShapelyPoly

        a = ShapelyPoly(poly_a)
        b = ShapelyPoly(poly_b)
        if not a.is_valid or not b.is_valid:
            a = a.buffer(0)
            b = b.buffer(0)
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except ImportError:
        pass

    # Fallback: axis-aligned bounding-box IoU
    def _aabb(poly: np.ndarray):
        return poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()

    ax0, ay0, ax1, ay1 = _aabb(poly_a)
    bx0, by0, bx1, by1 = _aabb(poly_b)
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections_to_components(
    detections: List[Dict[str, Any]],
    targets: List[LayoutComponent],
    iou_thresh: float = 0.1,
) -> Dict[str, Dict[str, Any]]:
    """Match YOLO OBB detections to layout components using polygon IoU.

    Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) for
    globally optimal matching, falling back to greedy assignment when scipy
    is not available.

    Returns a dict mapping component_id -> best matching detection dict
    (augmented with ``match_iou``).  Unmatched components are omitted.
    """
    if not detections or not targets:
        return {}

    target_ids = [c.id for c in targets]
    target_polys = [c.polygon() for c in targets]

    n_comp = len(targets)
    n_det = len(detections)
    iou_matrix = np.zeros((n_comp, n_det), dtype=np.float64)

    for ci, comp_poly in enumerate(target_polys):
        for di, det in enumerate(detections):
            det_poly = np.array(det["polygon"], dtype=np.float64).reshape(4, 2)
            iou_matrix[ci, di] = polygon_iou(comp_poly, det_poly)

    matches: Dict[str, Dict[str, Any]] = {}

    try:
        from scipy.optimize import linear_sum_assignment
        cost = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        for ci, di in zip(row_ind, col_ind):
            iou_val = iou_matrix[ci, di]
            if iou_val >= iou_thresh:
                det = dict(detections[di])
                det["match_iou"] = round(float(iou_val), 4)
                det["component_id"] = target_ids[ci]
                matches[target_ids[ci]] = det
    except ImportError:
        scored: List[Tuple[float, str, int]] = []
        for ci, comp_id in enumerate(target_ids):
            for di in range(n_det):
                if iou_matrix[ci, di] >= iou_thresh:
                    scored.append((iou_matrix[ci, di], comp_id, di))
        scored.sort(key=lambda x: -x[0])
        used_det_indices: set = set()
        for iou_val, comp_id, di in scored:
            if comp_id in matches or di in used_det_indices:
                continue
            det = dict(detections[di])
            det["match_iou"] = round(float(iou_val), 4)
            det["component_id"] = comp_id
            matches[comp_id] = det
            used_det_indices.add(di)

    return matches
