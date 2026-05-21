"""Helpers for reading video frames."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
import requests


def _download_to_tmp(url: str, timeout: int = 120) -> Path:
    """Download a URL to a temp file and return its path."""
    import tempfile

    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()

    suffix = Path(url.split("?")[0]).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in resp.iter_content(chunk_size=1 << 20):
        tmp.write(chunk)
    tmp.close()
    return Path(tmp.name)


def resolve_video_source(source: str) -> Path:
    """Accept a local path or HTTP(S) URL and return a local file path."""
    stripped = source.strip()
    if stripped.startswith(("http://", "https://")):
        return _download_to_tmp(stripped)
    p = Path(stripped)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")
    return p


def iterate_frames(
    video_path: Path,
    start_frame: int = 0,
    max_frames: int = 0,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_index, bgr_frame) tuples from a video file.

    Frames are yielded as BGR numpy arrays (OpenCV convention).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    idx = start_frame
    yielded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
            yielded += 1
            if max_frames > 0 and yielded >= max_frames:
                break
    finally:
        cap.release()


def get_video_info(video_path: Path) -> dict:
    """Return fps, width, height, total_frames for a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


