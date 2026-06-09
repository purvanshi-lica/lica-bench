"""motion_gen_eval — tracker + motion-correctness evaluation for animated videos.

Public surface (most users only need these):

    from motion_gen_eval.motion_metrics import (
        evaluate_sample,
        evaluation_to_dict,
        SampleEvaluation,
        aggregate,
    )

The ``layout``, ``frame_detector``, ``contour_tracker``, ``layout_tracker``,
``video_io``, ``metrics``, and ``config`` modules are also importable for
power users who want to run the trackers directly.

The runner scripts under ``scripts/motion_gen_eval/`` are command-line
entry points; see ``scripts/motion_gen_eval/README.md`` for usage.
"""

from __future__ import annotations

__all__ = [
    "motion_metrics",
    "layout",
    "frame_detector",
    "contour_tracker",
    "layout_tracker",
    "video_io",
    "metrics",
    "config",
]
