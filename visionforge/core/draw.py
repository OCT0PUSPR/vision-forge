"""Drawing/annotation helpers.

The color and label-formatting helpers are pure Python (stdlib only) so they
can be unit-tested without numpy/opencv. The actual image rendering uses
OpenCV when available and degrades gracefully otherwise.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from visionforge.core.schema import FrameResult

RGB = Tuple[int, int, int]

# A fixed, visually distinct palette (RGB). Indexed by class id so a given
# class always gets the same color across frames.
PALETTE: List[RGB] = [
    (255, 56, 56),
    (255, 159, 56),
    (255, 215, 56),
    (151, 255, 56),
    (56, 255, 116),
    (56, 255, 235),
    (56, 159, 255),
    (56, 76, 255),
    (151, 56, 255),
    (235, 56, 255),
    (255, 56, 159),
    (160, 160, 160),
]

# COCO-style 17-keypoint skeleton (pairs of keypoint indices to connect).
COCO_SKELETON: List[Tuple[int, int]] = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]


def color_for_index(index: int) -> RGB:
    """Deterministic palette color for a class/track index."""
    if index < 0:
        index = -index
    return PALETTE[index % len(PALETTE)]


def color_for_label(label: str) -> RGB:
    """Stable color derived from a label string (no class id needed)."""
    return color_for_index(sum(ord(c) for c in label))


def rgb_to_bgr(color: RGB) -> RGB:
    """OpenCV uses BGR ordering; flip an RGB triple."""
    r, g, b = color
    return (b, g, r)


def contrasting_text_color(bg: RGB) -> RGB:
    """Return black or white, whichever is more readable on ``bg``.

    Uses the standard perceptual luminance formula.
    """
    r, g, b = bg
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def format_label(
    label: str,
    confidence: Optional[float] = None,
    track_id: Optional[int] = None,
) -> str:
    """Build the on-box caption, e.g. ``"#3 person 0.87"``."""
    parts: List[str] = []
    if track_id is not None:
        parts.append(f"#{track_id}")
    parts.append(label)
    if confidence is not None:
        parts.append(f"{confidence:.2f}")
    return " ".join(parts)


def scale_bbox(
    bbox: Sequence[float],
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """Rescale an xyxy box between two image sizes (w, h)."""
    fw, fh = from_size
    tw, th = to_size
    sx = tw / fw if fw else 1.0
    sy = th / fh if fh else 1.0
    x1, y1, x2, y2 = bbox
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def _try_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:  # pragma: no cover - depends on environment
        return None


def draw_detections(
    image,
    result: FrameResult,
    *,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    draw_masks: bool = True,
    draw_keypoints: bool = True,
):
    """Annotate ``image`` (an RGB numpy array) in place-ish and return it.

    Requires numpy + opencv at call time. The pure helpers above are the parts
    covered by unit tests; this function is the rendering glue.
    """
    cv2 = _try_import_cv2()
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(
            "draw_detections requires opencv-python(-headless). Install it or "
            "use the pure helpers (color_for_index, format_label, ...) instead."
        )
    import numpy as np

    canvas = np.ascontiguousarray(image)
    overlay = canvas.copy()

    for det in result.detections:
        idx = det.class_id if det.class_id is not None else 0
        if det.track_id is not None:
            idx = det.track_id
        color = color_for_index(idx)
        bgr = rgb_to_bgr(color)
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)

        # Mask polygon fill (drawn into the overlay for alpha blending).
        if draw_masks and det.mask:
            try:
                pts = np.array(det.mask, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], bgr)
            except Exception:
                pass

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, line_thickness)

        caption = format_label(det.label, det.confidence, det.track_id)
        (tw, th), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        ty = max(y1, th + 4)
        cv2.rectangle(
            canvas, (x1, ty - th - baseline - 2), (x1 + tw + 2, ty), bgr, -1
        )
        text_color = rgb_to_bgr(contrasting_text_color(color))
        cv2.putText(
            canvas,
            caption,
            (x1 + 1, ty - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

        # Keypoints + skeleton.
        if draw_keypoints and det.keypoints:
            kps = det.keypoints
            for a, b in COCO_SKELETON:
                if a < len(kps) and b < len(kps):
                    ka, kb = kps[a], kps[b]
                    if ka.confidence > 0.2 and kb.confidence > 0.2:
                        cv2.line(
                            canvas,
                            (int(ka.x), int(ka.y)),
                            (int(kb.x), int(kb.y)),
                            (255, 255, 255),
                            max(1, line_thickness - 1),
                            cv2.LINE_AA,
                        )
            for kp in kps:
                if kp.confidence > 0.2:
                    cv2.circle(canvas, (int(kp.x), int(kp.y)), 3, bgr, -1, cv2.LINE_AA)

    if draw_masks:
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    return canvas


def summarize(result: FrameResult) -> str:
    """One-line human summary, e.g. ``"3 person, 1 dog (12.4ms)"``."""
    counts = result.count_by_label()
    if not counts:
        body = "no detections"
    else:
        body = ", ".join(f"{n} {label}" for label, n in sorted(counts.items()))
    return f"{body} ({result.inference_ms:.1f}ms)"
