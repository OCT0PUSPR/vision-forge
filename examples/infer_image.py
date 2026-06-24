"""Run a single image through the pipeline and save an annotated copy.

    python examples/infer_image.py path/to/image.jpg --task detection --out out.jpg

If no path is given, a synthetic demo image is generated first.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a single image.")
    parser.add_argument("image", nargs="?", default=None, help="Image path (optional).")
    parser.add_argument("--task", default="detection")
    parser.add_argument(
        "--backend",
        default=None,
        help="centernet (from-scratch default) | baseline (YOLO) | hf | onnx | centernet-onnx",
    )
    parser.add_argument("--out", default="annotated.jpg")
    args = parser.parse_args()

    import cv2

    from visionforge.pipeline import VisionPipeline

    if args.image is None:
        from visionforge.core.video import make_synthetic_frame

        print("No image given; using a synthetic demo frame.")
        frame = make_synthetic_frame(640, 480, seed=11, frame_index=4)
    else:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"Could not read image: {args.image}")
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pipeline = VisionPipeline(task=args.task, backend=args.backend)
    result = pipeline.infer_array(frame)
    print("Detections:", result.count_by_label())

    annotated = pipeline.annotate(frame, result)
    cv2.imwrite(args.out, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
