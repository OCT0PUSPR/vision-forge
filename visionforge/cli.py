"""Command-line interface for vision-forge.

Examples
--------
    # run detection on the synthetic demo stream (no hardware / weights needed
    # for the iterator; YOLO weights auto-download on first real inference)
    python -m visionforge.cli detect --source demo --task detection

    # a single image, saving an annotated copy
    python -m visionforge.cli detect --source path/to/img.jpg --save out.jpg

    # webcam tracking
    python -m visionforge.cli detect --source 0 --task tracking

    # launch the API + web GUI
    python -m visionforge.cli serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from visionforge import __version__
from visionforge.models.registry import VALID_BACKENDS, VALID_TASKS


def _add_detect_parser(sub: "argparse._SubParsersAction") -> None:
    p = sub.add_parser("detect", help="Run inference on a source.")
    p.add_argument(
        "--source",
        default="demo",
        help="'demo' (synthetic), an image/video path, or a webcam index (e.g. 0).",
    )
    p.add_argument("--task", default="detection", choices=list(VALID_TASKS))
    p.add_argument(
        "--backend",
        default=None,
        choices=list(VALID_BACKENDS),
        help="Force a backend (default: auto by task).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames (default: all / 30 for demo).",
    )
    p.add_argument(
        "--save",
        default=None,
        help="Path to save the last annotated frame (image).",
    )
    p.add_argument(
        "--no-annotate",
        action="store_true",
        help="Skip drawing (faster; just print detections).",
    )


def _add_serve_parser(sub: "argparse._SubParsersAction") -> None:
    p = sub.add_parser("serve", help="Launch the FastAPI server + web GUI.")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--reload", action="store_true")


def _add_demo_parser(sub: "argparse._SubParsersAction") -> None:
    p = sub.add_parser("demo-image", help="Write a synthetic demo image to disk (no model needed).")
    p.add_argument("--out", default="demo.jpg")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)


def cmd_detect(args: argparse.Namespace) -> int:
    from visionforge.core.draw import summarize
    from visionforge.pipeline import VisionPipeline

    source = args.source
    max_frames = args.max_frames
    if source == "demo" and max_frames is None:
        max_frames = 30

    pipeline = VisionPipeline(task=args.task, backend=args.backend)
    print(
        f"vision-forge {__version__} | task={args.task} "
        f"backend={pipeline.backend_name} device={pipeline.settings.resolved_device}"
    )

    last_frame = None
    last_result = None
    annotate = not args.no_annotate
    n = 0
    for idx, result, annotated in pipeline.run_stream(source, max_frames=max_frames, annotate=annotate):
        n += 1
        last_result = result
        if annotated is not None:
            last_frame = annotated
        print(f"[frame {idx:04d}] {summarize(result)}")

    if last_result is not None:
        print(f"\nProcessed {n} frame(s). Last: {summarize(last_result)}")

    if args.save and last_frame is not None:
        try:
            import cv2

            # last_frame is a numpy array at runtime but typed as `object`
            # (pipeline's optional annotated-frame); cv2 stubs are strict.
            cv2.imwrite(args.save, cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))  # type: ignore[call-overload]
            print(f"Saved annotated frame to {args.save}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not save image: {exc}", file=sys.stderr)
            return 1
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except Exception as exc:  # noqa: BLE001
        print(f"uvicorn is required to serve: {exc}", file=sys.stderr)
        return 1
    from visionforge.config import get_settings

    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port
    print(f"Serving vision-forge on http://{host}:{port}  (web GUI at /)")
    uvicorn.run(
        "visionforge.api.server:app",
        host=host,
        port=port,
        reload=args.reload,
    )
    return 0


def cmd_demo_image(args: argparse.Namespace) -> int:
    from visionforge.core.video import make_synthetic_frame

    frame = make_synthetic_frame(args.width, args.height, seed=7, frame_index=3)
    try:
        import cv2

        cv2.imwrite(args.out, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    except Exception:
        from PIL import Image

        Image.fromarray(frame).save(args.out)
    print(f"Wrote synthetic demo image to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionforge",
        description="vision-forge: real-time multi-task computer vision.",
    )
    parser.add_argument("--version", action="version", version=f"vision-forge {__version__}")
    sub = parser.add_subparsers(dest="command")
    _add_detect_parser(sub)
    _add_serve_parser(sub)
    _add_demo_parser(sub)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    if args.command == "detect":
        return cmd_detect(args)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "demo-image":
        return cmd_demo_image(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
