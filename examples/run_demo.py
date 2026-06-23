"""Run the synthetic demo end-to-end on CPU (no camera, no model weights).

This exercises the frame iterator + drawing path with a real backend. The YOLO
weights (~6 MB) auto-download on the first inference call. If ultralytics is not
installed, we fall back to printing the synthetic frame metadata only.

    python examples/run_demo.py
"""

from __future__ import annotations


def main() -> None:
    from visionforge.core.video import iter_synthetic

    print("Generating synthetic frames...")
    frames = list(iter_synthetic(width=640, height=480, n_frames=5))
    print(f"Produced {len(frames)} synthetic frames of shape {frames[0][1].shape}")

    try:
        from visionforge.pipeline import VisionPipeline

        pipeline = VisionPipeline(task="detection")
        print(
            f"Backend={pipeline.backend_name} "
            f"device={pipeline.settings.resolved_device}"
        )
        for idx, frame in frames:
            result = pipeline.infer_array(frame, frame_index=idx)
            print(f"[frame {idx}] {result.count_by_label()} ({result.inference_ms:.1f}ms)")
    except Exception as exc:  # noqa: BLE001
        print(f"(Skipping real inference: {exc})")
        print("Install 'ultralytics' to run actual detection.")


if __name__ == "__main__":
    main()
