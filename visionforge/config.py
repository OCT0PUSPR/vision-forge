"""Application configuration via pydantic-settings.

Every value can be overridden through environment variables (optionally from a
local ``.env`` file). See ``.env.example`` for the full list of knobs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except Exception:  # pragma: no cover - exercised only without the dep
    _HAS_PYDANTIC_SETTINGS = False


def detect_device(preferred: Optional[str] = None) -> str:
    """Return the best available torch device string.

    Order of preference: explicit ``preferred`` -> CUDA -> Apple MPS -> CPU.
    Falls back to ``"cpu"`` gracefully when torch is not installed, so this is
    safe to call in the lightweight (no-torch) environment.
    """
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # ``torch.backends.mps`` only exists on recent torch builds.
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# Default model identifiers. These are auto-downloaded at runtime by
# Ultralytics / HuggingFace; nothing is committed to the repo.
DEFAULT_MODELS = {
    "detection": "yolov8n.pt",
    "segmentation": "yolov8n-seg.pt",
    "pose": "yolov8n-pose.pt",
    "tracking": "yolov8n.pt",
    "classification": "google/vit-base-patch16-224",
}

# HuggingFace alternate detector backend.
HF_DETECTION_MODEL = "facebook/detr-resnet-50"


if _HAS_PYDANTIC_SETTINGS:

    class Settings(BaseSettings):
        """Strongly-typed, env-driven configuration."""

        model_config = SettingsConfigDict(
            env_prefix="VF_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # --- inference defaults ---
        device: str = "auto"
        conf_threshold: float = 0.25
        iou_threshold: float = 0.45
        max_detections: int = 300
        image_size: int = 640

        # --- model ids ---
        detection_model: str = DEFAULT_MODELS["detection"]
        segmentation_model: str = DEFAULT_MODELS["segmentation"]
        pose_model: str = DEFAULT_MODELS["pose"]
        tracking_model: str = DEFAULT_MODELS["tracking"]
        classification_model: str = DEFAULT_MODELS["classification"]
        hf_detection_model: str = HF_DETECTION_MODEL

        # --- tracking ---
        tracker: str = "bytetrack.yaml"

        # --- server ---
        host: str = "0.0.0.0"
        port: int = 8000
        cors_origins: str = "*"
        max_upload_mb: int = 25

        # --- auth / secrets (read from env, never hardcode) ---
        hf_token: Optional[str] = None

        @property
        def resolved_device(self) -> str:
            return detect_device(self.device)

        def model_for(self, task: str) -> str:
            mapping = {
                "detection": self.detection_model,
                "segmentation": self.segmentation_model,
                "pose": self.pose_model,
                "tracking": self.tracking_model,
                "classification": self.classification_model,
            }
            if task not in mapping:
                raise ValueError(f"Unknown task: {task!r}")
            return mapping[task]

else:  # pragma: no cover - fallback for the no-pydantic environment

    class Settings:  # type: ignore[no-redef]
        """Minimal env-driven fallback when pydantic-settings is unavailable."""

        def __init__(self) -> None:
            g = os.environ.get
            self.device = g("VF_DEVICE", "auto")
            self.conf_threshold = float(g("VF_CONF_THRESHOLD", "0.25"))
            self.iou_threshold = float(g("VF_IOU_THRESHOLD", "0.45"))
            self.max_detections = int(g("VF_MAX_DETECTIONS", "300"))
            self.image_size = int(g("VF_IMAGE_SIZE", "640"))
            self.detection_model = g("VF_DETECTION_MODEL", DEFAULT_MODELS["detection"])
            self.segmentation_model = g(
                "VF_SEGMENTATION_MODEL", DEFAULT_MODELS["segmentation"]
            )
            self.pose_model = g("VF_POSE_MODEL", DEFAULT_MODELS["pose"])
            self.tracking_model = g("VF_TRACKING_MODEL", DEFAULT_MODELS["tracking"])
            self.classification_model = g(
                "VF_CLASSIFICATION_MODEL", DEFAULT_MODELS["classification"]
            )
            self.hf_detection_model = g("VF_HF_DETECTION_MODEL", HF_DETECTION_MODEL)
            self.tracker = g("VF_TRACKER", "bytetrack.yaml")
            self.host = g("VF_HOST", "0.0.0.0")
            self.port = int(g("VF_PORT", "8000"))
            self.cors_origins = g("VF_CORS_ORIGINS", "*")
            self.max_upload_mb = int(g("VF_MAX_UPLOAD_MB", "25"))
            self.hf_token = g("HF_TOKEN") or g("VF_HF_TOKEN")

        @property
        def resolved_device(self) -> str:
            return detect_device(self.device)

        def model_for(self, task: str) -> str:
            mapping = {
                "detection": self.detection_model,
                "segmentation": self.segmentation_model,
                "pose": self.pose_model,
                "tracking": self.tracking_model,
                "classification": self.classification_model,
            }
            if task not in mapping:
                raise ValueError(f"Unknown task: {task!r}")
            return mapping[task]


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    """Return a cached singleton ``Settings`` instance."""
    return Settings()
