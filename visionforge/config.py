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
    except Exception:  # nosec B110 - torch optional; fall back to CPU on any probe error
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

# From-scratch CenterNet detector (the DEFAULT detection backend). The proof
# checkpoint trained on the procedural shapes dataset ships in the repo; point
# VF_CENTERNET_CHECKPOINT at a VOC/COCO-trained checkpoint for real classes.
CENTERNET_CHECKPOINT = "weights/centernet_shapes.pt"


if _HAS_PYDANTIC_SETTINGS:

    class Settings(BaseSettings):
        """Strongly-typed, env-driven configuration."""

        model_config = SettingsConfigDict(
            env_prefix="VF_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # --- runtime profile ---
        env: str = "development"  # development | production
        log_level: str = "INFO"
        json_logs: bool = True

        # --- inference defaults ---
        device: str = "auto"
        conf_threshold: float = 0.25
        iou_threshold: float = 0.45
        max_detections: int = 300
        image_size: int = 640
        inference_timeout_s: float = 30.0
        warmup_tasks: str = "detection"
        model_cache_size: int = 4

        # --- model ids ---
        detection_model: str = DEFAULT_MODELS["detection"]
        segmentation_model: str = DEFAULT_MODELS["segmentation"]
        pose_model: str = DEFAULT_MODELS["pose"]
        tracking_model: str = DEFAULT_MODELS["tracking"]
        classification_model: str = DEFAULT_MODELS["classification"]
        hf_detection_model: str = HF_DETECTION_MODEL
        onnx_model_path: str = "weights/yolov8n.onnx"

        # --- from-scratch CenterNet (default detector) ---
        default_detector: str = "centernet"  # centernet | baseline (yolo) | hf | onnx
        centernet_checkpoint: str = CENTERNET_CHECKPOINT
        centernet_onnx_path: str = "weights/centernet_shapes.onnx"
        centernet_conf: float = 0.30
        centernet_iou: float = 0.50
        centernet_image_size: int = 256
        centernet_topk: int = 100

        # --- tracking ---
        tracker: str = "bytetrack.yaml"

        # --- server ---
        host: str = "0.0.0.0"
        port: int = 8000
        cors_origins: str = "*"
        max_upload_mb: int = 25
        max_image_side: int = 8000

        # --- security ---
        require_auth: bool = False
        api_keys: Optional[str] = None  # comma-separated plaintext keys (dev)
        rate_limit_per_min: int = 120
        enable_security_headers: bool = True
        enable_hsts: bool = False

        # --- persistence / queue / storage ---
        database_url: str = "sqlite:///./visionforge.db"
        redis_url: Optional[str] = None
        s3_endpoint: Optional[str] = None
        s3_bucket: Optional[str] = None
        s3_access_key: Optional[str] = None
        s3_secret_key: Optional[str] = None

        # --- observability ---
        enable_metrics: bool = True
        otel_exporter_endpoint: Optional[str] = None

        # --- auth / secrets (read from env, never hardcode) ---
        hf_token: Optional[str] = None

        @property
        def resolved_device(self) -> str:
            return detect_device(self.device)

        @property
        def is_production(self) -> bool:
            return self.env.lower() in ("production", "prod")

        @property
        def warmup_task_list(self) -> list:
            return [t.strip() for t in self.warmup_tasks.split(",") if t.strip()]

        @property
        def cors_origin_list(self) -> list:
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

        def validate_startup(self) -> None:
            """Validate critical settings on boot; raise on misconfiguration."""
            if not 0.0 <= self.conf_threshold <= 1.0:
                raise ValueError("VF_CONF_THRESHOLD must be in [0, 1]")
            if not 0.0 <= self.iou_threshold <= 1.0:
                raise ValueError("VF_IOU_THRESHOLD must be in [0, 1]")
            if self.image_size <= 0:
                raise ValueError("VF_IMAGE_SIZE must be positive")
            if self.is_production and self.cors_origins.strip() == "*":
                raise ValueError(
                    "Wildcard CORS origin is not allowed in production; set VF_CORS_ORIGINS to an explicit allowlist."
                )
            if self.is_production and not self.require_auth:
                raise ValueError("VF_REQUIRE_AUTH must be true in production.")

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
            self.segmentation_model = g("VF_SEGMENTATION_MODEL", DEFAULT_MODELS["segmentation"])
            self.pose_model = g("VF_POSE_MODEL", DEFAULT_MODELS["pose"])
            self.tracking_model = g("VF_TRACKING_MODEL", DEFAULT_MODELS["tracking"])
            self.classification_model = g("VF_CLASSIFICATION_MODEL", DEFAULT_MODELS["classification"])
            self.hf_detection_model = g("VF_HF_DETECTION_MODEL", HF_DETECTION_MODEL)
            self.onnx_model_path = g("VF_ONNX_MODEL_PATH", "weights/yolov8n.onnx")
            self.default_detector = g("VF_DEFAULT_DETECTOR", "centernet")
            self.centernet_checkpoint = g("VF_CENTERNET_CHECKPOINT", CENTERNET_CHECKPOINT)
            self.centernet_onnx_path = g("VF_CENTERNET_ONNX_PATH", "weights/centernet_shapes.onnx")
            self.centernet_conf = float(g("VF_CENTERNET_CONF", "0.30"))
            self.centernet_iou = float(g("VF_CENTERNET_IOU", "0.50"))
            self.centernet_image_size = int(g("VF_CENTERNET_IMAGE_SIZE", "256"))
            self.centernet_topk = int(g("VF_CENTERNET_TOPK", "100"))
            self.tracker = g("VF_TRACKER", "bytetrack.yaml")
            self.host = g("VF_HOST", "0.0.0.0")
            self.port = int(g("VF_PORT", "8000"))
            self.cors_origins = g("VF_CORS_ORIGINS", "*")
            self.max_upload_mb = int(g("VF_MAX_UPLOAD_MB", "25"))
            self.max_image_side = int(g("VF_MAX_IMAGE_SIDE", "8000"))
            self.hf_token = g("HF_TOKEN") or g("VF_HF_TOKEN")
            # production extras
            self.env = g("VF_ENV", "development")
            self.log_level = g("VF_LOG_LEVEL", "INFO")
            self.json_logs = g("VF_JSON_LOGS", "true").lower() == "true"
            self.inference_timeout_s = float(g("VF_INFERENCE_TIMEOUT_S", "30.0"))
            self.warmup_tasks = g("VF_WARMUP_TASKS", "detection")
            self.model_cache_size = int(g("VF_MODEL_CACHE_SIZE", "4"))
            self.require_auth = g("VF_REQUIRE_AUTH", "false").lower() == "true"
            self.api_keys = g("VF_API_KEYS")
            self.rate_limit_per_min = int(g("VF_RATE_LIMIT_PER_MIN", "120"))
            self.enable_security_headers = g("VF_ENABLE_SECURITY_HEADERS", "true").lower() == "true"
            self.enable_hsts = g("VF_ENABLE_HSTS", "false").lower() == "true"
            self.database_url = g("DATABASE_URL", g("VF_DATABASE_URL", "sqlite:///./visionforge.db"))
            self.redis_url = g("REDIS_URL") or g("VF_REDIS_URL")
            self.s3_endpoint = g("VF_S3_ENDPOINT")
            self.s3_bucket = g("VF_S3_BUCKET")
            self.s3_access_key = g("VF_S3_ACCESS_KEY")
            self.s3_secret_key = g("VF_S3_SECRET_KEY")
            self.enable_metrics = g("VF_ENABLE_METRICS", "true").lower() == "true"
            self.otel_exporter_endpoint = g("VF_OTEL_EXPORTER_ENDPOINT")

        @property
        def resolved_device(self) -> str:
            return detect_device(self.device)

        @property
        def is_production(self) -> bool:
            return self.env.lower() in ("production", "prod")

        @property
        def warmup_task_list(self) -> list:
            return [t.strip() for t in self.warmup_tasks.split(",") if t.strip()]

        @property
        def cors_origin_list(self) -> list:
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

        def validate_startup(self) -> None:
            if not 0.0 <= self.conf_threshold <= 1.0:
                raise ValueError("VF_CONF_THRESHOLD must be in [0, 1]")
            if not 0.0 <= self.iou_threshold <= 1.0:
                raise ValueError("VF_IOU_THRESHOLD must be in [0, 1]")
            if self.image_size <= 0:
                raise ValueError("VF_IMAGE_SIZE must be positive")
            if self.is_production and self.cors_origins.strip() == "*":
                raise ValueError("Wildcard CORS not allowed in production")
            if self.is_production and not self.require_auth:
                raise ValueError("VF_REQUIRE_AUTH must be true in production")

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
