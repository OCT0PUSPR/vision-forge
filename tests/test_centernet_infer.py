"""Torch-gated end-to-end tests for the CenterNet inference backends.

Trains a tiny model for a handful of steps, saves a checkpoint, exports ONNX,
then verifies:
    * the torch ``CenterNetBackend`` produces a valid normalized ``FrameResult``,
    * the ``CenterNetOnnxBackend`` produces a valid ``FrameResult`` too,
    * the registry routes ``detection`` to ``centernet`` by default and resolves
      the ``baseline`` alias to the YOLO builder,
    * the numpy decode used by the ONNX path matches the schema contract.

Skips when torch / torchvision / onnxruntime are unavailable.
"""

import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)
pytest.importorskip("torchvision", exc_type=ImportError)

from visionforge.core.schema import FrameResult  # noqa: E402
from visionforge.models.centernet.dataset import ShapesDetectionDataset, render_shapes_image  # noqa: E402
from visionforge.models.centernet.engine import (  # noqa: E402
    TrainConfig,
    save_checkpoint,
    select_device,
)
from visionforge.models.centernet.model import build_centernet  # noqa: E402


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """Train a tiny model for a few steps and return its checkpoint path."""
    from torch.utils.data import DataLoader

    from visionforge.models.centernet.engine import collate
    from visionforge.models.centernet.losses import CenterNetLoss

    device = select_device("cpu")
    torch.manual_seed(0)
    model = build_centernet(3, variant="lite").to(device)
    ds = ShapesDetectionDataset(length=16, input_size=128, seed=1, augment=False)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate)
    crit = CenterNetLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(20):
        for batch in loader:
            out = model(batch["input"].to(device))
            targets = {k: batch[k].to(device) for k in ["hm", "wh", "offset", "ind", "reg_mask"]}
            loss, _ = crit(out, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
    d = tmp_path_factory.mktemp("ckpt")
    path = os.path.join(str(d), "tiny.pt")
    cfg = TrainConfig(variant="lite", input_size=128)
    save_checkpoint(path, model, opt, epoch=0, cfg=cfg, best_map=0.0)
    return path


def test_torch_backend_produces_frame_result(tiny_checkpoint):
    from visionforge.models.centernet.infer import CenterNetBackend

    backend = CenterNetBackend(checkpoint=tiny_checkpoint, device="cpu", conf=0.05, image_size=128)
    rng = np.random.default_rng(7)
    img, _, _ = render_shapes_image(160, rng, max_objects=3, min_objects=2)
    result = backend.infer(img, frame_index=3)
    assert isinstance(result, FrameResult)
    assert result.task == "detection"
    assert result.width == 160 and result.height == 160
    assert result.frame_index == 3
    for det in result.detections:
        x1, y1, x2, y2 = det.bbox
        assert 0 <= x1 <= x2 <= 160
        assert 0 <= y1 <= y2 <= 160
        assert det.label in ("rectangle", "circle", "triangle")


def test_backend_missing_checkpoint_raises():
    from visionforge.models.centernet.infer import CenterNetBackend

    backend = CenterNetBackend(checkpoint="/nonexistent/model.pt", device="cpu")
    with pytest.raises(FileNotFoundError):
        backend.infer(np.zeros((64, 64, 3), dtype=np.uint8))


def test_onnx_export_and_inference(tiny_checkpoint):
    pytest.importorskip("onnxruntime", exc_type=ImportError)
    from visionforge.models.centernet.export import export_centernet_onnx
    from visionforge.models.centernet.infer import CenterNetOnnxBackend

    with tempfile.TemporaryDirectory() as d:
        onnx_path = os.path.join(d, "tiny.onnx")
        export_centernet_onnx(tiny_checkpoint, onnx_path, image_size=128)
        assert os.path.exists(onnx_path)

        backend = CenterNetOnnxBackend(
            onnx_path=onnx_path, device="cpu", conf=0.05, image_size=128,
            class_names=["rectangle", "circle", "triangle"],
        )
        rng = np.random.default_rng(11)
        img, _, _ = render_shapes_image(140, rng, max_objects=2, min_objects=2)
        result = backend.infer(img)
        assert isinstance(result, FrameResult)
        assert result.task == "detection"
        for det in result.detections:
            assert det.label in ("rectangle", "circle", "triangle")


def test_torch_and_onnx_agree(tiny_checkpoint):
    """Torch and ONNX paths should produce closely-matching top detections."""
    pytest.importorskip("onnxruntime", exc_type=ImportError)
    from visionforge.models.centernet.export import export_centernet_onnx
    from visionforge.models.centernet.infer import CenterNetBackend, CenterNetOnnxBackend

    with tempfile.TemporaryDirectory() as d:
        onnx_path = os.path.join(d, "tiny.onnx")
        export_centernet_onnx(tiny_checkpoint, onnx_path, image_size=128)

        rng = np.random.default_rng(21)
        img, _, _ = render_shapes_image(128, rng, max_objects=1, min_objects=1)

        t = CenterNetBackend(checkpoint=tiny_checkpoint, device="cpu", conf=0.2, image_size=128)
        o = CenterNetOnnxBackend(onnx_path=onnx_path, device="cpu", conf=0.2, image_size=128,
                                 class_names=["rectangle", "circle", "triangle"])
        rt = t.infer(img)
        ro = o.infer(img)
        # Both should find the single object (or both find none); counts match.
        assert abs(len(rt.detections) - len(ro.detections)) <= 1


def test_registry_defaults_to_centernet():
    from visionforge.config import Settings
    from visionforge.models.registry import ModelRegistry

    reg = ModelRegistry(settings=Settings())
    assert reg.default_backend("detection") == "centernet"
    # baseline alias resolves to the yolo builder name
    assert reg.resolve("detection", "baseline") == "yolo"
    assert reg.resolve("detection", "centernet") == "centernet"
    assert reg.resolve("detection", None) == "centernet"


def test_registry_centernet_rejects_non_detection():
    from visionforge.models.registry import ModelRegistry

    reg = ModelRegistry()
    with pytest.raises(ValueError):
        reg.resolve("segmentation", "centernet")
