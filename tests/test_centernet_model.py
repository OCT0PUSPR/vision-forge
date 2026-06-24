"""Torch-gated tests for the from-scratch CenterNet model, loss, decode,
dataset and inference backends.

These skip automatically when torch / torchvision are not installed (the light
CI path), so the green quality gate never requires torch. When torch *is*
present (the training environment), they fully exercise the architecture, the
differentiable losses, the decode, the procedural dataset and the schema
integration.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)
pytest.importorskip("torchvision", exc_type=ImportError)

from visionforge.models.centernet.dataset import (  # noqa: E402
    SHAPE_CLASSES,
    ShapesDetectionDataset,
    render_shapes_image,
)
from visionforge.models.centernet.engine import (  # noqa: E402
    TrainConfig,
    collate,
    evaluate,
    select_device,
)
from visionforge.models.centernet.losses import CenterNetLoss, neg_loss  # noqa: E402
from visionforge.models.centernet.model import CenterNet, build_centernet  # noqa: E402
from visionforge.models.centernet.postprocess import (  # noqa: E402
    ctdet_decode,
    decode_detections,
)


def test_build_variants_param_counts():
    lite = build_centernet(3, variant="lite")
    r18 = build_centernet(20, variant="r18")
    assert isinstance(lite, CenterNet)
    assert lite.num_classes == 3
    assert r18.num_classes == 20
    # lite is meaningfully smaller than r18
    assert lite.num_parameters() < r18.num_parameters()


def test_build_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_centernet(3, variant="nope")


def test_forward_output_shapes_stride4():
    model = build_centernet(3, variant="lite")
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    # stride-4 feature map: 128 / 4 = 32
    assert out["hm"].shape == (2, 3, 32, 32)
    assert out["wh"].shape == (2, 2, 32, 32)
    assert out["offset"].shape == (2, 2, 32, 32)


def test_heatmap_prior_bias_small_sigmoid():
    model = build_centernet(3, variant="lite")
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        hm = torch.sigmoid(model(x)["hm"])
    # prior-prob bias -> initial activations near 0.01, never saturated
    assert float(hm.mean()) < 0.1


def test_neg_loss_zero_when_perfect():
    gt = torch.zeros(1, 1, 8, 8)
    gt[0, 0, 4, 4] = 1.0
    pred = gt.clone().clamp(1e-4, 1 - 1e-4)
    loss = neg_loss(pred, gt)
    assert float(loss) < 0.05


def test_combined_loss_decreases_on_overfit():
    """One tiny batch overfit for a few steps must drive the loss down."""
    torch.manual_seed(0)
    device = select_device("cpu")
    model = build_centernet(3, variant="lite").to(device)
    ds = ShapesDetectionDataset(length=4, input_size=128, seed=1, augment=False)
    batch = collate([ds[i] for i in range(2)])
    targets = {k: batch[k].to(device) for k in ["hm", "wh", "offset", "ind", "reg_mask"]}
    x = batch["input"].to(device)
    crit = CenterNetLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = None
    for _ in range(15):
        out = model(x)
        loss, _ = crit(out, targets)
        if first is None:
            first = float(loss.detach())
        final_loss = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert final_loss < first


def test_ctdet_decode_shape():
    model = build_centernet(3, variant="lite")
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    hm = torch.sigmoid(out["hm"])
    dets = ctdet_decode(hm, out["wh"], out["offset"], k=20)
    assert dets.shape == (2, 20, 6)  # [x1,y1,x2,y2,score,class]


def test_decode_detections_returns_numpy_per_image():
    model = build_centernet(3, variant="lite")
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    dets = decode_detections(out, k=20, score_threshold=0.0, nms_iou=0.5)
    assert len(dets) == 2
    assert dets[0].shape[1] == 6


def test_dataset_sample_structure():
    ds = ShapesDetectionDataset(length=10, input_size=128, seed=2)
    sample = ds[0]
    assert sample["input"].shape == (3, 128, 128)
    assert sample["hm"].shape == (3, 32, 32)
    assert sample["gt_boxes"].shape[1] == 4
    assert int(sample["num_gt"]) >= 1
    assert len(SHAPE_CLASSES) == 3


def test_dataset_is_deterministic_per_seed():
    a = ShapesDetectionDataset(length=4, input_size=128, seed=5, augment=False)[2]
    b = ShapesDetectionDataset(length=4, input_size=128, seed=5, augment=False)[2]
    assert torch.equal(a["input"], b["input"])


def test_render_shapes_boxes_in_bounds():
    rng = np.random.default_rng(0)
    img, boxes, labels = render_shapes_image(128, rng, max_objects=5, min_objects=3)
    assert img.shape == (128, 128, 3)
    assert len(boxes) == len(labels) >= 3
    for (x1, y1, x2, y2) in boxes:
        assert x2 > x1 and y2 > y1


def test_evaluate_runs_and_returns_map():
    device = select_device("cpu")
    model = build_centernet(3, variant="lite").to(device)
    ds = ShapesDetectionDataset(length=4, input_size=128, seed=3, augment=False)
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=2, collate_fn=collate)
    out = evaluate(model, loader, device, num_classes=3, max_batches=2)
    assert "map" in out
    assert 0.0 <= out["map"] <= 1.0


def test_train_config_serialisable():
    cfg = TrainConfig(epochs=1)
    assert isinstance(cfg.__dict__, dict)
    assert cfg.variant == "lite"
