import json

import numpy as np
import pytest
import torch
import yaml

from app.core.constants import CLASSES
from scripts.make_toy_dataset import make_dataset
from training.artifacts import load_artifact, sha256
from training.dataset import validate_manifest
from training.evaluate import evaluate
from training.export import export_onnx
from training.losses import make_loss
from training.metrics import classification_metrics
from training.models import create_model
from training.train import train


def test_hand_computed_metrics():
    result = classification_metrics([0, 1, 2, 3], [0, 1, 2, 2])
    assert result["accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["per_class"]["WSSV"]["precision"] == 0.5
    assert result["per_class"]["WSSV_BG"]["recall"] == 0
    assert result["confusion_matrix"] == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]


@pytest.mark.parametrize(
    "defect", ["group", "duplicate", "hash", "id", "path", "class", "missing_class"]
)
def test_manifest_leakage_and_integrity(tmp_path, defect):
    path = make_dataset(tmp_path / "data")
    manifest = json.loads(path.read_text())
    train_sample = manifest["samples"][0]
    val_sample = next(s for s in manifest["samples"] if s["split"] == "val")
    if defect == "group":
        val_sample["group_id"] = train_sample["group_id"]
    elif defect == "duplicate":
        (path.parent / val_sample["path"]).write_bytes(
            (path.parent / train_sample["path"]).read_bytes()
        )
        val_sample["sha256"] = train_sample["sha256"]
    elif defect == "hash":
        val_sample["sha256"] = "0" * 64
    elif defect == "id":
        val_sample["id"] = train_sample["id"]
    elif defect == "path":
        val_sample["path"] = "../escape.png"
    elif defect == "class":
        val_sample["label"] = "Unknown"
    else:
        manifest["samples"] = [s for s in manifest["samples"] if s["label"] != "BG"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_manifest(path)


@pytest.mark.parametrize("name", ["weighted_ce", "ldam"])
def test_losses_finite_and_differentiable(name):
    logits = torch.randn(4, 4, requires_grad=True)
    loss = make_loss(name, [8, 4, 2, 1])(logits, torch.arange(4))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "backbone", ["tiny", "yolo_csp_cls", "efficientnet_b0", "mobilenet_v3_large"]
)
def test_all_backbones_forward_and_backward(backbone):
    torch.set_num_threads(2)
    model = create_model(backbone, simam=True)
    output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 4)
    output.sum().backward()
    assert torch.isfinite(model.head.weight.grad).all()


def test_train_evaluate_export_provenance_and_reproducibility(tmp_path):
    manifest = make_dataset(tmp_path / "data")
    config = {
        "model_version": "integration-v1",
        "backbone": "tiny",
        "manifest": str(manifest),
        "output_root": str(tmp_path / "models"),
        "epochs": 2,
        "batch_size": 8,
        "seed": 123,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    directory = train(config_path)
    report = evaluate(directory / "weights.pt", reports_dir=tmp_path / "reports")
    saved = json.loads((tmp_path / "reports" / "eval_integration-v1.json").read_text())
    assert saved == report == json.loads((directory / "eval.json").read_text())
    matrix = np.array(report["confusion_matrix"])
    assert report["accuracy"] == matrix.trace() / matrix.sum()
    for i, label in enumerate(CLASSES):
        assert report["per_class"][label]["recall"] == matrix[i, i] / matrix[i].sum()
    assert report["macro_f1"] == pytest.approx(
        np.mean([c["f1"] for c in report["per_class"].values()])
    )
    assert report["checkpoint_sha256"] == sha256(directory / "weights.pt")
    assert report["manifest_sha256"] == sha256(manifest)
    assert report["seed"] == 123 and report["synthetic"] is True and report["publishable"] is False
    assert export_onnx(directory).exists()
    first, _, _, _ = load_artifact(directory)
    config["model_version"] = "integration-v2"
    config_path.write_text(yaml.safe_dump(config))
    second_directory = train(config_path)
    second, _, _, _ = load_artifact(second_directory)
    for name, parameter in first.state_dict().items():
        assert torch.equal(parameter, second.state_dict()[name]), name
    with pytest.raises(FileExistsError):
        train(config_path)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest differs"):
        evaluate(directory, reports_dir=tmp_path / "reports")
