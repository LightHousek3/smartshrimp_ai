import torch

from app.core.constants import CLASSES
from training.artifacts import load_artifact


def test_v1_loader_preserves_golden_probabilities(artifact):
    model, config, metadata, directory = load_artifact(artifact)
    assert directory == artifact and config["backbone"] == "tiny"
    assert metadata["classes"] == list(CLASSES)
    with torch.inference_mode():
        probs = model(torch.zeros(1, 3, 224, 224)).softmax(1)
    torch.testing.assert_close(probs, torch.tensor([[0.05, 0.15, 0.7, 0.1]]))
