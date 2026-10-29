import numpy as np
import pytest

from classification.constants import CHECKPOINT_BYTES, CHECKPOINT_SHA256
from classification.model import Classifier, aggregate, sha256, verify_checkpoint
from classification.settings import ROOT, Settings


def test_only_supplied_checkpoint_is_accepted(tmp_path):
    path = tmp_path / "best.pt"
    with pytest.raises(ValueError, match="Missing checkpoint"):
        verify_checkpoint(path)
    path.write_bytes(b"x" * CHECKPOINT_BYTES)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checkpoint(path)


def test_historical_candidate_is_explicitly_enabled(tmp_path):
    engine = Classifier(Settings(_env_file=None, api_key="x" * 64, model_dir=tmp_path))
    with pytest.raises(ValueError, match="explicit opt-in"):
        engine.load()
    assert not engine.ready and engine.model is None


def test_mean_probabilities_without_second_softmax():
    actual = {
        p["label"]: p["probability"]
        for p in aggregate([[0.8, 0.1, 0.05, 0.05], [0.2, 0.3, 0.4, 0.1]])
    }
    assert actual == pytest.approx({"Healthy": 0.5, "BG": 0.2, "WSSV": 0.225, "WSSV_BG": 0.075})


@pytest.mark.parametrize(
    "values",
    [
        [[np.nan, 0, 0, 1]],
        [[np.inf, 0, 0, 0]],
        [[-0.1, 0.1, 0.5, 0.5]],
        [[0.5] * 4],
        [[0, 0, 0]],
        [0, 0, 0, 1],
        [],
        [[0.25] * 4] * 6,
    ],
)
def test_invalid_outputs_fail_closed(values):
    with pytest.raises(ValueError):
        aggregate(values)


@pytest.mark.integration
def test_real_checkpoint_identity():
    path = ROOT / "classification/models/cvio-classification.pt"
    if not path.exists():
        pytest.skip("Local checkpoint required")
    verify_checkpoint(path)
    assert sha256(path) == CHECKPOINT_SHA256
