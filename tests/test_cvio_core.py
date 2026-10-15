"""Numerical regression against the pinned upstream source, without network access."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from training.cvio_core.attention import SimAMDCFR
from training.cvio_core.config import Recipe, read_recipe
from training.cvio_core.losses import ASLLDAMLoss

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cvio_core_golden.json").read_text())
CONFIG = Path(__file__).parents[1] / "training/configs"


def require_training_runtime():
    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip("Install requirements-training.txt for CVIO integration tests")
    from training.cvio_core.runtime import configure_runtime

    configure_runtime()


def test_loss_value_and_gradient_match_pinned_source():
    golden = FIXTURE["loss"]
    logits = torch.tensor(golden["input"], requires_grad=True)
    loss = ASLLDAMLoss()(logits, torch.tensor(golden["targets"]))
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(golden["value"]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(logits.grad, torch.tensor(golden["gradient"]), rtol=1e-6, atol=1e-6)


def test_attention_output_and_all_gradients_match_pinned_source():
    golden = FIXTURE["attention"]
    module = SimAMDCFR(3)
    module.load_state_dict({key: torch.tensor(value) for key, value in golden["state"].items()})
    value = torch.tensor(golden["input"], requires_grad=True)
    result = module(value)
    result.square().mean().backward()
    torch.testing.assert_close(result, torch.tensor(golden["output"]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        value.grad, torch.tensor(golden["input_gradient"]), rtol=1e-6, atol=1e-6
    )
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(
            parameter.grad, torch.tensor(golden["parameter_gradients"][name]), rtol=1e-6, atol=1e-6
        )


@pytest.mark.parametrize(
    "field,value",
    [("epochs", 31), ("lr0", 0.0005), ("seed", 43), ("checkpoint_selection", "macro_f1")],
)
def test_recipe_rejects_unlabelled_variants(field, value):
    recipe = read_recipe(CONFIG / "cvio_yolo26m_proposed.yaml").model_dump()
    recipe[field] = value
    with pytest.raises(ValueError):
        Recipe.model_validate(recipe)


def test_recipe_rejects_wrong_train_counts():
    recipe = read_recipe(CONFIG / "cvio_yolo26m_proposed.yaml").model_dump()
    recipe["loss"]["class_counts"] = [403, 198, 328, 220]
    with pytest.raises(ValueError, match="frozen CVIO recipe"):
        Recipe.model_validate(recipe)


def test_loss_config_is_not_ignored_and_eval_uses_logits():
    require_training_runtime()
    from training.cvio_core.trainer import ClassificationLoss

    config = read_recipe(CONFIG / "cvio_yolo26m_proposed.yaml").loss
    logits = torch.tensor(FIXTURE["loss"]["input"], requires_grad=True)
    batch = {"cls": torch.tensor(FIXTURE["loss"]["targets"])}
    base, _ = ClassificationLoss(config)(logits, batch)
    evaluation, _ = ClassificationLoss(config)((logits.softmax(1), logits), batch)
    torch.testing.assert_close(base, evaluation)
    changed = deepcopy(config)
    changed.update(
        gamma_pos=1,
        gamma_neg=2,
        label_smoothing=0.2,
        ldam_max_m=0.3,
        ldam_scale=5,
        class_counts=[4, 3, 2, 1],
    )
    actual, _ = ClassificationLoss(changed)(logits, batch)
    expected = ASLLDAMLoss([4, 3, 2, 1], 1, 2, 0.2, 0.3, 5)(logits, batch["cls"])
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, base)


def test_cpu_autocast_finite_backward():
    require_training_runtime()
    from training.cvio_core.trainer import ClassificationLoss

    layer = torch.nn.Linear(8, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        predictions = layer(torch.ones(4, 8))
        loss, _ = ClassificationLoss(read_recipe(CONFIG / "cvio_yolo26m_proposed.yaml").loss)(
            predictions, {"cls": torch.arange(4)}
        )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(layer.weight.grad).all()


@pytest.mark.parametrize("recipe_name,wrappers", [("ce", 0), ("proposed", 1)])
def test_model_injection_pretrained_transfer_reload_and_resume(tmp_path, recipe_name, wrappers):
    require_training_runtime()
    from ultralytics.nn.tasks import ClassificationModel, load_checkpoint

    from training.cvio_core.trainer import audit_model, build_model

    torch.set_num_threads(2)
    recipe = read_recipe(CONFIG / f"cvio_yolo26m_{recipe_name}.yaml").model_dump()
    # Small architecture for offline structural regression; the separate smoke uses actual m weights.
    source = ClassificationModel("yolo26n-cls.yaml", nc=1000, verbose=False)
    model = build_model(source.yaml, recipe, source)
    audit = audit_model(model)
    assert audit["attention_wrappers"] == wrappers
    assert audit["pretrained_transfer"]["exact_match"]
    if wrappers:
        assert audit["attention"]["params_added"] > 0
    model.train()
    logits = model(torch.rand(2, 3, 224, 224))
    loss, _ = model.init_criterion()(logits, {"cls": torch.tensor([0, 3])})
    if not wrappers:
        torch.testing.assert_close(
            loss, torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
        )
    loss.backward()
    assert torch.isfinite(loss)
    checkpoint = tmp_path / "own_trusted_checkpoint.pt"
    torch.save({"model": model}, checkpoint)
    reloaded, _ = load_checkpoint(checkpoint, device="cpu", fuse=False)
    resumed = build_model(source.yaml, recipe, reloaded, resume=True)
    assert audit_model(resumed)["attention_wrappers"] == wrappers
    for key, value in reloaded.state_dict().items():
        if key.startswith("criterion."):
            continue
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)
    reloaded.eval()
    resumed.eval()
    image = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(image)[0], resumed(image)[0], rtol=0, atol=0)
