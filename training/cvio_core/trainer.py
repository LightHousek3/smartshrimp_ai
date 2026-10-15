# SPDX-License-Identifier: AGPL-3.0-only
# Adapted from CVIO attention/patch_yolo.py at fd64876; see NOTICE.
# ruff: noqa: E402
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from training.cvio_core.runtime import configure_runtime

configure_runtime()

from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.nn.modules import Classify
from ultralytics.nn.tasks import ClassificationModel

from app.core.constants import CLASSES
from training.cvio_core.attention import SimAMDCFR
from training.cvio_core.losses import ASLLDAMLoss
from training.io import write_json


def extract_logits(predictions):
    # Version-locked Classify: training tensor; eval (probabilities, logits).
    if isinstance(predictions, tuple) and len(predictions) == 2:
        predictions = predictions[1]
    if not torch.is_tensor(predictions) or predictions.ndim != 2 or predictions.shape[1] != 4:
        raise ValueError("Expected four-class logits tensor or (probabilities, logits)")
    return predictions


class ClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.loss_fcn = ASLLDAMLoss(
            class_counts=config["class_counts"],
            gamma_pos=config["gamma_pos"],
            gamma_neg=config["gamma_neg"],
            label_smoothing=config["label_smoothing"],
            max_m=config["ldam_max_m"],
            scale=config["ldam_scale"],
        )

    def forward(self, predictions, batch):
        logits = extract_logits(predictions).float()
        targets = batch["cls"].long().view(-1).to(logits.device)
        self.loss_fcn.to(logits.device)
        loss = self.loss_fcn(logits, targets)
        return loss, loss.detach()


class CVIOClassificationModel(ClassificationModel):
    def init_criterion(self):
        if self.cvio_recipe["method"] == "baseline_ce":
            return super().init_criterion()
        return ClassificationLoss(self.cvio_recipe["loss"])


class AttentionBeforeClassify(nn.Module):
    def __init__(self, head, attention):
        super().__init__()
        self.attention, self.head = attention, head
        for attribute in ("f", "i", "type"):
            setattr(self, attribute, getattr(head, attribute))
        self.np = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, value):
        if isinstance(value, (list, tuple)):
            value = torch.cat(value, dim=1)
        return self.head(self.attention(value))


def inject_attention(model, epsilon):
    index, head = next(
        (i, m) for i, m in reversed(list(enumerate(model.model))) if isinstance(m, Classify)
    )
    captured = {}

    def capture(_module, inputs):
        captured["shape"] = list(inputs[0].shape)

    handle = head.register_forward_pre_hook(capture)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 3, 224, 224, device=next(model.parameters()).device))
    finally:
        handle.remove()
        model.train(was_training)
    channels = captured["shape"][1]
    attention = SimAMDCFR(channels, e_lambda=epsilon)
    model.model[index] = AttentionBeforeClassify(head, attention)
    model.attention_module_audit = dict(
        inserted=True,
        target_index=index,
        inferred_channels=channels,
        head_input_shape=captured["shape"],
        e_lambda=epsilon,
        params_added=sum(p.numel() for p in attention.parameters()),
    )


def build_model(cfg, recipe, weights=None, resume=False, verbose=False):
    if resume:
        if not isinstance(weights, CVIOClassificationModel) or weights.cvio_recipe != recipe:
            raise ValueError("Resume checkpoint recipe/architecture mismatch")
        model = deepcopy(weights).float()
        # Preserve the entire wrapped head and ordering of optimizer parameters.
        model.criterion = model.init_criterion()
    else:
        model = CVIOClassificationModel(cfg, nc=4, ch=3, verbose=verbose)
        model.cvio_recipe = deepcopy(recipe)
        if weights is not None:
            model.load(weights)
            before = weights.float().state_dict()
            after = model.state_dict()
            matched = [
                key
                for key, value in before.items()
                if key in after and value.shape == after[key].shape
            ]
            if not matched or not all(torch.equal(before[key], after[key]) for key in matched):
                raise ValueError("Pretrained transfer did not preserve matching tensors")
            model.pretrained_transfer_audit = dict(matched_tensors=len(matched), exact_match=True)
        if recipe["method"] != "baseline_ce":
            inject_attention(model, recipe["attention"]["e_lambda"])
    model.names = dict(enumerate(CLASSES))
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def audit_model(model):
    wrappers = [m for m in model.modules() if isinstance(m, AttentionBeforeClassify)]
    expected = model.cvio_recipe["method"] != "baseline_ce"
    if len(wrappers) != int(expected) or list(model.names.values()) != list(CLASSES):
        raise ValueError("Checkpoint attention or class order mismatch")
    if expected and not isinstance(model.init_criterion(), ClassificationLoss):
        raise ValueError("ASL-LDAM criterion missing")
    return dict(
        method=model.cvio_recipe["method"],
        class_order=list(CLASSES),
        attention_wrappers=len(wrappers),
        parameter_count=sum(p.numel() for p in model.parameters()),
        attention=getattr(model, "attention_module_audit", None),
        pretrained_transfer=getattr(model, "pretrained_transfer_audit", None),
    )


class CVIOTrainer(ClassificationTrainer):
    def __init__(self, recipe, overrides, **kwargs):
        self.recipe = recipe
        super().__init__(overrides=overrides, **kwargs)
        if self.world_size > 1:
            raise ValueError("Custom DDP is not qualified: use exactly one GPU or CPU")
        expected_folders = [f"{i:02d}_{label}" for i, label in enumerate(CLASSES)]
        if list(self.data["names"].values()) != expected_folders:
            raise ValueError("Prepared dataset class folders/order mismatch")
        self.data["names"] = dict(enumerate(CLASSES))

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            # Paths can change when restoring a Kaggle session; the runner has
            # already checked the immutable recipe/data/checkpoint identity.
            for key in ("data", "project", "name", "exist_ok"):
                setattr(self.args, key, overrides[key])
            self.args.save_dir = str(Path(overrides["project"]) / overrides["name"])

    def _load_checkpoint_state(self, ckpt):
        super()._load_checkpoint_state(ckpt)
        if self.resume:
            saved = ckpt["ema"].float().state_dict()
            current = self.ema.ema.state_dict()
            if not all(torch.equal(value, current[key]) for key, value in saved.items()):
                raise ValueError("Resumed EMA tensors differ from checkpoint")
            optimizer = self.optimizer.state_dict()
            saved_optimizer = ckpt["optimizer"]
            if optimizer["param_groups"] != saved_optimizer["param_groups"]:
                raise ValueError("Resumed optimizer parameter groups differ")
            for key, state in saved_optimizer["state"].items():
                for field, value in state.items():
                    actual = optimizer["state"][key][field]
                    if torch.is_tensor(value):
                        if not torch.equal(value.float().cpu(), actual.float().cpu()):
                            raise ValueError("Resumed optimizer tensors differ")
                    elif value != actual:
                        raise ValueError("Resumed optimizer state differs")
            write_json(
                Path(self.save_dir) / "resume_audit.json",
                dict(
                    checkpoint_epoch=ckpt["epoch"],
                    ema_tensors_exact=True,
                    ema_tensor_count=len(saved),
                    optimizer_state_exact=True,
                    optimizer_parameter_states=len(optimizer["state"]),
                    model=audit_model(self.model),
                    bitwise_continuation_claimed=False,
                ),
            )

    def get_model(self, cfg=None, weights=None, verbose=True):
        return build_model(cfg, self.recipe, weights, resume=bool(self.resume), verbose=verbose)

    def final_eval(self):
        # Keep optimizer/scaler in best/last for audit and interrupted-run resume.
        # Explicit runner evaluation uses only validation here; test is a separate command.
        audit_model(self.model)
