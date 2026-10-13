"""Shared pieces of the CIFAR-10 examples: data, ResNet-18 backbone, evaluation, logging.

Requires ``torchvision``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from optima import BayesianLinear
from optima.metrics import (
    accuracy,
    auroc,
    expected_calibration_error,
    negative_log_likelihood,
    predict_loader,
    predictive_entropy,
)

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def make_loaders(root: str, batch_size: int, crop_flip: bool, normalize: bool,
                 workers: int = 4) -> tuple[DataLoader, DataLoader, DataLoader]:
    """CIFAR-10 train/test loaders and an SVHN test loader (for OOD detection).

    ``crop_flip`` adds the standard ``RandomCrop(32, 4)`` + ``RandomHorizontalFlip`` to every
    method, including "no augmentation"; the learned augmentation acts on top of it.
    """
    tail = [transforms.ToTensor()]
    if normalize:
        tail.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))
    head = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    train_tf = transforms.Compose((head if crop_flip else []) + tail)
    test_tf = transforms.Compose(tail)
    train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    svhn = datasets.SVHN(root, split="test", download=True, transform=test_tf)
    kw = dict(num_workers=workers, pin_memory=True)
    return (DataLoader(train, batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(test, 256, **kw), DataLoader(svhn, 256, **kw))


def make_resnet18(pretrained: bool, num_classes: int = 10, prior_std: float = 1.0,
                  init_std: float = math.exp(-5)) -> nn.Module:
    """ResNet-18 with a CIFAR stem (3x3 conv, no max-pool) and a Bayesian last layer.

    With ``pretrained=True`` the ImageNet weights are loaded first; the new stem is
    re-initialised either way.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = BayesianLinear(model.fc.in_features, num_classes, prior_std=prior_std,
                              init_std=init_std)
    return model


def reliability_bins(probs: Tensor, labels: Tensor, num_bins: int = 10) -> dict[str, list]:
    """Per-bin mean confidence, accuracy and count (for reliability diagrams)."""
    confidence, prediction = probs.max(dim=-1)
    correct = (prediction == labels).float()
    edges = torch.linspace(0, 1, num_bins + 1)
    out: dict[str, list] = {"confidence": [], "accuracy": [], "count": []}
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        n = int(in_bin.sum())
        out["count"].append(n)
        out["confidence"].append(confidence[in_bin].mean().item() if n else None)
        out["accuracy"].append(correct[in_bin].mean().item() if n else None)
    return out


def evaluate(model: nn.Module, test_loader: DataLoader, ood_loader: DataLoader | None,
             mc_samples: int, device: str) -> dict:
    """Accuracy, NLL, ECE (and SVHN AUROC by predictive entropy) of the MC predictive."""
    probs, labels = predict_loader(model, test_loader, mc_samples, device)
    metrics = {
        "acc": accuracy(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
        "ece": expected_calibration_error(probs, labels),
        "reliability": reliability_bins(probs, labels),
    }
    if ood_loader is not None:
        ood_probs, _ = predict_loader(model, ood_loader, mc_samples, device)
        metrics["ood_auroc"] = auroc(predictive_entropy(probs), predictive_entropy(ood_probs))
    return metrics


def save_json(path: str | None, payload: dict) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))
        print(f"saved {path}")


def summary(metrics: dict) -> str:
    keys = ("acc", "nll", "ece", "ood_auroc")
    return "  ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics)
