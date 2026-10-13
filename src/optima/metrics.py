"""Evaluation: Monte Carlo predictive, accuracy, NLL, ECE and OOD AUROC."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


@torch.no_grad()
def mc_predict(model: nn.Module, x: Tensor, num_samples: int = 10) -> Tensor:
    """Predictive class probabilities averaged over ``num_samples`` forward passes.

    For a deterministic model, use ``num_samples=1``.
    """
    probs = torch.stack([model(x).softmax(dim=-1) for _ in range(num_samples)])
    return probs.mean(dim=0)


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    num_samples: int = 10,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Predictive probabilities and labels for a whole data loader (on CPU)."""
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        all_probs.append(mc_predict(model, x.to(device), num_samples).cpu())
        all_labels.append(y.cpu())
    return torch.cat(all_probs), torch.cat(all_labels)


def accuracy(probs: Tensor, labels: Tensor) -> float:
    return (probs.argmax(dim=-1) == labels).float().mean().item()


def negative_log_likelihood(probs: Tensor, labels: Tensor, eps: float = 1e-12) -> float:
    return -torch.log(probs.gather(-1, labels.unsqueeze(-1)).clamp_min(eps)).mean().item()


def expected_calibration_error(probs: Tensor, labels: Tensor, num_bins: int = 10) -> float:
    """ECE with equal-width confidence bins: ``sum_b |B_b|/n * |acc(B_b) - conf(B_b)|``."""
    confidence, prediction = probs.max(dim=-1)
    correct = (prediction == labels).float()
    edges = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if lower == 0:
            in_bin |= confidence == 0
        if in_bin.any():
            weight = in_bin.float().mean()
            ece += weight * (correct[in_bin].mean() - confidence[in_bin].mean()).abs()
    return ece.item()


def predictive_entropy(probs: Tensor, eps: float = 1e-12) -> Tensor:
    """Per-example entropy ``-sum_c p_c log p_c``."""
    return -(probs * probs.clamp_min(eps).log()).sum(dim=-1)


def auroc(scores_in: Tensor, scores_out: Tensor) -> float:
    """Area under the ROC curve for detecting OOD examples by a score
    (higher score = more likely OOD), via the Mann-Whitney U statistic with ties."""
    scores = torch.cat([scores_in, scores_out]).double()
    is_out = torch.cat([torch.zeros(len(scores_in)), torch.ones(len(scores_out))]).bool()
    sorted_scores, order = scores.sort()
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    ends = counts.cumsum(dim=0).double()
    average_rank = ends - (counts.double() - 1) / 2  # 1-based average rank of each tie group
    ranks = torch.empty_like(scores)
    ranks[order] = average_rank[inverse]
    n_out, n_in = len(scores_out), len(scores_in)
    u_statistic = ranks[is_out].sum() - n_out * (n_out + 1) / 2
    return (u_statistic / (n_in * n_out)).item()
