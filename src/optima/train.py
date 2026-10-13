"""One training step and one epoch of OPTIMA (Algorithms 1 and 2 of the paper)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn

from optima.augmenters.base import Augmenter
from optima.bayesian import kl_divergence
from optima.objective import ObjectiveOutput, OptimaObjective

LogLikelihood = Callable[[Tensor, Tensor], Tensor]


def optima_step(
    model: nn.Module,
    augmenter: Augmenter,
    objective: OptimaObjective,
    x: Tensor,
    y: Tensor,
    log_likelihood: LogLikelihood,
    num_samples: int = 1,
) -> ObjectiveOutput:
    """Compute the OPTIMA loss on one minibatch (no optimiser step).

    For each of ``num_samples`` draws, ``phi ~ q(phi)`` is sampled once and
    ``gamma_i ~ p(gamma | phi)`` once per example; the model is evaluated on the
    transformed batch. ``model`` may be Bayesian (Algorithm 1, uses
    :class:`~optima.bayesian.BayesianLinear` layers) or deterministic (Algorithm 2).

    Args:
        log_likelihood: Maps ``(model_output, target)`` to per-example log-likelihoods ``[B]``.
        num_samples: Number of augmentation samples ``K`` per example.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    log_liks, log_probs = [], []
    for _ in range(num_samples):
        batch = augmenter(x, y)
        log_liks.append(log_likelihood(model(batch.x), batch.y))
        if batch.log_prob is not None:
            log_probs.append(batch.log_prob)
    log_prob = torch.stack(log_probs) if log_probs else None
    return objective(torch.stack(log_liks), kl_divergence(model), augmenter.kl(), log_prob)


def train_epoch(
    model: nn.Module,
    augmenter: Augmenter,
    objective: OptimaObjective,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    log_likelihood: LogLikelihood,
    num_samples: int = 1,
    device: torch.device | str = "cpu",
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """Run one epoch and return the average diagnostics.

    The optimiser should contain both the model parameters and the augmenter
    parameters (typically as two parameter groups with different learning rates).
    """
    model.train()
    augmenter.train()
    totals: dict[str, float] = {}
    num_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = optima_step(model, augmenter, objective, x, y, log_likelihood, num_samples)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        if max_grad_norm is not None:
            params = [p for group in optimizer.param_groups for p in group["params"]]
            nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()

        for key, value in out.as_dict().items():
            totals[key] = totals.get(key, 0.0) + value
        num_batches += 1
    return {key: value / max(num_batches, 1) for key, value in totals.items()}
