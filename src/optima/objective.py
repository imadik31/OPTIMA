"""Likelihoods and the augmented variational objective.

Notation (per training example ``i`` and augmentation sample ``k = 1..K``):

    l_ik = log p(y_i | T_{gamma_ik}(x_i), theta),   gamma_ik ~ p(gamma | phi),  phi ~ q(phi)

Data terms supported by :class:`OptimaObjective` (``bound=``):

``"elbo"``          ``(1/K) sum_k l_ik``
    Monte Carlo estimate of the augmented ELBO of the paper (Eq. 7), which
    uses ``E_gamma log p``. With ``K = 1`` this is Algorithm 1/2.

``"multi_sample"``  ``log (1/K) sum_k exp(l_ik)``
    A tighter bound on the marginalised likelihood ``log E_gamma p`` (Eq. 5).
    It is non-decreasing in ``K``, equals ``"elbo"`` at ``K = 1`` and converges to
    ``log E_gamma p`` as ``K -> infinity`` (the argument of Burda et al., 2016).

``"summed"``        ``sum_k l_ik``
    Naive replication: counts every augmented copy as a separate observation.
    This raises the likelihood to the power ``K`` (a tempered posterior). Provided
    only as a baseline.

The loss is the negative objective *per training example*:

    loss = -[ mean_i data_i - (beta_net KL(q(theta)||p(theta)) + beta_aug KL(q(phi)||p(phi))) / N ]

With ``beta_net = beta_aug = 1`` this is exactly ``-ELBO / N``. Note that the
KL terms are divided by the training-set size ``N``: adding an unscaled KL to a
mean-reduced loss makes the prior ``N`` times stronger than the ELBO implies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

Bound = Literal["elbo", "multi_sample", "summed"]


# --------------------------------------------------------------------------- likelihoods
def categorical_log_likelihood(logits: Tensor, target: Tensor) -> Tensor:
    """Per-example ``log p(y | logits)`` for integer labels or soft labels.

    For soft labels ``t`` (e.g. Mixup) this returns ``sum_c t_c log p_c``.
    """
    log_p = logits.log_softmax(dim=-1)
    if target.dtype in (torch.int32, torch.int64):
        return log_p.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (target * log_p).sum(dim=-1)


def gaussian_log_likelihood(pred: Tensor, target: Tensor, noise_std: float | Tensor) -> Tensor:
    """Per-example Gaussian ``log N(target | pred, noise_std^2)``, summed over output dims."""
    noise_std = torch.as_tensor(noise_std, dtype=pred.dtype, device=pred.device)
    log_normaliser = torch.log(noise_std) + 0.5 * math.log(2 * math.pi)
    log_p = -0.5 * ((target - pred) / noise_std) ** 2 - log_normaliser
    return log_p.flatten(1).sum(dim=1) if log_p.dim() > 1 else log_p


# --------------------------------------------------------------------------- objective
@dataclass
class ObjectiveOutput:
    """Loss to minimise and detached diagnostics (all per training example)."""

    loss: Tensor
    objective: Tensor
    data_term: Tensor
    kl_net: Tensor
    kl_aug: Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss.item(),
            "objective": self.objective.item(),
            "data_term": self.data_term.item(),
            "kl_net": self.kl_net.item(),
            "kl_aug": self.kl_aug.item(),
        }


class OptimaObjective:
    """Negative augmented variational objective, per training example.

    Args:
        num_train: Training-set size ``N`` (used to scale the KL terms).
        bound: Data term, see the module docstring.
        beta_net: Weight on ``KL(q(theta) || p(theta))``. 1 gives the ELBO.
        beta_aug: Weight on ``KL(q(phi) || p(phi))``. 1 gives the ELBO.
    """

    def __init__(
        self,
        num_train: int,
        bound: Bound = "elbo",
        beta_net: float = 1.0,
        beta_aug: float = 1.0,
    ) -> None:
        if bound not in ("elbo", "multi_sample", "summed"):
            raise ValueError(f"Unknown bound: {bound!r}")
        if num_train <= 0:
            raise ValueError("num_train must be positive")
        self.num_train = num_train
        self.bound = bound
        self.beta_net = beta_net
        self.beta_aug = beta_aug

    def data_term(self, log_lik: Tensor) -> Tensor:
        """Per-example data term from ``log_lik`` of shape ``[K, B]``; returns ``[B]``."""
        if self.bound == "elbo":
            return log_lik.mean(dim=0)
        if self.bound == "multi_sample":
            return torch.logsumexp(log_lik, dim=0) - math.log(log_lik.shape[0])
        return log_lik.sum(dim=0)

    def __call__(
        self,
        log_lik: Tensor,
        kl_net: Tensor,
        kl_aug: Tensor,
        log_prob: Tensor | None = None,
    ) -> ObjectiveOutput:
        """Compute the loss.

        Args:
            log_lik: ``l_ik``, shape ``[K, B]``.
            kl_net: ``KL(q(theta) || p(theta))`` (zero for a deterministic network).
            kl_aug: ``KL(q(phi) || p(phi))``.
            log_prob: ``log p(gamma_ik | phi)``, shape ``[K, B]``, for augmenters that are
                not reparameterisable. Adds a score-function gradient for ``phi``.
        """
        if log_lik.dim() != 2:
            raise ValueError("log_lik must have shape [K, B]")
        device = log_lik.device
        kl_net, kl_aug = kl_net.to(device), kl_aug.to(device)

        data = self.data_term(log_lik)
        objective = data.mean() - (self.beta_net * kl_net + self.beta_aug * kl_aug) / self.num_train
        loss = -objective
        if log_prob is not None:
            surrogate = self._score_function_surrogate(log_lik, data, log_prob)
            loss = loss + (surrogate - surrogate.detach())  # gradient only, value unchanged

        return ObjectiveOutput(
            loss=loss,
            objective=objective.detach(),
            data_term=data.mean().detach(),
            kl_net=kl_net.detach(),
            kl_aug=kl_aug.detach(),
        )

    def _score_function_surrogate(self, log_lik: Tensor, data: Tensor, log_prob: Tensor) -> Tensor:
        """Surrogate whose gradient is the (negated) REINFORCE estimate for ``phi``.

        Uses a leave-one-out batch mean as baseline, which keeps the estimator unbiased.
        """
        if log_prob.shape != log_lik.shape:
            raise ValueError("log_prob must have the same shape [K, B] as log_lik")
        if self.bound == "multi_sample":
            # f_i depends jointly on the K samples: weight the summed score by f_i.
            signal, score = data.detach(), log_prob.sum(dim=0)  # [B], [B]
            return -((signal - _leave_one_out_mean(signal)) * score).mean()
        scale = log_lik.shape[0] if self.bound == "summed" else 1
        signal = log_lik.detach()  # [K, B]
        return -scale * ((signal - _leave_one_out_mean(signal)) * log_prob).mean()


def _leave_one_out_mean(values: Tensor) -> Tensor:
    """Mean over the last dimension excluding each element itself (0 if size 1)."""
    n = values.shape[-1]
    if n == 1:
        return torch.zeros_like(values)
    return (values.sum(dim=-1, keepdim=True) - values) / (n - 1)
