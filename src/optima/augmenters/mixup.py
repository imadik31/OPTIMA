"""Mixup with a learnable Beta concentration ``alpha``."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta

from optima.augmenters.base import AugmentedBatch, Augmenter
from optima.hyperparameter import HyperParameter, Mode


class LearnableMixup(Augmenter):
    """Mixup (Zhang et al., 2018) with ``lambda_i ~ Beta(alpha, alpha)`` and learnable ``alpha``.

    Each example is mixed with a random partner from the same batch, with its own
    ``lambda_i``. Targets become soft labels, so the log-likelihood of the mixed
    example is ``lambda log p(y_a) + (1 - lambda) log p(y_b)``.
    ``alpha`` gets pathwise gradients through ``Beta.rsample`` (implicit
    reparameterisation).

    Args:
        num_classes: Number of classes (to build soft labels).
        init_alpha: Initial ``alpha``.
        prior_std: Prior std of ``log alpha``; the prior is centred at ``init_alpha``.
        mode: See :class:`HyperParameter`.
        eps: ``lambda`` is clamped to ``[eps, 1 - eps]`` for numerical stability.

    Note:
        As ``alpha -> 0``, ``lambda`` concentrates on {0, 1}, i.e. no mixing.
    """

    def __init__(
        self,
        num_classes: int,
        init_alpha: float = 0.2,
        prior_std: float = 2.0,
        mode: Mode = "variational",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = HyperParameter(init_alpha, transform="exp", prior_std=prior_std, mode=mode)
        self.eps = eps

    def forward(self, x: Tensor, y: Tensor) -> AugmentedBatch:
        batch_size = x.shape[0]
        alpha = self.alpha.sample()
        lam = Beta(alpha, alpha).rsample((batch_size,)).clamp(self.eps, 1 - self.eps)
        perm = torch.randperm(batch_size, device=x.device)

        lam_x = lam.view(-1, *([1] * (x.dim() - 1))).to(x.dtype)
        x_mix = lam_x * x + (1 - lam_x) * x[perm]

        y_soft = F.one_hot(y, self.num_classes).to(lam.dtype) if y.dim() == 1 else y
        lam_y = lam.unsqueeze(1)
        y_mix = lam_y * y_soft + (1 - lam_y) * y_soft[perm]
        return AugmentedBatch(x=x_mix, y=y_mix)
