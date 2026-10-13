"""Common interface for learnable augmentations ``T_gamma`` with ``gamma ~ p(gamma | phi)``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from optima.hyperparameter import HyperParameter


@dataclass
class AugmentedBatch:
    """Output of an augmenter.

    Attributes:
        x: Transformed inputs ``T_gamma(x)``.
        y: Targets. Integer labels, or soft labels for label-mixing augmentations.
        log_prob: Per-example ``log p(gamma_i | phi)``, shape ``[B]``. Only set by
            augmenters whose sample is not reparameterisable (e.g. discrete masks);
            the objective then adds a score-function (REINFORCE) gradient for ``phi``.
    """

    x: Tensor
    y: Tensor
    log_prob: Tensor | None = None


def gamma_prior_kl(mean: Tensor, std: Tensor, prior_std: float | Tensor) -> Tensor:
    """``KL(N(mean, std^2) || N(0, prior_std^2))``, summed over elements.

    The penalty used in the paper's experiments (Sec. 5.1, App. F.2). It puts a prior on
    the transformation ``gamma`` itself and treats the augmentation distribution
    ``p(gamma | phi)`` as its variational posterior; it is minimised at ``mean = 0,
    std = prior_std``. This differs from ``KL(q(phi) || p(phi))`` in the augmented ELBO
    (Eq. 7), which :meth:`Augmenter.kl` returns. See the README, "Two objectives".
    """
    prior_std = torch.as_tensor(prior_std, dtype=std.dtype, device=std.device)
    kl = torch.log(prior_std / std) + (std**2 + mean**2) / (2 * prior_std**2) - 0.5
    return kl.sum()


class Augmenter(nn.Module, ABC):
    """Base class. Subclasses hold their hyperparameters as :class:`HyperParameter`
    attributes and implement :meth:`forward`.

    ``phi`` is sampled once per call (one draw from ``q(phi)`` per minibatch), and
    ``gamma_i`` is sampled independently for every example, as in Algorithm 1.
    """

    #: ``False`` for augmenters that need score-function gradients.
    reparameterized: bool = True

    @abstractmethod
    def forward(self, x: Tensor, y: Tensor) -> AugmentedBatch:
        """Apply a random transformation to every example of the batch."""

    def kl(self) -> Tensor:
        """Sum of the penalties of all hyperparameters (``KL(q(phi) || p(phi))``)."""
        terms = [m.kl() for m in self.modules() if isinstance(m, HyperParameter)]
        if not terms:
            return torch.zeros(())
        return torch.stack(terms).sum()

    def hyperparameters(self) -> dict[str, Tensor]:
        """Current point values ``g(m)`` of every hyperparameter, for logging."""
        return {
            name: module.value
            for name, module in self.named_modules()
            if isinstance(module, HyperParameter)
        }


class NoAugmentation(Augmenter):
    """Identity transformation (the "No Aug" baseline)."""

    def forward(self, x: Tensor, y: Tensor) -> AugmentedBatch:
        return AugmentedBatch(x=x, y=y)
