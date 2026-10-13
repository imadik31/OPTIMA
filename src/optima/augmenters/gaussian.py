"""Additive Gaussian perturbation ``T_gamma(x) = x + gamma``, ``gamma ~ N(mu, sigma^2)``."""

from __future__ import annotations

from torch import Tensor

from optima.augmenters.base import AugmentedBatch, Augmenter, gamma_prior_kl
from optima.hyperparameter import HyperParameter, Mode


class GaussianShift(Augmenter):
    """Additive Gaussian noise with learnable mean and standard deviation.

    Used for the synthetic regression experiment (paper, Sec. 5.1) and the
    Gaussian-translation ablation (App. F.2). The noise is drawn independently
    for every input element.

    Args:
        init_std: Initial ``sigma``.
        init_mean: Initial ``mu``.
        std_prior_std: Prior std of ``log sigma``; the prior is centred at ``init_std``.
        mean_prior_std: Prior std of ``mu``; the prior is centred at 0.
        learn_mean: If ``False``, ``mu`` is fixed at ``init_mean``.
        mode: Treatment of ``sigma`` (and ``mu`` if learned); see :class:`HyperParameter`.
            Use ``mode="fixed"`` for the fixed-augmentation baseline.
    """

    def __init__(
        self,
        init_std: float = 0.1,
        init_mean: float = 0.0,
        std_prior_std: float = 1.0,
        mean_prior_std: float = 0.2,
        learn_mean: bool = True,
        mode: Mode = "variational",
    ) -> None:
        super().__init__()
        self.std = HyperParameter(init_std, transform="exp", prior_std=std_prior_std, mode=mode)
        self.mean = HyperParameter(
            init_mean,
            transform="identity",
            prior_mean=0.0,
            prior_std=mean_prior_std,
            mode=mode if learn_mean else "fixed",
        )

    def forward(self, x: Tensor, y: Tensor) -> AugmentedBatch:
        mean, std = self.mean.sample(), self.std.sample()
        gamma = mean + std * x.new_empty(x.shape).normal_()
        return AugmentedBatch(x=x + gamma, y=y)

    def kl_to_gamma_prior(self, prior_std: float) -> Tensor:
        """``KL(N(mu, sigma^2) || N(0, prior_std^2))`` at the point estimate of ``(mu, sigma)``.

        A prior on the shift ``gamma`` itself rather than on ``phi``; it is minimised at
        ``mu = 0, sigma = prior_std``. This is the regulariser used for Fig. 2 of the paper
        (Sec. 5.1). It is *not* the ``KL(q(phi) || p(phi))`` term of the augmented ELBO,
        which :meth:`kl` returns.
        """
        mean = self.mean.forward_transform(self.mean.loc)
        std = self.std.forward_transform(self.std.loc)
        return gamma_prior_kl(mean, std, prior_std)
