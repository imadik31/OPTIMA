"""Mean-field Gaussian layers for (partially) Bayesian networks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BayesianLinear(nn.Module):
    """Linear layer with a factorised Gaussian posterior over weights and biases.

    ``q(w) = N(mu, sigma^2)`` with ``sigma = softplus(rho)`` and prior ``N(0, prior_std^2)``.
    The forward pass uses the local reparameterisation trick (Kingma et al., 2015):
    pre-activations are sampled directly, which gives independent weight noise per
    example and lower-variance gradients than sampling one weight matrix per batch.

    Set ``layer.sample = False`` to use the posterior mean (deterministic forward).

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        prior_std: Standard deviation of the zero-mean Gaussian prior.
        init_std: Initial posterior standard deviation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        init_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self.sample = True

        bound = 1.0 / math.sqrt(in_features)
        rho = math.log(math.expm1(init_std))
        weight_mu = torch.empty(out_features, in_features).uniform_(-bound, bound)
        self.weight_mu = nn.Parameter(weight_mu)
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), rho))
        self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))
        self.bias_rho = nn.Parameter(torch.full((out_features,), rho))

    @property
    def weight_std(self) -> Tensor:
        return F.softplus(self.weight_rho)

    @property
    def bias_std(self) -> Tensor:
        return F.softplus(self.bias_rho)

    def forward(self, x: Tensor) -> Tensor:
        mean = F.linear(x, self.weight_mu, self.bias_mu)
        if not self.sample:
            return mean
        var = F.linear(x**2, self.weight_std**2, self.bias_std**2)
        return mean + var.clamp_min(1e-16).sqrt() * torch.randn_like(mean)

    def kl(self) -> Tensor:
        """``KL(q(w) || p(w))`` summed over all weights and biases."""
        return _gaussian_kl(self.weight_mu, self.weight_std, self.prior_std) + _gaussian_kl(
            self.bias_mu, self.bias_std, self.prior_std
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"prior_std={self.prior_std}"
        )


def _gaussian_kl(mu: Tensor, std: Tensor, prior_std: float) -> Tensor:
    """``KL(N(mu, std^2) || N(0, prior_std^2))`` summed over elements."""
    kl = math.log(prior_std) - torch.log(std) + (std**2 + mu**2) / (2 * prior_std**2) - 0.5
    return kl.sum()


def kl_divergence(model: nn.Module) -> Tensor:
    """Sum of ``KL(q || p)`` over all :class:`BayesianLinear` layers of ``model``.

    Returns zero for a deterministic network (Algorithm 2 of the paper).
    """
    terms = [m.kl() for m in model.modules() if isinstance(m, BayesianLinear)]
    if not terms:
        return torch.zeros(())
    return torch.stack(terms).sum()


def set_sampling(model: nn.Module, sample: bool) -> None:
    """Switch every :class:`BayesianLinear` layer between sampling and posterior mean."""
    for module in model.modules():
        if isinstance(module, BayesianLinear):
            module.sample = sample
