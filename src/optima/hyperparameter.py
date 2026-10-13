"""Augmentation hyperparameters phi with a prior p(phi) and a posterior q(phi).

Every learnable augmentation hyperparameter (a noise scale, a Mixup alpha, a
dropout rate, ...) is written as ``phi = g(u)``, where ``u`` is unconstrained
and ``g`` maps it to the valid range. We place Gaussians on ``u``:

    prior          p(u) = N(m0, s0^2)
    variational    q(u) = N(m,  s^2)

so the KL term in the augmented ELBO (paper, Eq. 7) is available in closed form.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

Transform = Literal["identity", "exp", "softplus", "sigmoid"]
Mode = Literal["variational", "map", "fixed"]


class HyperParameter(nn.Module):
    """A (scalar or vector) augmentation hyperparameter ``phi = g(u)``.

    Args:
        init: Initial value of ``phi`` in the *constrained* space.
        transform: Map ``g`` from unconstrained ``u`` to ``phi``:
            ``"identity"`` (real line), ``"exp"`` / ``"softplus"`` (positive),
            ``"sigmoid"`` (interval ``(0, upper)``).
        prior_mean: Prior centre in the *constrained* space (defaults to ``init``).
        prior_std: Prior standard deviation of ``u`` (unconstrained space).
        init_std: Initial posterior standard deviation of ``u``.
        upper: Upper bound used by the ``"sigmoid"`` transform.
        mode:
            ``"variational"``: ``u ~ q(u)`` by reparameterisation, penalty ``KL(q || p)``
            (Algorithm 1/2 of the paper).
            ``"map"``: point estimate ``u = m``, penalty ``-log p(m)`` up to a constant.
            ``"fixed"``: ``u = m`` is frozen and has no penalty (fixed-augmentation baseline).
    """

    def __init__(
        self,
        init: float | Tensor,
        transform: Transform = "identity",
        prior_mean: float | Tensor | None = None,
        prior_std: float = 1.0,
        init_std: float = 0.1,
        upper: float = 1.0,
        mode: Mode = "variational",
    ) -> None:
        super().__init__()
        if mode not in ("variational", "map", "fixed"):
            raise ValueError(f"Unknown mode: {mode!r}")
        if transform not in ("identity", "exp", "softplus", "sigmoid"):
            raise ValueError(f"Unknown transform: {transform!r}")
        self.transform = transform
        self.upper = upper
        self.mode = mode

        init_u = self.inverse(torch.as_tensor(init, dtype=torch.get_default_dtype()))
        prior_u = init_u if prior_mean is None else self.inverse(
            torch.as_tensor(prior_mean, dtype=torch.get_default_dtype())
        ).expand_as(init_u)

        self.loc = nn.Parameter(init_u.clone(), requires_grad=(mode != "fixed"))
        if mode == "variational":
            self.log_scale = nn.Parameter(torch.full_like(init_u, math.log(init_std)))
        else:
            self.register_parameter("log_scale", None)
        self.register_buffer("prior_loc", prior_u.clone())
        self.register_buffer("prior_scale", torch.full_like(init_u, prior_std))

    # ------------------------------------------------------------------ transforms
    def forward_transform(self, u: Tensor) -> Tensor:
        """Map unconstrained ``u`` to ``phi = g(u)``."""
        if self.transform == "identity":
            return u
        if self.transform == "exp":
            return u.exp()
        if self.transform == "softplus":
            return F.softplus(u)
        return self.upper * torch.sigmoid(u)

    def inverse(self, phi: Tensor) -> Tensor:
        """Map ``phi`` back to ``u = g^{-1}(phi)``."""
        if self.transform == "identity":
            return phi
        if self.transform == "exp":
            return phi.log()
        if self.transform == "softplus":
            return torch.log(torch.expm1(phi))
        return torch.logit(phi / self.upper)

    # ------------------------------------------------------------------ sampling
    @property
    def scale(self) -> Tensor | None:
        """Posterior standard deviation of ``u`` (``None`` unless variational)."""
        return None if self.log_scale is None else self.log_scale.exp()

    def sample(self) -> Tensor:
        """Draw ``phi``: a reparameterised sample from ``q`` in variational mode,
        the point estimate otherwise."""
        if self.mode == "variational":
            u = self.loc + self.scale * torch.randn_like(self.loc)
        else:
            u = self.loc
        return self.forward_transform(u)

    @property
    def value(self) -> Tensor:
        """Deterministic summary ``g(m)`` (for logging and evaluation)."""
        return self.forward_transform(self.loc).detach()

    # ------------------------------------------------------------------ penalty
    def kl(self) -> Tensor:
        """Regulariser on ``phi``: ``KL(q || p)``, ``-log p(m)`` or zero, by mode."""
        m0, s0 = self.prior_loc, self.prior_scale
        if self.mode == "variational":
            s = self.scale
            kl = torch.log(s0 / s) + (s**2 + (self.loc - m0) ** 2) / (2 * s0**2) - 0.5
            return kl.sum()
        if self.mode == "map":
            return (0.5 * ((self.loc - m0) / s0) ** 2 + torch.log(s0)).sum()
        return self.loc.new_zeros(())

    def extra_repr(self) -> str:
        return f"transform={self.transform}, mode={self.mode}, value={self.value.tolist()}"
