"""Random rotations and translations of images with learnable distributions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from optima.augmenters.base import AugmentedBatch, Augmenter, gamma_prior_kl
from optima.hyperparameter import HyperParameter, Mode


class AffineAugmenter(Augmenter):
    """Rotation and translation with ``gamma ~ N(mu, diag(sigma^2))``, one ``gamma`` per image.

    ``gamma = (omega, t_x, t_y)``, or ``(t_x, t_y)`` with ``rotation=False``.
    ``omega`` is in radians; ``t_x, t_y`` are in normalised image coordinates
    (the image spans ``[-1, 1]``, so ``t = 0.1`` is a shift of 5% of the width).
    The scales ``sigma`` (and optionally the means ``mu``) are learned. The warp is
    differentiable, so they get pathwise (reparameterised) gradients.

    Used for App. F.1 (rotation + translation) and App. F.2 (Gaussian translation,
    ``rotation=False, learn_mean=True, padding_mode="reflection"``).

    Args:
        init_std: Initial ``sigma`` per component (3 values, or 2 with ``rotation=False``).
        prior_std: Prior std of ``log sigma`` for :meth:`kl`; the prior is centred at ``init_std``.
        mode: See :class:`HyperParameter`.
        padding_mode: Passed to :func:`torch.nn.functional.grid_sample`.
        rotation: Include the rotation component.
        learn_mean: Learn ``mu``; otherwise ``mu = 0``.
        mean_prior_std: Prior std of ``mu`` for :meth:`kl` (centred at 0).
    """

    def __init__(
        self,
        init_std: Sequence[float] = (0.1, 0.1, 0.1),
        prior_std: float = 1.0,
        mode: Mode = "variational",
        padding_mode: str = "zeros",
        rotation: bool = True,
        learn_mean: bool = False,
        mean_prior_std: float = 0.2,
    ) -> None:
        super().__init__()
        num_components = 3 if rotation else 2
        if len(init_std) != num_components:
            raise ValueError(f"init_std needs {num_components} values, got {len(init_std)}")
        self.rotation = rotation
        self.padding_mode = padding_mode
        self.std = HyperParameter(torch.tensor(init_std), transform="exp", prior_std=prior_std,
                                  mode=mode)
        if learn_mean:
            self.mean = HyperParameter(torch.zeros(num_components), transform="identity",
                                       prior_mean=0.0, prior_std=mean_prior_std, mode=mode)
        else:
            self.mean = None

    def _point_mean(self) -> Tensor:
        if self.mean is None:
            return torch.zeros_like(self.std.loc)
        return self.mean.forward_transform(self.mean.loc)

    def forward(self, x: Tensor, y: Tensor) -> AugmentedBatch:
        if x.dim() != 4:
            raise ValueError(f"Expected images of shape [B, C, H, W], got {tuple(x.shape)}")
        batch_size = x.shape[0]
        std = self.std.sample()
        mean = self.mean.sample() if self.mean is not None else torch.zeros_like(std)
        eps = torch.randn(batch_size, len(std), device=x.device, dtype=x.dtype)
        gamma = mean + std * eps
        if self.rotation:
            omega, tx, ty = gamma.unbind(dim=1)
        else:
            tx, ty = gamma.unbind(dim=1)
            omega = torch.zeros_like(tx)
        cos, sin = omega.cos(), omega.sin()
        theta = torch.stack(
            [torch.stack([cos, -sin, tx], dim=1), torch.stack([sin, cos, ty], dim=1)], dim=1
        )  # [B, 2, 3]
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x_aug = F.grid_sample(x, grid, padding_mode=self.padding_mode, align_corners=False)
        return AugmentedBatch(x=x_aug, y=y)

    def kl_to_gamma_prior(self, prior_std: float | Sequence[float]) -> Tensor:
        """``KL(N(mu, sigma^2) || N(0, prior_std^2))`` at the point estimate of ``(mu, sigma)``.

        The prior on ``gamma`` used in the paper's CIFAR-10 experiments; see
        :func:`~optima.augmenters.base.gamma_prior_kl`.
        """
        std = self.std.forward_transform(self.std.loc)
        return gamma_prior_kl(self._point_mean(), std, torch.as_tensor(prior_std, dtype=std.dtype))
