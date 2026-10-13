"""Token dropout for text with a learnable dropout rate (a discrete augmentation)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Bernoulli

from optima.augmenters.base import AugmentedBatch, Augmenter
from optima.hyperparameter import HyperParameter, Mode


class TokenDropout(Augmenter):
    """Replace each token by ``[MASK]`` with probability ``p_drop`` (paper, Sec. 5.4).

    ``gamma_t ~ Bernoulli(1 - p_drop)`` is a discrete keep-mask, so there is no
    pathwise gradient. The augmenter returns ``log p(gamma_i | phi)`` per example and
    the objective adds a score-function (REINFORCE) gradient for ``p_drop``.

    ``p_drop = p_max * sigmoid(u)``, with a Gaussian prior on ``u`` centred at
    ``prior_mean``. The paper instead puts a Gaussian prior directly on ``p_drop``; see
    :meth:`quadratic_prior_penalty`.

    Args:
        mask_token_id: Id written in place of dropped tokens.
        pad_token_id: Padding id; padded positions are never dropped. ``None`` to disable.
        special_token_ids: Ids that are never dropped (e.g. ``[CLS]``, ``[SEP]``).
        init_p: Initial ``p_drop``.
        p_max: Upper bound on ``p_drop``.
        prior_mean: Prior centre for ``p_drop`` (defaults to ``init_p``).
        prior_std: Prior std of ``u = logit(p_drop / p_max)``.
        mode: See :class:`HyperParameter`.
    """

    reparameterized = False

    def __init__(
        self,
        mask_token_id: int,
        pad_token_id: int | None = 0,
        special_token_ids: tuple[int, ...] = (),
        init_p: float = 0.04,
        p_max: float = 0.5,
        prior_mean: float | None = None,
        prior_std: float = 1.0,
        mode: Mode = "variational",
    ) -> None:
        super().__init__()
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.special_token_ids = tuple(special_token_ids)
        self.p_drop = HyperParameter(
            init_p, transform="sigmoid", upper=p_max, prior_mean=prior_mean,
            prior_std=prior_std, mode=mode,
        )

    def _droppable(self, x: Tensor) -> Tensor:
        """Boolean mask of positions that may be dropped."""
        droppable = torch.ones_like(x, dtype=torch.bool)
        if self.pad_token_id is not None:
            droppable &= x != self.pad_token_id
        for token_id in self.special_token_ids:
            droppable &= x != token_id
        return droppable

    def forward(self, x: Tensor, y: Tensor) -> AugmentedBatch:
        if x.dtype not in (torch.int32, torch.int64):
            raise ValueError("TokenDropout expects integer token ids")
        p_drop = self.p_drop.sample()
        keep_dist = Bernoulli(probs=(1 - p_drop).expand(x.shape))
        keep = keep_dist.sample()

        droppable = self._droppable(x)
        keep = torch.where(droppable, keep, torch.ones_like(keep))
        x_aug = torch.where(keep.bool(), x, torch.full_like(x, self.mask_token_id))

        log_prob = (keep_dist.log_prob(keep) * droppable).flatten(1).sum(dim=1)
        return AugmentedBatch(x=x_aug, y=y, log_prob=log_prob)

    def quadratic_prior_penalty(self, prior_mean: float, prior_std: float) -> Tensor:
        """``(p_drop - prior_mean)^2 / (2 prior_std^2)`` at the point estimate of ``p_drop``.

        The Gaussian prior placed directly on ``p_drop`` in the paper (App. H), as opposed to
        the prior on ``logit(p_drop / p_max)`` used by :meth:`kl`.
        """
        p_drop = self.p_drop.forward_transform(self.p_drop.loc)
        return (0.5 * ((p_drop - prior_mean) / prior_std) ** 2).sum()
