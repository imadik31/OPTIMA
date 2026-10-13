"""OPTIMA: Optimizing Data Augmentation through Bayesian Model Selection (ICLR 2026)."""

from optima.augmenters import (
    AffineAugmenter,
    AugmentedBatch,
    Augmenter,
    GaussianShift,
    LearnableMixup,
    NoAugmentation,
    TokenDropout,
    gamma_prior_kl,
)
from optima.bayesian import BayesianLinear, kl_divergence, set_sampling
from optima.hyperparameter import HyperParameter
from optima.objective import (
    ObjectiveOutput,
    OptimaObjective,
    categorical_log_likelihood,
    gaussian_log_likelihood,
)
from optima.train import optima_step, train_epoch

__version__ = "0.1.0"

__all__ = [
    "AffineAugmenter",
    "AugmentedBatch",
    "Augmenter",
    "BayesianLinear",
    "GaussianShift",
    "HyperParameter",
    "LearnableMixup",
    "NoAugmentation",
    "ObjectiveOutput",
    "OptimaObjective",
    "TokenDropout",
    "categorical_log_likelihood",
    "gamma_prior_kl",
    "gaussian_log_likelihood",
    "kl_divergence",
    "optima_step",
    "set_sampling",
    "train_epoch",
]
