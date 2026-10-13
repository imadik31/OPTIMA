"""Synthetic 1-D regression (paper, Sec. 5.1, Fig. 2).

Data: 50 training points with x ~ U(-3, 3) and 1000 test points on a grid, from
    y = sin(2x) + 0.5 cos(3x) + eps1 + eps2 sin(x),  eps1 ~ N(0, 0.2^2), eps2 ~ N(0, 0.15^2).

Methods (deterministic ReLU MLP 1-128-512-256-64-1, Adam, full batch, MSE loss):
    none     no augmentation
    fixed    Gaussian input shift with fixed sigma = 0.1 (K = 1)
    naive    fixed sigma = 0.1, loss averaged over K = 5 augmentations
    optima   learned point estimate of (mu, sigma), loss = MSE + KL(N(mu, sigma^2) || N(0, 0.2^2))

These are the settings that produced Fig. 2. The objective is an exact ELBO, but for a
latent-shift model rather than for Eq. 7 of the paper. Each input gets its own shift
``gamma_i ~ N(0, 0.2^2)`` with variational posterior ``q(gamma_i) = N(mu, sigma^2)``.
With a Gaussian likelihood of variance 0.5, the per-example ELBO is
``-(MSE + KL(N(mu, sigma^2) || N(0, 0.2^2)))`` up to a constant, which is this loss
(see :meth:`~optima.GaussianShift.kl_to_gamma_prior`). Eq. 7 instead places the prior on
``phi`` and penalises ``KL(q(phi) || p(phi))``.

The widening of sigma to about 0.18 comes mainly from this prior, whose optimum is
sigma = 0.2. The implied noise sd (about 0.71) is larger than the data's (about 0.23), so the
data term is weak. Under Eq. 7 with a deterministic network (Algorithm 2), sigma shrinks
on this problem: the objective has no Occam factor over the network weights, so less
augmentation always fits the training data better.

``--method all`` runs the four methods in order on one shared random stream, as in the
original script, so ``--seed 42`` reproduces the numbers in Fig. 2. The learned sigma
reaches about 0.18 for every seed tried. The test MSE of learned augmentation varies with
the seed: it edges out ``fixed`` at seed 42 and is worse at seeds 0 and 1.

Usage:
    python examples/synthetic_regression.py --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from optima import GaussianShift, NoAugmentation
from optima.augmenters import Augmenter

GAMMA_PRIOR_STD = 0.2  # prior N(0, 0.2^2) on the shift gamma


def make_data(num_train: int = 50, num_test: int = 1000) -> tuple[Tensor, ...]:
    """Sample the data from NumPy's global random stream (seeded in :func:`main`)."""

    def target(x: np.ndarray) -> np.ndarray:
        eps1 = 0.2 * np.random.normal(0, 1, len(x))
        eps2 = 0.15 * np.sin(x) * np.random.normal(0, 1, len(x))
        return np.sin(2 * x) + 0.5 * np.cos(3 * x) + eps1 + eps2

    x_train = np.random.uniform(-3, 3, num_train)
    y_train = target(x_train)
    x_test = np.linspace(-3, 3, num_test)
    y_test = target(x_test)

    def as_tensor(a: np.ndarray) -> Tensor:
        return torch.tensor(a, dtype=torch.float32).unsqueeze(1)

    return tuple(as_tensor(a) for a in (x_train, y_train, x_test, y_test))


def make_model() -> nn.Module:
    widths = [1, 128, 512, 256, 64]
    layers: list[nn.Module] = []
    for d_in, d_out in zip(widths[:-1], widths[1:]):
        layers += [nn.Linear(d_in, d_out), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(widths[-1], 1))


def make_augmenter(method: str) -> tuple[Augmenter, int]:
    """Return the augmenter and the number of augmentation samples K."""
    if method == "none":
        return NoAugmentation(), 1
    if method == "fixed":
        return GaussianShift(init_std=0.1, learn_mean=False, mode="fixed"), 1
    if method == "naive":
        return GaussianShift(init_std=0.1, learn_mean=False, mode="fixed"), 5
    if method == "optima":
        return GaussianShift(init_std=0.1, mode="map"), 1
    raise ValueError(f"Unknown method: {method!r}")


def run(method: str, data: tuple[Tensor, ...], epochs: int = 3000, lr: float = 1e-4,
        aug_lr: float = 1e-3, log_every: int = 500) -> dict:
    x_train, y_train, x_test, y_test = data
    model = make_model()
    augmenter, num_samples = make_augmenter(method)

    param_groups = [{"params": model.parameters(), "lr": lr}]
    aug_params = [p for p in augmenter.parameters() if p.requires_grad]
    if aug_params:
        param_groups.append({"params": aug_params, "lr": aug_lr})
    optimizer = torch.optim.Adam(param_groups)
    mse = nn.MSELoss()
    sigma_trace = []

    for epoch in range(1, epochs + 1):  # full-batch training
        loss = sum(mse(model(augmenter(x_train, y_train).x), y_train)
                   for _ in range(num_samples)) / num_samples
        if method == "optima":
            loss = loss + augmenter.kl_to_gamma_prior(GAMMA_PRIOR_STD)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if isinstance(augmenter, GaussianShift):
            sigma_trace.append(augmenter.std.value.item())
        if log_every and epoch % log_every == 0:
            hparams = {k: [round(v, 4) for v in t.flatten().tolist()]
                       for k, t in augmenter.hyperparameters().items()}
            print(f"[{method}] epoch {epoch:5d}  loss {loss.item():.4f}  {hparams}")

    with torch.no_grad():
        results = {
            "train_mse": mse(model(x_train), y_train).item(),
            "test_mse": mse(model(x_test), y_test).item(),
            "pred_test": model(x_test).squeeze(1).tolist(),
        }
    if isinstance(augmenter, GaussianShift):
        results["sigma"] = augmenter.std.value.item()
        results["mu"] = augmenter.mean.value.item()
        results["sigma_trace"] = sigma_trace
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", choices=["none", "fixed", "naive", "optima", "all"],
                        default="all")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="write results (with traces) to this JSON")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = make_data()
    methods = ["none", "fixed", "naive", "optima"] if args.method == "all" else [args.method]
    all_results = {}
    for method in methods:
        results = run(method, data, epochs=args.epochs)
        all_results[method] = results
        summary = "  ".join(f"{k}={v:.4f}" for k, v in results.items() if isinstance(v, float))
        print(f"==> {method:7s} {summary}")
    if args.out:
        x_train, y_train, x_test, y_test = (t.squeeze(1).tolist() for t in data)
        payload = {"args": vars(args), "results": all_results,
                   "data": {"x_train": x_train, "y_train": y_train, "x_test": x_test}}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
