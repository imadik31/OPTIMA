"""Build the README figures from the JSON files written by the examples (``--out``).

Usage:
    python scripts/make_figures.py --runs ../runs --out docs/figures

Expected layout under ``--runs`` (one directory per configuration, one JSON per seed):
    synthetic/seed42.json
    t_none/ t_opt_p0.1/ t_opt_p0.5/ t_opt_p1.0/   (cifar10_translation.py)
    s_opt_mu0.1/ s_opt_mu0.3/                      (sst5_token_dropout.py)
Missing inputs are skipped. Requires matplotlib.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-3 of the reference data-viz palette (validated for all-pairs use),
# neutral inks for text, references and raw data.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def label_end(ax, x, y, text, color, dy=0):
    """Direct label at the end of a line, in text ink, nudged by ``dy`` points."""
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                color=INK_2, fontsize=9)
    ax.plot([x], [y], "o", color=color, markersize=4)


def synthetic(runs: Path, out: Path) -> None:
    data = load(runs / "synthetic" / "seed42.json")
    if data is None:
        return
    res, d = data["results"], data["data"]
    xt = d["x_test"]
    truth = [math.sin(2 * x) + 0.5 * math.cos(3 * x) for x in xt]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.scatter(d["x_train"], d["y_train"], s=14, color=MUTED, label="training data", zorder=3)
    ax1.plot(xt, truth, color=INK_2, linestyle="--", linewidth=1.5, label="true function")
    ax1.plot(xt, res["fixed"]["pred_test"], color=BLUE,
             label=f"fixed σ=0.1 (test MSE {res['fixed']['test_mse']:.3f})")
    ax1.plot(xt, res["optima"]["pred_test"], color=ORANGE,
             label=f"learned σ (test MSE {res['optima']['test_mse']:.3f})")
    ax1.set(xlabel="x", ylabel="y", title="Predictions (seed 42)")
    ax1.legend(fontsize=8, loc="lower left")

    trace = res["optima"]["sigma_trace"]
    ax2.plot(range(1, len(trace) + 1), trace, color=ORANGE)
    ax2.axhline(0.1, color=INK_2, linestyle="--", linewidth=1.2)
    ax2.axhline(0.2, color=MUTED, linestyle=":", linewidth=1.2)
    ax2.text(len(trace), 0.1, " fixed σ = 0.1", va="bottom", ha="right", color=INK_2, fontsize=9)
    ax2.text(len(trace), 0.2, " prior optimum σ = 0.2", va="bottom", ha="right", color=INK_2,
             fontsize=9)
    label_end(ax2, len(trace), trace[-1], f"{trace[-1]:.3f}", ORANGE)
    ax2.set(xlabel="epoch", ylabel="learned σ", title="Learned shift scale σ", ylim=(0.08, 0.22))
    fig.tight_layout()
    fig.savefig(out / "synthetic_regression.png", dpi=160)
    plt.close(fig)


def first_seed(runs: Path, name: str) -> dict | None:
    files = sorted((runs / name).glob("seed*.json"))
    return load(files[0]) if files else None


def translation_sigma(runs: Path, out: Path) -> None:
    priors = [("t_opt_p0.1", 0.1), ("t_opt_p0.5", 0.5), ("t_opt_p1.0", 1.0)]
    loaded = [(first_seed(runs, n), p) for n, p in priors]
    if not all(r for r, _ in loaded):
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, (run, prior) in zip(axes, loaded):
        hist = run["history"]
        epochs = [h["epoch"] for h in hist]
        ends = [hist[-1]["std"][i] / prior for i in range(2)]
        close = abs(ends[0] - ends[1]) < 0.03  # nudge overlapping end labels apart
        for i, (name, color) in enumerate([("t_x", BLUE), ("t_y", ORANGE)]):
            ratio = [h["std"][i] / prior for h in hist]
            ax.plot(epochs, ratio, color=color, label=name)
            dy = (6 if ends[i] >= ends[1 - i] else -6) if close else 0
            label_end(ax, epochs[-1], ratio[-1], name, color, dy)
        ax.axhline(1.0, color=INK_2, linestyle="--", linewidth=1.2)
        ax.set(title=f"prior σ = {prior}", xlabel="epoch")
    axes[0].set_ylabel("learned σ / prior σ")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("CIFAR-10 Gaussian translation: learned scale relative to the prior", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "cifar10_translation_sigma.png", dpi=160)
    plt.close(fig)


def reliability(runs: Path, out: Path) -> None:
    pairs = [("t_none", "crop + flip only", BLUE),
             ("t_opt_p0.5", "OPTIMA, prior σ = 0.5", ORANGE)]
    loaded = [(first_seed(runs, n), lab, c) for n, lab, c in pairs]
    if not all(r for r, _, _ in loaded):
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=1.2, label="perfect calibration")
    for run, lab, color in loaded:
        rel = run["test"]["reliability"]
        pts = [(c, a) for c, a, n in zip(rel["confidence"], rel["accuracy"], rel["count"])
               if n >= 50]
        ax.plot(*zip(*pts), color=color, marker="o", markersize=5,
                label=f"{lab} (ECE {run['test']['ece']:.3f})")
    ax.set(xlabel="confidence", ylabel="accuracy", title="CIFAR-10 reliability diagram",
           xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "cifar10_reliability.png", dpi=160)
    plt.close(fig)


def sst_pdrop(runs: Path, out: Path) -> None:
    configs = [("s_opt_mu0.1", 0.1, BLUE), ("s_opt_mu0.3", 0.3, ORANGE)]
    loaded = [(first_seed(runs, n), mu, c) for n, mu, c in configs]
    if not all(r for r, _, _ in loaded):
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for run, mu, color in loaded:
        hist = run["history"]
        epochs = [0] + [h["epoch"] for h in hist]
        p = [run["args"]["init_p"]] + [h["p_drop"] for h in hist]
        ax.plot(epochs, p, color=color, marker="o", markersize=5, label=f"prior mean μ = {mu}")
        ax.axhline(mu, color=color, linestyle=":", linewidth=1.0, alpha=0.7)
        label_end(ax, epochs[-1], p[-1], f"{p[-1]:.3f}", color)
    ax.set(xlabel="epoch", ylabel="learned p_drop", title="SST-5: learned token-dropout rate")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "sst5_pdrop.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="docs/figures")
    args = parser.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for build in (synthetic, translation_sigma, reliability, sst_pdrop):
        build(runs, out)
    print("figures:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
