"""CIFAR-10 with learned Gaussian translations (paper, App. F.2, Table 6).

Port of the script behind Table 6, with its gradient bug fixed (see the README, "Notes on
the paper"). Pretrained ResNet-18 with a CIFAR stem and a Bayesian last layer, trained
with Adam (lr 1e-3, StepLR /10 every epochs/3) on CIFAR-10 with the standard crop + flip
for every method.

Methods:
    none     crop + flip only
    fixed    plus Gaussian translation (t_x, t_y) ~ N(0, prior_sigma^2), not learned
    naive    as fixed, but the loss is averaged over K = --num-samples translations
    optima   plus translation ~ N(mu, diag(sigma^2)) with (mu, sigma) learned (lr 1e-2), under
             loss = CE + 1e-4 KL(q(w) || p(w)) + KL(N(mu, sigma^2) || N(0, prior_sigma^2))

The translation is in normalised coordinates (1.0 = half the image width) with reflection
padding. Evaluation uses the final-epoch model, 30 MC samples of the Bayesian layer, and
SVHN as the OOD set (predictive-entropy AUROC).

Usage:
    python examples/cifar10_translation.py --method optima --prior-sigma 0.5 --data ./data
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from cifar10_common import evaluate, make_loaders, make_resnet18, save_json, summary

from optima import AffineAugmenter, NoAugmentation, kl_divergence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", choices=["none", "fixed", "naive", "optima"], default="optima")
    parser.add_argument("--prior-sigma", type=float, default=0.5,
                        help="prior std of the translation; also the fixed/naive sigma")
    parser.add_argument("--init-sigma", type=float, default=0.05)
    parser.add_argument("--num-samples", type=int, default=2, help="K for --method naive")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aug-lr", type=float, default=1e-2)
    parser.add_argument("--kl-net", type=float, default=1e-4, help="weight on KL(q(w)||p(w))")
    parser.add_argument("--kl-aug", type=float, default=1.0, help="weight on the gamma prior KL")
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default=None, help="write metrics and history to this JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader, ood_loader = make_loaders(args.data, args.batch_size,
                                                         crop_flip=True, normalize=False,
                                                         workers=args.workers)
    model = make_resnet18(pretrained=True).to(device)

    if args.method == "none":
        augmenter = NoAugmentation()
    elif args.method in ("fixed", "naive"):
        augmenter = AffineAugmenter((args.prior_sigma,) * 2, rotation=False, mode="fixed",
                                    padding_mode="reflection")
    else:
        augmenter = AffineAugmenter((args.init_sigma,) * 2, rotation=False, learn_mean=True,
                                    mode="map", padding_mode="reflection")
    augmenter.to(device)
    num_samples = args.num_samples if args.method == "naive" else 1

    groups = [{"params": list(model.parameters()), "lr": args.lr}]
    aug_params = [p for p in augmenter.parameters() if p.requires_grad]
    if aug_params:
        groups.append({"params": aug_params, "lr": args.aug_lr})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1),
                                                gamma=0.1)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            ce = sum(F.cross_entropy(model(augmenter(x, y).x), y)
                     for _ in range(num_samples)) / num_samples
            loss = ce + args.kl_net * kl_divergence(model)
            if args.method == "optima":
                loss = loss + args.kl_aug * augmenter.kl_to_gamma_prior(args.prior_sigma)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total, batches = total + loss.item(), batches + 1
        scheduler.step()
        hparams = {k: v.flatten().tolist() for k, v in augmenter.hyperparameters().items()}
        history.append({"epoch": epoch, "loss": total / batches, **hparams})
        print(f"epoch {epoch:3d}  loss {total / batches:.4f}  "
              + "  ".join(f"{k}={[round(v, 4) for v in vals]}" for k, vals in hparams.items()),
              flush=True)

    metrics = evaluate(model, test_loader, ood_loader, args.mc_samples, device)
    print("test:", summary(metrics))
    save_json(args.out, {"args": vars(args), "test": metrics, "history": history})


if __name__ == "__main__":
    main()
