"""CIFAR-10 with learned rotations and translations (paper, App. F.1, Table 5).

Reimplementation from the paper's description; the original script was not available.
App. F.1 specifies: ResNet-18 with a Bayesian last layer, CIFAR normalisation, Adam with
lr 1e-4 for the network and 1e-2 for the augmentation, beta_aug = 1, beta_net = 0.1,
30 epochs, batch size 128, 100 MC samples at test time, and a fixed baseline with
rotation and translation scales of 0.1.

Choices the paper leaves open, made here:
    * CIFAR stem (3x3 conv, no max-pool), trained from scratch, no crop/flip.
    * The augmentation penalty is the prior on gamma used by the paper's other experiments,
      KL(N(mu, sigma^2) || N(0, prior_sigma^2)) with prior_sigma = 0.1 per component and
      initial sigma = 0.05, added unscaled to the mean cross-entropy.
    * beta_net multiplies KL(q(w) || p(w)) / N (N = 50000), the per-example ELBO scaling.

Numbers will not match Table 5 exactly; see the README for what this example reproduces.

Methods:
    none     no augmentation
    fixed    gamma = (omega, t_x, t_y) ~ N(0, 0.1^2 I), not learned
    optima   gamma ~ N(mu, diag(sigma^2)) with (mu, sigma) learned

Usage:
    python examples/cifar10_affine.py --method optima --data ./data
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from cifar10_common import evaluate, make_loaders, make_resnet18, save_json, summary

from optima import AffineAugmenter, NoAugmentation, kl_divergence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", choices=["none", "fixed", "optima"], default="optima")
    parser.add_argument("--prior-sigma", type=float, default=0.1,
                        help="prior std of each gamma component; also the fixed sigma")
    parser.add_argument("--init-sigma", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aug-lr", type=float, default=1e-2)
    parser.add_argument("--beta-net", type=float, default=0.1)
    parser.add_argument("--beta-aug", type=float, default=1.0)
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default=None, help="write metrics and history to this JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader, ood_loader = make_loaders(args.data, args.batch_size,
                                                         crop_flip=False, normalize=True,
                                                         workers=args.workers)
    num_train = len(train_loader.dataset)
    model = make_resnet18(pretrained=args.pretrained).to(device)

    if args.method == "none":
        augmenter = NoAugmentation()
    elif args.method == "fixed":
        augmenter = AffineAugmenter((args.prior_sigma,) * 3, mode="fixed")
    else:
        augmenter = AffineAugmenter((args.init_sigma,) * 3, learn_mean=True, mode="map")
    augmenter.to(device)

    groups = [{"params": list(model.parameters()), "lr": args.lr}]
    aug_params = [p for p in augmenter.parameters() if p.requires_grad]
    if aug_params:
        groups.append({"params": aug_params, "lr": args.aug_lr})
    optimizer = torch.optim.Adam(groups)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = (F.cross_entropy(model(augmenter(x, y).x), y)
                    + args.beta_net * kl_divergence(model) / num_train)
            if args.method == "optima":
                loss = loss + args.beta_aug * augmenter.kl_to_gamma_prior(args.prior_sigma)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total, batches = total + loss.item(), batches + 1
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
