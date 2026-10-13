"""SST-5 sentiment classification with learned token dropout (paper, Sec. 5.4, Table 4).

Reimplementation from the paper's description (App. H); the original script was not
available. Fine-tunes DistilBERT for 5 epochs with HuggingFace's default optimisation
(AdamW, lr 2e-5, batch size 8, linear decay, no warm-up, no weight decay).

Token dropout replaces each token by [MASK] with probability p_drop = p_max * sigmoid(s),
p_max = 0.5. The mask is discrete, so ``s`` gets a score-function (REINFORCE) gradient
(:class:`optima.TokenDropout` + :class:`optima.OptimaObjective`). ``s`` uses Adam with a
constant lr 5e-2.

Methods:
    none     no augmentation
    fixed    token dropout with fixed p_drop (--p, default 0.04 = the OPTIMA initialisation)
    optima   p_drop learned from 0.04 under a Gaussian prior on p_drop itself,
             N(--prior-mean, 0.1^2), added as an unscaled quadratic penalty to the mean
             cross-entropy

App. H states s_0 = -2 and an initial p_drop of about 0.04; with p_max = 0.5 these
disagree (0.5 sigmoid(-2) = 0.06). This script starts at 0.04, the value of the paper's
fixed baseline.

Requires ``transformers`` and ``datasets``.

Usage:
    python examples/sst5_token_dropout.py --method optima --prior-mean 0.3 --seed 0
    python examples/sst5_token_dropout.py --method fixed --p 0.0625
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from optima import NoAugmentation, OptimaObjective, TokenDropout, categorical_log_likelihood
from optima.metrics import accuracy, expected_calibration_error, negative_log_likelihood

MODEL = "distilbert-base-uncased"


def encode(tokenizer, split, max_length: int) -> TensorDataset:
    enc = tokenizer(split["text"], padding="max_length", truncation=True, max_length=max_length,
                    return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(split["label"]))


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    probs, labels = [], []
    for ids, mask, y in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        probs.append(logits.softmax(-1).cpu())
        labels.append(y)
    probs, labels = torch.cat(probs), torch.cat(labels)
    return {"acc": accuracy(probs, labels), "nll": negative_log_likelihood(probs, labels),
            "ece": expected_calibration_error(probs, labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", choices=["none", "fixed", "optima"], default="optima")
    parser.add_argument("--p", type=float, default=0.04, help="p_drop for --method fixed")
    parser.add_argument("--init-p", type=float, default=0.04)
    parser.add_argument("--p-max", type=float, default=0.5)
    parser.add_argument("--prior-mean", type=float, default=0.1)
    parser.add_argument("--prior-std", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--aug-lr", type=float, default=5e-2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--out", default=None, help="write metrics and history to this JSON")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    data = load_dataset("SetFit/sst5")
    train = encode(tokenizer, data["train"], args.max_length)
    test = encode(tokenizer, data["test"], args.max_length)
    train_loader = DataLoader(train, args.batch_size, shuffle=True)
    test_loader = DataLoader(test, 64)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=5).to(device)
    specials = (tokenizer.cls_token_id, tokenizer.sep_token_id)
    if args.method == "none":
        augmenter = NoAugmentation()
    else:
        augmenter = TokenDropout(
            mask_token_id=tokenizer.mask_token_id, pad_token_id=tokenizer.pad_token_id,
            special_token_ids=specials, p_max=args.p_max,
            init_p=args.p if args.method == "fixed" else args.init_p,
            mode="fixed" if args.method == "fixed" else "map",
        )
    augmenter.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps)
    aug_optimizer = (torch.optim.Adam(augmenter.parameters(), lr=args.aug_lr)
                     if args.method == "optima" else None)
    # Data term and REINFORCE gradient only; the prior on p_drop is added below.
    objective = OptimaObjective(num_train=len(train), beta_net=0.0, beta_aug=0.0)
    zero = torch.zeros((), device=device)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            batch = augmenter(ids, y)
            logits = model(input_ids=batch.x, attention_mask=mask).logits
            log_lik = categorical_log_likelihood(logits, y).unsqueeze(0)  # [K=1, B]
            log_prob = batch.log_prob.unsqueeze(0) if batch.log_prob is not None else None
            loss = objective(log_lik, zero, zero, log_prob).loss
            if args.method == "optima":
                loss = loss + augmenter.quadratic_prior_penalty(args.prior_mean, args.prior_std)
            optimizer.zero_grad(set_to_none=True)
            if aug_optimizer is not None:
                aug_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if aug_optimizer is not None:
                aug_optimizer.step()
            total += loss.item()
        p_drop = augmenter.p_drop.value.item() if hasattr(augmenter, "p_drop") else 0.0
        history.append({"epoch": epoch, "loss": total / len(train_loader), "p_drop": p_drop})
        print(f"epoch {epoch}  loss {total / len(train_loader):.4f}  p_drop {p_drop:.4f}",
              flush=True)

    metrics = evaluate(model, test_loader, device)
    print("test:", "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"args": vars(args), "test": metrics, "history": history}, indent=2))


if __name__ == "__main__":
    main()
