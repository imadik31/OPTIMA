import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from optima import (
    BayesianLinear,
    LearnableMixup,
    OptimaObjective,
    TokenDropout,
    categorical_log_likelihood,
    optima_step,
    train_epoch,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_optima_step_gives_gradients_to_model_and_augmenter():
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 8), nn.ReLU(), BayesianLinear(8, 3))
    augmenter = LearnableMixup(num_classes=3)
    objective = OptimaObjective(num_train=100, bound="multi_sample")
    x, y = torch.randn(6, 3, 2, 2), torch.randint(0, 3, (6,))
    out = optima_step(model, augmenter, objective, x, y, categorical_log_likelihood,
                      num_samples=3)
    out.loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    assert augmenter.alpha.loc.grad is not None
    assert out.kl_net.item() > 0


def test_score_function_path_reaches_discrete_augmenter():
    model = nn.Sequential(nn.Embedding(50, 8), nn.Flatten(), nn.Linear(8 * 5, 2))
    augmenter = TokenDropout(mask_token_id=1, pad_token_id=0, init_p=0.2)
    objective = OptimaObjective(num_train=100)
    x, y = torch.randint(2, 50, (8, 5)), torch.randint(0, 2, (8,))
    out = optima_step(model, augmenter, objective, x, y, categorical_log_likelihood)
    out.loss.backward()
    assert augmenter.p_drop.loc.grad is not None


def test_train_epoch_runs_and_updates_parameters():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), BayesianLinear(12, 3))
    augmenter = LearnableMixup(num_classes=3)
    data = TensorDataset(torch.randn(32, 3, 2, 2), torch.randint(0, 3, (32,)))
    loader = DataLoader(data, batch_size=8)
    objective = OptimaObjective(num_train=len(data))
    optimizer = torch.optim.Adam([*model.parameters(), *augmenter.parameters()], lr=1e-2)
    before = augmenter.alpha.loc.detach().clone()
    stats = train_epoch(model, augmenter, objective, loader, optimizer,
                        categorical_log_likelihood, max_grad_norm=1.0)
    assert set(stats) == {"loss", "objective", "data_term", "kl_net", "kl_aug"}
    assert not torch.equal(before, augmenter.alpha.loc.detach())


def test_synthetic_regression_example_smoke():
    spec = importlib.util.spec_from_file_location("synthetic_regression",
                                                  EXAMPLES / "synthetic_regression.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.make_data()
    for method in ("none", "fixed", "naive", "optima"):
        results = module.run(method, data, epochs=5, log_every=0)
        assert results["test_mse"] > 0
