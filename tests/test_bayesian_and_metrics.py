import pytest
import torch
from torch import nn

from optima import BayesianLinear, kl_divergence, set_sampling
from optima.metrics import (
    accuracy,
    auroc,
    expected_calibration_error,
    mc_predict,
    predictive_entropy,
)


def test_bayesian_linear_is_stochastic_and_has_positive_kl():
    layer = BayesianLinear(4, 3, init_std=0.5)
    x = torch.randn(10, 4)
    assert not torch.allclose(layer(x), layer(x))
    assert layer.kl().item() > 0


def test_bayesian_linear_mean_mode_is_deterministic():
    layer = BayesianLinear(4, 3)
    set_sampling(layer, False)
    x = torch.randn(10, 4)
    assert torch.equal(layer(x), layer(x))
    expected = nn.functional.linear(x, layer.weight_mu, layer.bias_mu)
    assert torch.allclose(layer(x), expected)


def test_kl_divergence_sums_layers_and_is_zero_for_deterministic_nets():
    model = nn.Sequential(BayesianLinear(4, 4), nn.ReLU(), BayesianLinear(4, 2))
    assert torch.allclose(kl_divergence(model), model[0].kl() + model[2].kl())
    assert kl_divergence(nn.Linear(3, 3)).item() == 0.0


def test_mc_predict_returns_probabilities():
    model = nn.Sequential(BayesianLinear(4, 3))
    probs = mc_predict(model, torch.randn(5, 4), num_samples=7)
    assert probs.shape == (5, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(5))


def test_ece_is_zero_for_confident_correct_predictions():
    labels = torch.tensor([0, 1, 2])
    probs = torch.eye(3)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)
    assert accuracy(probs, labels) == 1.0


def test_ece_known_value():
    # Two predictions with confidence 0.8, one correct: |0.5 - 0.8| = 0.3.
    probs = torch.tensor([[0.8, 0.2], [0.8, 0.2]])
    labels = torch.tensor([0, 1])
    assert expected_calibration_error(probs, labels) == pytest.approx(0.3)


def test_auroc_extremes_and_ties():
    assert auroc(torch.tensor([0.1, 0.2]), torch.tensor([0.8, 0.9])) == pytest.approx(1.0)
    assert auroc(torch.tensor([0.8, 0.9]), torch.tensor([0.1, 0.2])) == pytest.approx(0.0)
    assert auroc(torch.ones(4), torch.ones(3)) == pytest.approx(0.5)


def test_predictive_entropy_bounds():
    uniform = torch.full((1, 4), 0.25)
    assert predictive_entropy(uniform).item() == pytest.approx(torch.log(torch.tensor(4.0)).item())
    assert predictive_entropy(torch.eye(4)).max().item() == pytest.approx(0.0, abs=1e-6)
