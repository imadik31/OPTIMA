import math

import pytest
import torch
from torch.distributions import Bernoulli

from optima import OptimaObjective, categorical_log_likelihood, gaussian_log_likelihood

ZERO = torch.zeros(())


def test_categorical_log_likelihood_hard_and_soft_labels_agree():
    logits = torch.randn(6, 4)
    labels = torch.randint(0, 4, (6,))
    soft = torch.nn.functional.one_hot(labels, 4).float()
    assert torch.allclose(categorical_log_likelihood(logits, labels),
                          categorical_log_likelihood(logits, soft))


def test_gaussian_log_likelihood_matches_torch():
    pred, target = torch.randn(5, 2), torch.randn(5, 2)
    expected = torch.distributions.Normal(pred, 0.3).log_prob(target).sum(dim=1)
    assert torch.allclose(gaussian_log_likelihood(pred, target, 0.3), expected)


def test_kl_is_scaled_by_dataset_size():
    obj = OptimaObjective(num_train=100)
    log_lik = torch.full((1, 4), -2.0)
    out = obj(log_lik, kl_net=torch.tensor(10.0), kl_aug=torch.tensor(5.0))
    assert out.loss.item() == pytest.approx(2.0 + 15.0 / 100)


def test_bounds_agree_at_one_sample():
    log_lik = torch.randn(1, 8)
    values = [OptimaObjective(10, bound=b).data_term(log_lik) for b in
              ("elbo", "multi_sample", "summed")]
    for v in values[1:]:
        assert torch.allclose(values[0], v)


def test_multi_sample_bound_dominates_elbo_and_summed_is_k_times():
    log_lik = torch.randn(5, 8)
    elbo = OptimaObjective(10, "elbo").data_term(log_lik)
    multi = OptimaObjective(10, "multi_sample").data_term(log_lik)
    summed = OptimaObjective(10, "summed").data_term(log_lik)
    assert (multi >= elbo - 1e-6).all()  # Jensen
    assert torch.allclose(summed, 5 * elbo)


def test_multi_sample_bound_is_log_mean_exp():
    log_lik = torch.tensor([[math.log(0.2)], [math.log(0.6)]])
    multi = OptimaObjective(10, "multi_sample").data_term(log_lik)
    assert multi.item() == pytest.approx(math.log(0.4))


@pytest.mark.parametrize("bound", ["elbo", "multi_sample"])
def test_score_function_gradient_is_unbiased(bound):
    # gamma ~ Bernoulli(q), l = gamma (log-lik independent of q pathwise).
    # Data term expectation: E[l] = q for "elbo"; d/dq = 1 so d loss / dq = -1.
    torch.manual_seed(0)
    q = torch.tensor(0.3, requires_grad=True)
    k, batch = (1, 200_000)
    dist = Bernoulli(probs=q.expand(k, batch))
    gamma = dist.sample()
    log_lik = gamma  # any function of the sample, no pathwise dependence on q
    out = OptimaObjective(10, bound)(log_lik, ZERO, ZERO, log_prob=dist.log_prob(gamma))
    out.loss.backward()
    assert q.grad.item() == pytest.approx(-1.0, abs=0.02)


def test_score_function_does_not_change_loss_value():
    log_lik = torch.randn(2, 6)
    log_prob = torch.randn(2, 6, requires_grad=True)
    obj = OptimaObjective(10)
    with_score = obj(log_lik, ZERO, ZERO, log_prob=log_prob).loss
    without = obj(log_lik, ZERO, ZERO).loss
    assert with_score.item() == pytest.approx(without.item())


def test_invalid_inputs():
    with pytest.raises(ValueError):
        OptimaObjective(10, bound="iwae")
    with pytest.raises(ValueError):
        OptimaObjective(0)
    with pytest.raises(ValueError):
        OptimaObjective(10)(torch.randn(4), ZERO, ZERO)
