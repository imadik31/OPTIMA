import pytest
import torch

from optima import HyperParameter


@pytest.mark.parametrize("transform", ["identity", "exp", "softplus", "sigmoid"])
def test_inverse_is_left_inverse(transform):
    hp = HyperParameter(0.3, transform=transform, upper=0.5)
    phi = torch.tensor([0.05, 0.2, 0.4])
    assert torch.allclose(hp.forward_transform(hp.inverse(phi)), phi, atol=1e-6)


def test_init_value_is_respected():
    hp = HyperParameter(0.2, transform="exp")
    assert hp.value.item() == pytest.approx(0.2)


def test_kl_zero_when_posterior_equals_prior():
    hp = HyperParameter(0.2, transform="exp", prior_std=0.1, init_std=0.1)
    assert hp.kl().item() == pytest.approx(0.0, abs=1e-7)


def test_kl_positive_and_differentiable():
    hp = HyperParameter(0.2, transform="exp", prior_mean=1.0, prior_std=0.5, init_std=0.1)
    kl = hp.kl()
    assert kl.item() > 0
    kl.backward()
    assert hp.loc.grad is not None and hp.log_scale.grad is not None


def test_vector_hyperparameter_shapes():
    hp = HyperParameter(torch.tensor([0.1, 0.2, 0.3]), transform="exp")
    assert hp.sample().shape == (3,)
    assert hp.kl().dim() == 0


def test_map_mode_is_deterministic():
    hp = HyperParameter(0.2, transform="exp", mode="map")
    assert hp.log_scale is None
    assert torch.equal(hp.sample(), hp.sample())


def test_fixed_mode_has_no_trainable_parameters_and_no_penalty():
    hp = HyperParameter(0.2, transform="exp", mode="fixed")
    assert not any(p.requires_grad for p in hp.parameters())
    assert hp.kl().item() == 0.0


def test_invalid_arguments():
    with pytest.raises(ValueError):
        HyperParameter(0.2, mode="bayes")
    with pytest.raises(ValueError):
        HyperParameter(0.2, transform="tanh")
