import pytest
import torch

from optima import AffineAugmenter, GaussianShift, LearnableMixup, NoAugmentation, TokenDropout


def test_no_augmentation_is_identity():
    x, y = torch.randn(4, 3), torch.arange(4)
    out = NoAugmentation()(x, y)
    assert out.x is x and out.y is y and out.log_prob is None
    assert NoAugmentation().kl().item() == 0.0


def test_gaussian_shift_shapes_and_gradients():
    aug = GaussianShift(init_std=0.1)
    x, y = torch.randn(8, 2), torch.randn(8, 1)
    out = aug(x, y)
    assert out.x.shape == x.shape and out.y is y
    out.x.pow(2).sum().backward()
    assert aug.std.loc.grad is not None and aug.mean.loc.grad is not None


def test_gaussian_shift_fixed_mode():
    aug = GaussianShift(init_std=0.1, learn_mean=False, mode="fixed")
    assert not any(p.requires_grad for p in aug.parameters())
    assert aug.kl().item() == 0.0
    assert set(aug.hyperparameters()) == {"std", "mean"}


def test_gaussian_shift_gamma_prior_kl():
    aug = GaussianShift(init_std=0.2, mode="map")
    assert aug.kl_to_gamma_prior(0.2).item() == pytest.approx(0.0, abs=1e-6)
    aug = GaussianShift(init_std=0.1, init_mean=0.05, mode="map")
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(0.05, 0.1), torch.distributions.Normal(0.0, 0.2)
    )
    kl = aug.kl_to_gamma_prior(0.2)
    assert kl.item() == pytest.approx(expected.item(), rel=1e-5)
    kl.backward()
    assert aug.std.loc.grad.item() < 0  # sigma below the prior std is pushed up


def test_affine_identity_when_scale_is_tiny():
    aug = AffineAugmenter(init_std=(1e-8, 1e-8, 1e-8), mode="fixed")
    x = torch.rand(2, 3, 8, 8)
    assert torch.allclose(aug(x, torch.zeros(2)).x, x, atol=1e-5)


def test_affine_gradients_reach_scales():
    aug = AffineAugmenter(init_std=(0.3, 0.1, 0.1))
    x = torch.rand(4, 1, 16, 16)
    aug(x, torch.zeros(4)).x.sum().backward()
    assert aug.std.loc.grad is not None and torch.isfinite(aug.std.loc.grad).all()


def test_affine_translation_only_with_learned_mean():
    aug = AffineAugmenter(init_std=(0.05, 0.05), rotation=False, learn_mean=True,
                          padding_mode="reflection")
    x = torch.rand(4, 3, 16, 16)
    aug(x, torch.zeros(4)).x.square().mean().backward()
    assert aug.std.loc.grad is not None and aug.mean.loc.grad is not None
    assert set(aug.hyperparameters()) == {"std", "mean"}


def test_affine_init_std_length_must_match_components():
    with pytest.raises(ValueError):
        AffineAugmenter(init_std=(0.1, 0.1, 0.1), rotation=False)
    with pytest.raises(ValueError):
        AffineAugmenter(init_std=(0.1, 0.1))


def test_affine_gamma_prior_kl_is_zero_at_prior():
    aug = AffineAugmenter(init_std=(0.1, 0.2, 0.3), mode="map")
    assert aug.kl_to_gamma_prior((0.1, 0.2, 0.3)).item() == pytest.approx(0.0, abs=1e-6)
    assert aug.kl_to_gamma_prior(0.5).item() > 0


def test_affine_rejects_non_images():
    with pytest.raises(ValueError):
        AffineAugmenter()(torch.randn(4, 10), torch.zeros(4))


def test_mixup_soft_labels_and_gradients():
    aug = LearnableMixup(num_classes=5, init_alpha=0.5)
    x, y = torch.randn(16, 3, 4, 4), torch.randint(0, 5, (16,))
    out = aug(x, y)
    assert out.x.shape == x.shape and out.y.shape == (16, 5)
    assert torch.allclose(out.y.sum(dim=1), torch.ones(16))
    out.x.sum().backward()
    assert aug.alpha.loc.grad is not None


def test_token_dropout_respects_padding_and_bounds():
    torch.manual_seed(0)
    aug = TokenDropout(mask_token_id=103, pad_token_id=0, special_token_ids=(101,),
                       init_p=0.45, p_max=0.5)
    x = torch.randint(1000, 2000, (32, 20))
    x[:, 0] = 101
    x[:, 15:] = 0
    out = aug(x, torch.zeros(32, dtype=torch.long))
    assert not aug.reparameterized
    assert torch.equal(out.x[:, 15:], x[:, 15:])  # padding untouched
    assert torch.equal(out.x[:, 0], x[:, 0])  # special tokens untouched
    assert (out.x == 103).any()
    assert out.log_prob.shape == (32,)
    assert 0 < aug.p_drop.value.item() < 0.5


def test_token_dropout_quadratic_prior_penalty():
    aug = TokenDropout(mask_token_id=103, init_p=0.1, p_max=0.5, mode="map")
    assert aug.quadratic_prior_penalty(0.1, 0.1).item() == pytest.approx(0.0, abs=1e-6)
    penalty = aug.quadratic_prior_penalty(0.3, 0.1)
    assert penalty.item() == pytest.approx(0.5 * 2.0**2, rel=1e-5)
    penalty.backward()
    assert aug.p_drop.loc.grad.item() < 0  # pulled up toward the prior mean


def test_token_dropout_rejects_float_inputs():
    with pytest.raises(ValueError):
        TokenDropout(mask_token_id=1)(torch.randn(2, 3), torch.zeros(2))
