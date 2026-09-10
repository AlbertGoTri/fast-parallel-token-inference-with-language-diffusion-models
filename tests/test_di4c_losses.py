"""Di4C distillation loss — KL(teacher || student), per-position."""

import pytest
import torch

from LLaDA.di4c.losses import distillation_loss, mixture_log_prob, consistency_loss


def test_zero_when_identical():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 16)
    assert float(distillation_loss(logits, logits.clone())) == pytest.approx(0.0, abs=1e-5)


def test_positive_when_different():
    torch.manual_seed(1)
    student = torch.randn(2, 4, 16)
    teacher = torch.randn(2, 4, 16)
    assert float(distillation_loss(student, teacher)) > 0.0


def test_full_mask_equals_no_mask():
    torch.manual_seed(2)
    student = torch.randn(2, 4, 8)
    teacher = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    assert float(distillation_loss(student, teacher, mask=mask)) == pytest.approx(
        float(distillation_loss(student, teacher))
    )


def test_direction_is_teacher_to_student():
    # KL is asymmetric; KL(teacher||student) != KL(student||teacher) in general.
    torch.manual_seed(3)
    student = torch.randn(1, 3, 8)
    teacher = torch.randn(1, 3, 8)
    forward = float(distillation_loss(student, teacher))
    reverse = float(distillation_loss(teacher, student))
    assert forward != pytest.approx(reverse)


# --- mixture_log_prob ---

def test_mixture_log_prob_single_component_is_factorized_sum():
    torch.manual_seed(0)
    logits = torch.randn(1, 2, 3, 5)                 # N=1, B=2, L=3, V=5
    tokens = torch.randint(0, 5, (2, 3))
    got = mixture_log_prob(logits, tokens)
    logp = torch.log_softmax(logits[0], dim=-1)
    manual = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).sum(dim=-1)   # [B]
    assert got.shape == (2,)
    assert torch.allclose(got, manual, atol=1e-5)


def test_mixture_of_identical_components_equals_single():
    torch.manual_seed(1)
    single = torch.randn(1, 2, 3, 5)
    doubled = torch.cat([single, single.clone()], dim=0)                     # N=2, identical
    tokens = torch.randint(0, 5, (2, 3))
    assert torch.allclose(mixture_log_prob(doubled, tokens),
                          mixture_log_prob(single, tokens), atol=1e-5)


def test_mixture_mask_excludes_positions():
    torch.manual_seed(2)
    logits = torch.randn(1, 1, 4, 5)
    tokens = torch.randint(0, 5, (1, 4))
    mask = torch.tensor([[True, True, False, False]])
    got = mixture_log_prob(logits, tokens, mask=mask)
    logp = torch.log_softmax(logits[0], dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    manual = (logp * mask).sum(dim=-1)
    assert torch.allclose(got, manual, atol=1e-5)


# --- consistency_loss ---

def test_consistency_loss_backprops_into_direct_student():
    torch.manual_seed(3)
    direct = torch.randn(2, 1, 3, 5, requires_grad=True)
    target = torch.randint(0, 5, (1, 3))
    loss = consistency_loss(direct, target)
    loss.backward()
    assert direct.grad is not None and torch.isfinite(direct.grad).all()


def test_consistency_loss_small_when_direct_matches_target():
    target = torch.randint(0, 5, (1, 3))
    direct = torch.full((1, 1, 3, 5), -10.0)
    for pos in range(3):
        direct[0, 0, pos, target[0, pos]] = 10.0     # strongly favor the target tokens
    assert float(consistency_loss(direct, target)) < 0.1


def test_control_variate_centers_value_but_keeps_gradient():
    torch.manual_seed(5)
    direct = torch.randn(2, 1, 3, 5, requires_grad=True)
    target = torch.randint(0, 5, (1, 3))
    loss = consistency_loss(direct, target, baseline_lambda_logits=direct)
    assert abs(float(loss)) < 1e-4                    # -logp + logp.detach() == 0 in value
    loss.backward()
    assert direct.grad is not None                    # gradient still flows via -logp
