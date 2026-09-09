"""Di4C distillation loss — KL(teacher || student), per-position."""

import pytest
import torch

from LLaDA.di4c.losses import distillation_loss


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
