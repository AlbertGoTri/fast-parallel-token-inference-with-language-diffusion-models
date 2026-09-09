"""Distillation strategy: registry, schedule hooks, teacher promotion, loss."""

from types import SimpleNamespace

import pytest
import torch

from LLaDA.distill import get_strategy, ProgressiveHalvingStrategy, SDTTStrategy, DUOStrategy


def test_registry_returns_progressive_halving():
    assert isinstance(get_strategy("progressive_halving"), ProgressiveHalvingStrategy)


def test_halve_alias_maps_to_progressive_halving():
    s = get_strategy("halve")
    assert isinstance(s, ProgressiveHalvingStrategy)
    assert s.name == "progressive_halving"


def test_unknown_strategy_raises_valueerror():
    with pytest.raises(ValueError):
        get_strategy("does-not-exist")


@pytest.mark.parametrize("teacher,expected", [(128, 64), (64, 32), (2, 1), (1, 1)])
def test_next_student_steps_halves(teacher, expected):
    assert ProgressiveHalvingStrategy().next_student_steps(teacher, 1) == expected


def test_next_student_steps_respects_floor():
    # max(min_steps, teacher // 2): floor wins when halving would go below it.
    assert ProgressiveHalvingStrategy().next_student_steps(4, 4) == 4


@pytest.mark.parametrize("teacher,expected", [(128, 64), (64, 32), (2, 1)])
def test_cache_target_step_is_midpoint(teacher, expected):
    assert ProgressiveHalvingStrategy().cache_target_step(teacher) == expected


def test_promote_teacher_uses_student_checkpoint_and_steps():
    result = SimpleNamespace(checkpoint_dir="/ckpt/round3", student_steps=16)
    ts = ProgressiveHalvingStrategy().promote_teacher(result)
    assert ts.path == "/ckpt/round3"
    assert ts.steps == 16


def test_compute_loss_zero_for_identical_logits():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 32)
    loss = ProgressiveHalvingStrategy().compute_loss(logits, logits.clone(), temperature=2.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-4)


def test_compute_loss_positive_for_different_logits():
    torch.manual_seed(1)
    student = torch.randn(1, 5, 32)
    target = torch.randn(1, 5, 32)
    loss = ProgressiveHalvingStrategy().compute_loss(student, target, temperature=2.0)
    assert float(loss) > 0.0


def test_sdtt_registered_and_named():
    s = get_strategy("sdtt")
    assert isinstance(s, SDTTStrategy)
    assert s.name == "sdtt"


@pytest.mark.parametrize("teacher", [128, 64, 8, 2])
def test_sdtt_shares_schedule_with_progressive(teacher):
    # SDTT differs only in build_target; schedule/target-step must match progressive halving
    # so the two methods are directly comparable.
    sdtt, prog = get_strategy("sdtt"), get_strategy("progressive_halving")
    assert sdtt.next_student_steps(teacher, 1) == prog.next_student_steps(teacher, 1)
    assert sdtt.cache_target_step(teacher) == prog.cache_target_step(teacher)


def test_sdtt_shares_kl_loss_with_progressive():
    torch.manual_seed(0)
    student = torch.randn(1, 4, 16)
    target = torch.randn(1, 4, 16)
    sdtt_loss = float(get_strategy("sdtt").compute_loss(student, target, temperature=2.0))
    prog_loss = float(get_strategy("progressive_halving").compute_loss(student, target, temperature=2.0))
    assert sdtt_loss == pytest.approx(prog_loss)


def test_duo_registered_and_named():
    s = get_strategy("duo")
    assert isinstance(s, DUOStrategy)
    assert s.name == "duo"


@pytest.mark.parametrize("teacher", [128, 64, 8, 2])
def test_duo_shares_schedule_with_progressive(teacher):
    duo, prog = get_strategy("duo"), get_strategy("progressive_halving")
    assert duo.next_student_steps(teacher, 1) == prog.next_student_steps(teacher, 1)
    assert duo.cache_target_step(teacher) == prog.cache_target_step(teacher)


def test_duo_shares_kl_loss_with_progressive():
    torch.manual_seed(0)
    student = torch.randn(1, 4, 16)
    target = torch.randn(1, 4, 16)
    duo_loss = float(get_strategy("duo").compute_loss(student, target, temperature=2.0))
    prog_loss = float(get_strategy("progressive_halving").compute_loss(student, target, temperature=2.0))
    assert duo_loss == pytest.approx(prog_loss)
