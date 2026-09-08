"""Reduction-schedule logic, including the resume regression.

The resume bug: max_rounds was recomputed from the reduced teacher step count
while start_round was offset by completed rounds, so a resumed run stopped short
and skipped the final rounds (e.g. the 1-step round). These tests lock the fixed
behavior: the full ladder is covered from any resume point.
"""

import pytest

from nested_distillation import calculate_max_rounds, resolve_max_rounds
from LLaDA.distill import get_strategy

STRAT = get_strategy("halve")
MIN = 1
# Student step counts for a full run starting from 128 steps.
FULL_LADDER = [64, 32, 16, 8, 4, 2, 1]


def _simulate(current_round, current_teacher_steps, is_resume):
    """Reproduce main()'s loop bounds and step threading; return (max_rounds, students)."""
    max_rounds = resolve_max_rounds(STRAT, current_round, current_teacher_steps, MIN)
    start_round = current_round + 1 if is_resume else 1
    teacher = current_teacher_steps
    students = []
    for _ in range(start_round, max_rounds + 1):
        student = STRAT.next_student_steps(teacher, MIN)
        if student < MIN:
            break
        students.append(student)
        teacher = student
    return max_rounds, students


@pytest.mark.parametrize("initial,expected", [
    (128, 7), (64, 6), (32, 5), (2, 1), (1, 0), (100, 6),  # 100->50->25->12->6->3->1
])
def test_calculate_max_rounds(initial, expected):
    assert calculate_max_rounds(STRAT, initial, MIN) == expected


def test_fresh_run_covers_full_ladder():
    max_rounds, students = _simulate(0, 128, is_resume=False)
    assert max_rounds == 7
    assert students == FULL_LADDER


def test_resolve_max_rounds_fresh_matches_calculate():
    # Fresh state: current_round=0, teacher=initial_steps.
    assert resolve_max_rounds(STRAT, 0, 128, MIN) == calculate_max_rounds(STRAT, 128, MIN)


@pytest.mark.parametrize("done", [1, 2, 3, 4, 5, 6])
def test_resume_completes_full_ladder_including_one_step(done):
    """Regression: resuming after any round must still reach the 1-step round."""
    teacher_at_resume = FULL_LADDER[done - 1]
    max_rounds, remaining = _simulate(done, teacher_at_resume, is_resume=True)
    covered = FULL_LADDER[:done] + remaining
    assert max_rounds == 7, "total ladder length must be preserved on resume"
    assert covered == FULL_LADDER, "resume must not truncate the ladder"
    assert remaining[-1] == 1, "resume must reach the final 1-step round"


def test_resume_after_last_round_adds_nothing():
    # Completed all 7 rounds (teacher now at 1 == min): no further rounds.
    max_rounds, remaining = _simulate(7, 1, is_resume=True)
    assert remaining == []
