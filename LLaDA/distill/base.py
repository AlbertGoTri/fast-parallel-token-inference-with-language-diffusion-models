"""Distillation strategy interface.

A strategy encapsulates the parts of the nested-distillation pipeline that differ
between distillation methods (progressive step-halving, SDTT, ...): the step
schedule, the teacher target construction, the training loss, and how the teacher
is promoted between rounds. Everything else in the pipeline (server, evaluation,
state/resume, model loading) is strategy-agnostic and shared.

This module changes no pipeline behavior on its own; ``ProgressiveHalvingStrategy``
reproduces the original hard-coded logic exactly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class TeacherState:
    """The teacher handed to the next round."""
    path: str
    steps: int


class DistillationStrategy(ABC):
    """Pluggable distillation method.

    Subclasses own the four seams that differ between methods; the orchestrator in
    ``nested_distillation.py`` calls these instead of hard-coding the halving math,
    the midpoint cache target, or the KL loss.
    """

    #: Registry key and leaderboard label for this strategy.
    name: str = "base"

    @abstractmethod
    def next_student_steps(self, teacher_steps: int, min_steps: int) -> int:
        """Denoising-step budget for the student distilled from a ``teacher_steps`` teacher."""

    @abstractmethod
    def cache_target_step(self, teacher_steps: int) -> int:
        """Trajectory step at which the teacher's distribution is captured for caching."""

    @abstractmethod
    def build_target(
        self,
        model,
        input_ids: torch.Tensor,
        *,
        teacher_steps: int,
        target_step: int,
        gen_length: int,
        block_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Produce one ``(input_x, target_logits, attn_mask)`` distillation pair."""

    @abstractmethod
    def compute_loss(
        self,
        student_logits_gen: torch.Tensor,
        target_logits_gen: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        """Distillation loss over generated positions."""

    def promote_teacher(self, result) -> TeacherState:
        """Teacher for the next round. Default: the student just trained becomes teacher."""
        return TeacherState(path=result.checkpoint_dir, steps=result.student_steps)
