"""Progressive step-halving distillation (the original pipeline behavior).

Each student is trained to reproduce the teacher's midpoint distribution while
running exactly half the teacher's denoising steps; the student then becomes the
next round's teacher, so the step budget halves every round down to ``min_steps``.

Every method here is a faithful extraction of logic that previously lived inline in
``nested_distillation.py`` / ``generate_cache.py`` -- behavior is unchanged.
"""

import torch

from .base import DistillationStrategy
from ..generate_cache import generate_and_cache_trajectory


class ProgressiveHalvingStrategy(DistillationStrategy):
    name = "progressive_halving"

    def next_student_steps(self, teacher_steps, min_steps):
        # Strict halving with a floor; core assumption of the nested schedule.
        return max(min_steps, teacher_steps // 2)

    def cache_target_step(self, teacher_steps):
        # Midpoint caching exposes the teacher when roughly 50% of tokens are still
        # masked; earlier steps are too noisy, later steps too easy.
        return teacher_steps // 2

    def build_target(self, model, input_ids, *, teacher_steps, target_step, gen_length, block_length):
        return generate_and_cache_trajectory(
            model,
            input_ids,
            steps=teacher_steps,
            gen_length=gen_length,
            block_length=block_length,
            target_step=target_step,
        )

    def compute_loss(self, student_logits_gen, target_logits_gen, *, temperature):
        # Scaling by T^2 preserves gradient magnitude across temperatures, as in
        # standard distillation (Hinton et al.).
        return torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits_gen / temperature, dim=-1),
            torch.nn.functional.softmax(target_logits_gen / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)
