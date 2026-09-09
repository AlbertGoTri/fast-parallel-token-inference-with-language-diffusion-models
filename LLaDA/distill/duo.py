"""DUO-inspired consistency distillation strategy (adapted to masked diffusion).

Same reduction schedule (halving), midpoint target step, KL loss, and teacher promotion as
progressive halving and SDTT -- the ONLY difference is the training target's horizon. Where
SDTT rolls the teacher 2 steps ahead, DUO rolls it all the way to x0, so the student learns to
jump straight from a partially-masked state to the teacher's fully-denoised output (the
consistency property f(x_t) ~= x0 for all t).

Inspired by "The Diffusion Duality" (Sahoo et al., ICML 2025), whose Discrete Consistency
Distillation is defined for uniform-state diffusion. LLaDA is a masked/absorbing-state model,
so this is the consistency *idea* adapted to the existing supervised-KL pipeline -- not the
literal uniform-state algorithm (no EMA/target network or boundary-condition CD loop).
"""

from .progressive_halving import ProgressiveHalvingStrategy
from ..generate_cache import generate_rollout_target


class DUOStrategy(ProgressiveHalvingStrategy):
    name = "duo"

    def build_target(self, model, input_ids, *, teacher_steps, target_step, gen_length, block_length):
        # k=None rolls the teacher to the end of the trajectory (x0): the consistency target.
        return generate_rollout_target(
            model,
            input_ids,
            steps=teacher_steps,
            gen_length=gen_length,
            block_length=block_length,
            start_step=target_step,
            k=None,
        )
