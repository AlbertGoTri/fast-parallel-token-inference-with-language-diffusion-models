"""SDTT (Self-Distillation Through Time) distillation strategy.

Same reduction schedule (halving), midpoint target step, KL loss, and teacher promotion as
progressive halving -- the ONLY difference is the training target: instead of the teacher's
instantaneous logits at the midpoint, SDTT uses the teacher's ``K``-step-ahead denoising
distribution. See Deschenaux & Gulcehre, "Beyond Autoregression: Fast LLMs via
Self-Distillation Through Time" (ICLR 2025).

Subclassing ProgressiveHalvingStrategy makes the shared behavior explicit: SDTT reuses the
schedule and loss, so a run differs from progressive halving only in build_target -- the
cleanest possible ablation.
"""

from .progressive_halving import ProgressiveHalvingStrategy
from ..generate_cache import generate_sdtt_target


class SDTTStrategy(ProgressiveHalvingStrategy):
    name = "sdtt"

    # Teacher denoising steps compressed into a single student step. With the halving
    # schedule the student runs at half the teacher's steps, so K=2 is the natural setting.
    K = 2

    def build_target(self, model, input_ids, *, teacher_steps, target_step, gen_length, block_length):
        # target_step is the midpoint (inherited cache_target_step) -- the same partially
        # masked state progressive halving snapshots; SDTT rolls K steps forward from there.
        return generate_sdtt_target(
            model,
            input_ids,
            steps=teacher_steps,
            gen_length=gen_length,
            block_length=block_length,
            start_step=target_step,
            k=self.K,
        )
