"""Di4C: mixture-model correlation distillation for LLaDA (Hayakawa et al., ICML 2025).

Unlike halving/SDTT/DUO (which share a factorized per-position student and only change the
target), Di4C makes the student a *mixture* over a latent lambda so it can represent
cross-token correlations, and distills with correlation-aware losses. This package holds the
lambda-conditioned student wrapper and the Di4C losses; it is a parallel training path, not a
`DistillationStrategy.build_target` hook.

Phase 0 (feasibility gate) exposes: the lambda-conditioned wrapper (`model`) and the
distillation loss (`losses`). Consistency/auxiliary losses and the training loop come in later
phases.
"""

from .model import LambdaConditioner, wrap_lambda_conditioned, set_lambda
from .losses import distillation_loss

__all__ = [
    "LambdaConditioner",
    "wrap_lambda_conditioned",
    "set_lambda",
    "distillation_loss",
]
