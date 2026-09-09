"""Pluggable distillation strategies for the nested-distillation pipeline.

Register a new method by adding its class to ``_STRATEGIES``; the orchestrator
selects one by name via ``get_strategy`` (config ``schedule.strategy`` or the
``--strategy`` CLI flag).
"""

from .base import DistillationStrategy, TeacherState
from .progressive_halving import ProgressiveHalvingStrategy
from .sdtt import SDTTStrategy
from .duo import DUOStrategy

# "halve" is the historical config value for the original behavior; keep it as an
# alias so existing config files select the same strategy they always have.
_STRATEGIES = {
    ProgressiveHalvingStrategy.name: ProgressiveHalvingStrategy,
    "halve": ProgressiveHalvingStrategy,
    SDTTStrategy.name: SDTTStrategy,
    DUOStrategy.name: DUOStrategy,
}


def get_strategy(name: str) -> DistillationStrategy:
    """Instantiate a distillation strategy by name."""
    try:
        return _STRATEGIES[name]()
    except KeyError:
        available = ", ".join(sorted(_STRATEGIES))
        raise ValueError(
            f"Unknown distillation strategy '{name}'. Available: {available}"
        )


__all__ = [
    "DistillationStrategy",
    "TeacherState",
    "ProgressiveHalvingStrategy",
    "SDTTStrategy",
    "DUOStrategy",
    "get_strategy",
]
