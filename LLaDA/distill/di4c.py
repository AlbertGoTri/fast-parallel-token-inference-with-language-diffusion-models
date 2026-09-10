"""Di4C strategy: mixture-model correlation distillation wired into the shared pipeline.

Di4C trains differently from the per-position strategies (its own loop with the lambda-mixture
student and correlation losses), so it sets ``custom_training = True``: the orchestrator's
run_single_round routes the round to ``train_round`` instead of cache_stage + train_stage. Every
other strategy leaves ``custom_training`` unset, so they take the unchanged standard path.

The reduction schedule, teacher promotion, and the eval are shared (inherited / strategy-
agnostic), so Di4C produces a run_di4c_* leaderboard in the same format as the others.
"""

from .progressive_halving import ProgressiveHalvingStrategy


class Di4CStrategy(ProgressiveHalvingStrategy):
    name = "di4c"

    #: Tells run_single_round to call train_round() instead of the cache -> per-position-KL path.
    custom_training = True

    def build_target(self, *args, **kwargs):
        raise NotImplementedError("Di4C uses custom_training; build_target is not called.")

    def train_round(self, config, checkpoint_dir, student_steps, logger=None):
        """Train a Di4C student for this round and save it to ``checkpoint_dir``. Returns bool."""
        # Lazy import keeps the torch-heavy training module out of strategy registration.
        from ..di4c.train import run_di4c_training

        d = config.get("di4c", {}) or {}
        return run_di4c_training(
            config,
            checkpoint_dir,
            n_lambda=int(d.get("n_lambda", 2)),
            consistency_weight=float(d.get("consistency_weight", 1.0)),
            lr=float(d.get("lr", config["student"]["learning_rate"])),
            mem_fraction=float(d.get("mem_fraction", 0.92)),
            logger=logger,
        )
