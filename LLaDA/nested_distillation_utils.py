"""Utility functions for nested distillation pipeline."""

import os
import sys
import json
import gc
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field


class StageTimings:
    """Wall-clock stage durations and per-prompt latency aggregates.

    Timings are wall-clock, not GPU-kernel-only, because the bottleneck is often
    dataset loading and Ollama judge latency.
    """

    def __init__(self):
        self.cache_s: float = 0.0
        self.train_s: float = 0.0
        self.eval_s: float = 0.0
        self.total_pipeline_s: float = 0.0

        self.avg_generation_ms: float = 0.0
        self.median_generation_ms: float = 0.0
        self.p95_generation_ms: float = 0.0
        self.min_generation_ms: float = 0.0
        self.max_generation_ms: float = 0.0
        self.num_prompts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_s": round(self.cache_s, 2),
            "train_s": round(self.train_s, 2),
            "eval_s": round(self.eval_s, 2),
            "total_pipeline_s": round(self.total_pipeline_s, 2),
            "avg_generation_ms": round(self.avg_generation_ms, 2),
            "median_generation_ms": round(self.median_generation_ms, 2),
            "p95_generation_ms": round(self.p95_generation_ms, 2),
            "min_generation_ms": round(self.min_generation_ms, 2),
            "max_generation_ms": round(self.max_generation_ms, 2),
            "num_prompts": self.num_prompts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageTimings":
        t = cls()
        for key, value in data.items():
            if hasattr(t, key):
                setattr(t, key, value)
        return t
import torch
import psutil


def log_memory_usage(stage: str = "") -> Dict[str, float]:
    """Log current CUDA memory usage."""
    if not torch.cuda.is_available():
        return {"available": False}

    # Allocated = tensors + caches; Reserved = CUDA pool; Max peaks help detect fragmentation.
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3

    prefix = f"[{stage}] " if stage else ""
    print(f"{prefix}CUDA Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "available": True
    }


def cleanup_gpu_memory() -> None:
    """Explicitly free GPU memory by deleting references and clearing cache."""
    if not torch.cuda.is_available():
        return

    # Python may hold references to GPU tensors in exception tracebacks; gc clears them.
    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(0)

    print("GPU memory cleanup completed")


def verify_gpu_empty() -> bool:
    """Verify that no model objects remain on GPU."""
    if not torch.cuda.is_available():
        return True

    gc.collect()
    torch.cuda.empty_cache()

    allocated = torch.cuda.memory_allocated(0)
    # 100 MB is a pragmatic threshold: PyTorch's caching allocator usually holds a
    # small pool even after empty_cache.
    threshold = 100 * 1024 * 1024

    if allocated > threshold:
        print(f"WARNING: GPU memory still allocated: {allocated / 1024**3:.2f}GB")
        print("There may be model objects still on GPU!")
        return False

    print(f"GPU verification passed: {allocated / 1024**3:.3f}GB allocated (below threshold)")
    return True


def unload_model(model: Any, name: str = "model") -> None:
    """Safely unload a model from GPU and delete references."""
    if model is None:
        return

    print(f"Unloading {name} from GPU...")

    try:
        # Explicit CPU migration is safer than del alone because CUDA memory is not
        # freed until the tensor leaves the GPU context.
        if hasattr(model, 'cpu'):
            model.cpu()
    except Exception as e:
        print(f"Warning: Could not move {name} to CPU: {e}")

    del model

    cleanup_gpu_memory()

    print(f"{name} unloaded successfully")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Determinism slows convolutions but makes distillation runs bitwise
        # reproducible for debugging.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        import yaml
    except ImportError:
        # Auto-install is a convenience for fresh academic cluster nodes where
        # dependencies are not pre-installed.
        print("PyYAML not installed. Installing...")
        os.system("pip install pyyaml -q")
        import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save data as JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_csv(rows: list, path: str, headers: Optional[list] = None) -> None:
    """Save data as CSV file."""
    import csv

    ensure_dir(os.path.dirname(path))

    with open(path, 'w', newline='', encoding='utf-8') as f:
        if rows and isinstance(rows[0], dict):
            if headers is None:
                headers = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(f)
            if headers:
                writer.writerow(headers)
            writer.writerows(rows)


def format_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().isoformat()


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO format timestamp."""
    return datetime.fromisoformat(ts)


@dataclass
class RoundResult:
    """Result record for a single distillation round."""
    round_number: int
    student_name: str
    teacher_name: str
    student_steps: int
    teacher_steps: int
    promptfoo_percent: float
    perplexity: float
    passed: bool
    timestamp: str
    cache_dir: str
    checkpoint_dir: str
    eval_dir: str
    latency_seconds: float = 0.0
    previous_latency: float = 0.0
    stage_timings: StageTimings = field(default_factory=StageTimings)
    per_prompt_latencies: list = field(default_factory=list, repr=False)
    previous_avg_generation_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['stage_timings'] = self.stage_timings.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoundResult':
        # Separate per_prompt_latencies if present to avoid kwarg conflict
        pd = data.pop('per_prompt_latencies', []) if isinstance(data, dict) else []
        timings = StageTimings.from_dict(data.pop('stage_timings', {})) if isinstance(data, dict) else StageTimings()
        obj = cls(**data, per_prompt_latencies=pd, stage_timings=timings)
        return obj

    def format_summary(self) -> str:
        """Format a one-line summary with assertion-level Promptfoo score and speedup."""
        st = self.stage_timings
        avg_ms = st.avg_generation_ms
        med_ms = st.median_generation_ms
        p95_ms = st.p95_generation_ms
        num = st.num_prompts

        speedup_avg = ""
        speedup_med = ""
        if self.previous_avg_generation_ms > 0 and avg_ms > 0:
            s_avg = self.previous_avg_generation_ms / avg_ms
            speedup_avg = f", avg_speedup={s_avg:.2f}x"
        if self.previous_avg_generation_ms > 0 and med_ms > 0:
            s_med = self.previous_avg_generation_ms / med_ms
            speedup_med = f", med_speedup={s_med:.2f}x"

        return (
            f"Student {self.student_steps} steps: promptfoo assertion {self.promptfoo_percent:.1f}%, "
            f"perplexity {self.perplexity:.1f}, "
            f"avg_gen={avg_ms:.0f}ms, med_gen={med_ms:.0f}ms, p95={p95_ms:.0f}ms, n={num}"
            f"{speedup_avg}{speedup_med}"
        )


@dataclass
class ExperimentState:
    """Complete state of the experiment for resume support."""
    experiment_name: str
    current_round: int
    current_teacher_path: str
    current_teacher_steps: int
    completed_rounds: list
    is_running: bool
    last_updated: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentState':
        return cls(**data)


class StateManager:
    """Manages experiment state for resume support."""

    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state: Optional[ExperimentState] = None

    def load(self) -> Optional[ExperimentState]:
        """Load state from file if it exists."""
        if os.path.exists(self.state_file):
            try:
                data = load_json(self.state_file)
                self.state = ExperimentState.from_dict(data)
                return self.state
            except Exception as e:
                print(f"Warning: Could not load state file: {e}")
                return None
        return None

    def save(self, state: ExperimentState) -> None:
        """Save state to file."""
        ensure_dir(os.path.dirname(self.state_file))
        state.last_updated = format_timestamp()
        save_json(state.to_dict(), self.state_file)
        self.state = state

    def initialize(self, experiment_name: str, initial_teacher_path: str,
                   initial_teacher_steps: int) -> ExperimentState:
        """Initialize new experiment state."""
        self.state = ExperimentState(
            experiment_name=experiment_name,
            current_round=0,
            current_teacher_path=initial_teacher_path,
            current_teacher_steps=initial_teacher_steps,
            completed_rounds=[],
            is_running=True,
            last_updated=format_timestamp()
        )
        self.save(self.state)
        return self.state


class ProgressLogger:
    """Logs progress with stage information."""

    def __init__(self, total_rounds: int):
        self.total_rounds = total_rounds
        self.current_round = 0
        self.current_stage = ""

    def start_round(self, round_num: int, steps: int) -> None:
        """Log start of a new round."""
        self.current_round = round_num
        print("\n" + "=" * 70)
        print(f"ROUND {round_num}/{self.total_rounds} | Student Steps: {steps}")
        print("=" * 70)

    def start_stage(self, stage_name: str) -> None:
        """Log start of a stage."""
        self.current_stage = stage_name
        print(f"\n>>> [{stage_name}] Starting...")

    def end_stage(self, stage_name: str, success: bool = True) -> None:
        """Log end of a stage."""
        status = "COMPLETED" if success else "FAILED"
        print(f">>> [{stage_name}] {status}")

    def log(self, message: str) -> None:
        """Log a message with current context."""
        prefix = f"[Round {self.current_round}]"
        if self.current_stage:
            prefix += f"[{self.current_stage}]"
        print(f"{prefix} {message}")


def print_leaderboard(results: list) -> None:
    """Print a formatted leaderboard to console."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 70)
    print("LEADERBOARD")
    print("=" * 70)

    for r in results:
        if isinstance(r, dict):
            r = RoundResult.from_dict(r)
        print(f"  {r.format_summary()}")

    print("=" * 70 + "\n")


def validate_step_reduction(teacher_steps: int, student_steps: int,
                           expected_decrement: int = 1) -> bool:
    """Validate that student uses exactly one fewer step than teacher."""
    expected = teacher_steps - expected_decrement
    if student_steps != expected:
        raise ValueError(
            f"Step reduction validation failed: "
            f"Teacher={teacher_steps}, Student={student_steps}, "
            f"Expected={expected}"
        )
    return True


def run_command(cmd: str, cwd: Optional[str] = None, check: bool = True) -> Tuple[int, str, str]:
    """Run a shell command and capture output."""
    import subprocess

    print(f"Running: {cmd}")

    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")

    return result.returncode, result.stdout, result.stderr


def compute_latency_aggregates(latencies: list[Dict[str, Any]]) -> "StageTimings":
    """Compute aggregate timing statistics from per-prompt timing dicts.

    Each dict should contain at least a 'generation_ms' key (float, milliseconds).
    """
    t = StageTimings()
    # Filter out non-dict entries because malformed JSONL lines can appear when
    # the server crashes mid-write.
    generation_times = [float(item.get("generation_ms", 0)) for item in latencies if isinstance(item, dict)]
    n = len(generation_times)
    t.num_prompts = n
    if n == 0:
        return t

    total = sum(generation_times)
    avg = total / n
    t.avg_generation_ms = avg

    sorted_g = sorted(generation_times)
    t.min_generation_ms = sorted_g[0]
    t.max_generation_ms = sorted_g[-1]

    # Median
    mid = n // 2
    if n % 2 == 0:
        t.median_generation_ms = (sorted_g[mid - 1] + sorted_g[mid]) / 2.0
    else:
        t.median_generation_ms = sorted_g[mid]

    # P95: linear interpolation between closest ranks avoids bias from discrete sample sizes.
    idx = 0.95 * (n - 1)
    low = int(idx)
    high = low + 1
    frac = idx - low
    if high < n:
        t.p95_generation_ms = sorted_g[low] + frac * (sorted_g[high] - sorted_g[low])
    else:
        t.p95_generation_ms = sorted_g[-1]

    return t


def check_ollama_running() -> bool:
    """Check if Ollama server is running for promptfoo evaluation."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/tags",
            method='GET'
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, Exception):
        return False
