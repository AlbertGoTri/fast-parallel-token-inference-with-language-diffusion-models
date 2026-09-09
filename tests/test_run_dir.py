"""Run-directory resolution: new runs are tagged with the strategy name; resume/override
paths are unaffected."""

import os
from types import SimpleNamespace

from nested_distillation import _resolve_run_dir


def _args(**kw):
    base = dict(run_dir=None, resume=False, status=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_new_run_dir_includes_strategy_name(tmp_path):
    out = _resolve_run_dir(str(tmp_path), _args(), strategy_name="sdtt")
    name = os.path.basename(out)
    assert name.startswith("run_sdtt_")   # run_<strategy>_<timestamp>
    assert os.path.isdir(out)


def test_new_run_dir_without_strategy_is_plain(tmp_path):
    out = _resolve_run_dir(str(tmp_path), _args(), strategy_name=None)
    name = os.path.basename(out)
    assert name.startswith("run_")
    assert "sdtt" not in name


def test_strategy_name_is_sanitized(tmp_path):
    out = _resolve_run_dir(str(tmp_path), _args(), strategy_name="odd/name")
    name = os.path.basename(out)
    assert "/" not in name and "\\" not in name
    assert name.startswith("run_odd_name_")


def test_run_dir_override_ignores_strategy(tmp_path):
    override = str(tmp_path / "custom")
    out = _resolve_run_dir(str(tmp_path), _args(run_dir=override), strategy_name="sdtt")
    assert os.path.abspath(out) == os.path.abspath(override)


def test_resume_returns_the_previous_run(tmp_path):
    # A fresh run writes the latest-run pointer; --resume must return that same dir.
    first = _resolve_run_dir(str(tmp_path), _args(), strategy_name="sdtt")
    resumed = _resolve_run_dir(str(tmp_path), _args(resume=True), strategy_name="sdtt")
    assert os.path.abspath(resumed) == os.path.abspath(first)
