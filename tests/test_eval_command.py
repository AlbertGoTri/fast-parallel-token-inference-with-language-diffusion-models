"""Promptfoo launcher resolution (prefers the pinned local install)."""

import os

import LLaDA.nested_distillation_eval as ev
from LLaDA.nested_distillation_eval import _promptfoo_command, PROMPTFOO_VERSION


def test_version_is_pinned():
    assert PROMPTFOO_VERSION == "0.121.15"


def test_command_is_nonempty_and_mentions_promptfoo():
    cmd = _promptfoo_command()
    assert isinstance(cmd, str) and cmd
    assert "promptfoo" in cmd


def test_command_prefers_local_install_else_pinned_npx():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(ev.__file__)))
    local = os.path.join(repo_root, "node_modules", ".bin", "promptfoo")
    cmd = _promptfoo_command()
    if os.path.exists(local + ".cmd") or os.path.exists(local):
        assert "node_modules" in cmd
    else:
        # Fallback must still be version-pinned, never bare `npx promptfoo`.
        assert f"promptfoo@{PROMPTFOO_VERSION}" in cmd
