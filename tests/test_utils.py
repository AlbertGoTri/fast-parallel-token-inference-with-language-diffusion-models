"""Pure helpers: teacher-path resolution and latency aggregation."""

import json

from nested_distillation import resolve_teacher_model_paths
from LLaDA.nested_distillation_utils import compute_latency_aggregates


def test_plain_model_path_has_no_adapter():
    base, adapter = resolve_teacher_model_paths("GSAI-ML/LLaDA-8B-Instruct", "fallback/model")
    assert base == "GSAI-ML/LLaDA-8B-Instruct"
    assert adapter is None


def test_lora_checkpoint_resolves_base_from_config(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "base/model"})
    )
    base, adapter = resolve_teacher_model_paths(str(ckpt), "fallback/model")
    assert base == "base/model"
    assert adapter == str(ckpt)


def test_lora_checkpoint_without_base_falls_back(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(json.dumps({}))
    base, adapter = resolve_teacher_model_paths(str(ckpt), "fallback/model")
    assert base == "fallback/model"
    assert adapter == str(ckpt)


def test_latency_aggregates_empty_is_zeroed():
    t = compute_latency_aggregates([])
    assert t.num_prompts == 0
    assert t.avg_generation_ms == 0.0


def test_latency_aggregates_basic_stats():
    recs = [{"generation_ms": 10}, {"generation_ms": 20}, {"generation_ms": 30}]
    t = compute_latency_aggregates(recs)
    assert t.num_prompts == 3
    assert t.avg_generation_ms == 20.0
    assert t.median_generation_ms == 20.0
    assert t.min_generation_ms == 10.0
    assert t.max_generation_ms == 30.0


def test_latency_aggregates_filters_malformed_entries():
    recs = [{"generation_ms": 10}, "garbage-line", {"generation_ms": 20}]
    t = compute_latency_aggregates(recs)
    assert t.num_prompts == 2
    assert t.avg_generation_ms == 15.0
