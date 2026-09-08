"""Cache save/load round-trip (the pickle+gzip path used on Windows)."""

import torch

from nested_distillation import _save_cache_object, _load_cache_object


def test_roundtrip_preserves_tensors_and_scalars(tmp_path):
    path = str(tmp_path / "batch_0.pkl.gz")
    payload = {
        "input_x": torch.arange(6).reshape(2, 3),
        "target_logits": torch.randn(2, 3, 4),
        "attn_mask": None,
        "prompt_len": 7,
        "target_step": 3,
    }
    _save_cache_object(payload, path)
    loaded = _load_cache_object(path)

    assert torch.equal(loaded["input_x"], payload["input_x"])
    assert loaded["input_x"].dtype == payload["input_x"].dtype
    assert torch.allclose(loaded["target_logits"], payload["target_logits"])
    assert loaded["attn_mask"] is None
    assert loaded["prompt_len"] == 7
    assert loaded["target_step"] == 3


def test_roundtrip_preserves_attn_mask_tensor(tmp_path):
    path = str(tmp_path / "batch_1.pkl.gz")
    mask = torch.ones(1, 8, dtype=torch.long)
    _save_cache_object({"attn_mask": mask, "prompt_len": 4}, path)
    loaded = _load_cache_object(path)
    assert torch.equal(loaded["attn_mask"], mask)
