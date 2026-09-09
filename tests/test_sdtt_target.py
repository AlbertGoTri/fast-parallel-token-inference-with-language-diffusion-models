"""SDTT target construction (generate_sdtt_target) — assembly logic, no GPU/8B needed.

Uses a fake model whose logits are constant per call (so the produced target can be traced
back to the model call that generated it) with a tiny position-dependent bump so the argmax
token and the confidence ordering are deterministic (higher position -> higher confidence,
so the highest-index masked position is denoised first).
"""

from types import SimpleNamespace

import torch

from LLaDA.generate_cache import generate_sdtt_target

MASK_ID = 7
VOCAB = 8


class _FakeModel:
    def __init__(self, vocab=VOCAB, eps=1e-3, device="cpu"):
        self.vocab = vocab
        self.eps = eps
        self.device = device
        self.calls = 0

    def __call__(self, x, attention_mask=None):
        c = float(self.calls)
        self.calls += 1
        B, L = x.shape
        logits = torch.full((B, L, self.vocab), c, dtype=torch.float32)
        pos = torch.arange(L, dtype=torch.float32)
        logits[:, :, 0] = c + self.eps * pos  # channel 0 wins argmax; confidence grows with pos
        return SimpleNamespace(logits=logits)


def _prompt():
    return torch.tensor([[1, 2]], dtype=torch.long)  # P=2, tokens != MASK_ID


def test_target_shapes_and_contract():
    x_t, target, attn = generate_sdtt_target(
        _FakeModel(), _prompt(), attention_mask=None,
        steps=6, gen_length=6, block_length=6, start_step=3, k=2, mask_id=MASK_ID,
    )
    assert x_t.shape == (1, 8)          # P + gen_length
    assert target.shape == (1, 8, VOCAB)
    assert target.dtype == torch.float16
    assert attn is None


def test_midpoint_state_is_partially_denoised():
    x_t, _, _ = generate_sdtt_target(
        _FakeModel(), _prompt(),
        steps=6, gen_length=6, block_length=6, start_step=3, k=2, mask_id=MASK_ID,
    )
    # 3 steps run before start_step, each denoising the highest-confidence masked position.
    assert (x_t[0, 2:5] == MASK_ID).all()   # positions 2,3,4 still masked at the midpoint
    assert (x_t[0, 5:8] != MASK_ID).all()   # positions 5,6,7 already denoised


def test_target_provenance_reflects_two_step_rollout():
    """Each position's target should come from the step it was denoised; still-masked
    positions from the last rollout step. Provenance = the model-call index, read from a
    non-argmax channel."""
    x_t, target, _ = generate_sdtt_target(
        _FakeModel(), _prompt(),
        steps=6, gen_length=6, block_length=6, start_step=3, k=2, mask_id=MASK_ID,
    )
    prov = target[0, :, 1].round().to(torch.int64)  # call index that produced each position

    # x_t is produced by call 3; the 2-step rollout uses calls 3 (r=0) then 4 (r=1).
    assert prov[0].item() == 3 and prov[1].item() == 3   # prompt/context -> base (x_t) logits
    assert prov[5].item() == 3                           # denoised before midpoint -> base
    assert prov[4].item() == 3                           # rollout r=0 uses the x_t logits
    assert prov[3].item() == 4                           # rollout r=1 -> 2-step-ahead target
    assert prov[2].item() == 4                           # still masked -> last rollout step

    # The whole point of SDTT: the target is NOT just the midpoint snapshot (all 3s).
    assert (prov == 4).any()


def test_k_is_clamped_near_trajectory_end():
    # start_step=5 leaves only one step in a 6-step trajectory; k=2 must clamp to 1.
    x_t, target, _ = generate_sdtt_target(
        _FakeModel(), _prompt(),
        steps=6, gen_length=6, block_length=6, start_step=5, k=2, mask_id=MASK_ID,
    )
    assert x_t.shape == (1, 8)
    assert target.shape == (1, 8, VOCAB)
    assert x_t[0, 2].item() == MASK_ID       # only position 2 remained masked at start_step
    assert (x_t[0, 3:8] != MASK_ID).all()


def test_attention_mask_is_extended_and_returned():
    attn_in = torch.ones((1, 2), dtype=torch.long)
    _, _, attn_out = generate_sdtt_target(
        _FakeModel(), _prompt(), attention_mask=attn_in,
        steps=6, gen_length=6, block_length=6, start_step=3, k=2, mask_id=MASK_ID,
    )
    assert attn_out is not None
    assert attn_out.shape == (1, 8)          # extended by gen_length
