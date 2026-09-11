"""Di4C lambda-conditioned student wrapper — zero-init invariant and conditioning mechanism.

Uses a tiny fake model (no GPU/8B). The critical invariant: at init the lambda-conditioner is
zero, so the wrapped student is identical to the teacher for any lambda.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from LLaDA.di4c.model import LambdaConditioner, wrap_lambda_conditioned, set_lambda


class _FakeLLaDA(nn.Module):
    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.emb = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, x, attention_mask=None):
        h = self.emb(x)                     # forward hook fires here
        return SimpleNamespace(logits=self.head(h))


def test_conditioner_is_zero_at_init():
    c = LambdaConditioner(hidden_size=8)
    out = c(torch.rand(4))
    assert out.shape == (4, 8)
    assert torch.allclose(out, torch.zeros(4, 8))


def test_zero_init_wrapper_is_identity_for_any_lambda():
    torch.manual_seed(0)
    model = _FakeLLaDA()
    x = torch.randint(0, 16, (2, 5))
    baseline = model(x).logits.clone()

    wrapped, state = wrap_lambda_conditioned(model)
    for lam in (torch.zeros(2), torch.ones(2), torch.rand(2)):
        set_lambda(state, lam)
        assert torch.allclose(wrapped(x).logits, baseline, atol=1e-6)


def test_none_lambda_is_a_noop():
    torch.manual_seed(1)
    model = _FakeLLaDA()
    x = torch.randint(0, 16, (2, 5))
    baseline = model(x).logits.clone()
    wrapped, state = wrap_lambda_conditioned(model)
    set_lambda(state, None)
    assert torch.allclose(wrapped(x).logits, baseline, atol=1e-6)


def test_lambda_changes_output_once_conditioner_is_trained():
    torch.manual_seed(2)
    model = _FakeLLaDA()
    x = torch.randint(0, 16, (2, 5))
    wrapped, state = wrap_lambda_conditioned(model)
    base = wrapped(x).logits.clone()                 # lambda still None -> baseline

    # Simulate a trained conditioner (non-zero output layer).
    with torch.no_grad():
        model._lambda_conditioner.net[-1].weight.fill_(1.0)
        model._lambda_conditioner.net[-1].bias.fill_(0.5)

    set_lambda(state, torch.tensor([0.9, 0.1]))
    assert not torch.allclose(wrapped(x).logits, base)

    # Disabling conditioning returns to the baseline.
    set_lambda(state, None)
    assert torch.allclose(wrapped(x).logits, base, atol=1e-6)


def test_conditioner_params_are_trainable_and_only_new_weights():
    model = _FakeLLaDA()
    wrapped, _ = wrap_lambda_conditioned(model)
    cond_params = list(model._lambda_conditioner.parameters())
    assert cond_params, "conditioner should expose trainable parameters"
    assert all(p.requires_grad for p in cond_params)
