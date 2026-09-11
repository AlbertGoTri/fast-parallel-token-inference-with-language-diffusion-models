"""Lambda-conditioned mixture student for Di4C.

Di4C makes the student a mixture over a latent lambda ~ Uniform[0,1]:
    p^theta(x_s | x_t) = E_lambda[ prod_d p^theta_d(x_s^d | x_t; lambda) ].
For each fixed lambda the model stays factorized (per-position), but averaging over lambda
introduces cross-dimensional correlations.

We inject lambda the way the paper injects it (like the timestep, through a small
*zero-initialized* sub-network) without touching the frozen base weights: a `LambdaConditioner`
maps the scalar lambda to a per-hidden-dim additive bias that is added to the token-embedding
output via a forward hook. Because the conditioner's last layer is zero-initialized, at init
the bias is exactly 0 -> the wrapped student is identical to the teacher for any lambda, and
lambda only starts to matter once the conditioner is trained. This keeps the base model and its
LoRA/serving behavior unchanged until Di4C training happens.
"""

import torch
import torch.nn as nn


class LambdaConditioner(nn.Module):
    """Maps a scalar lambda in [0,1] to a per-hidden-dim additive bias.

    The final linear layer is zero-initialized, so at init the output is 0 for any lambda.
    """

    def __init__(self, hidden_size: int, mlp_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_size),
        )
        # Zero-init the output layer: bias == 0 at init => student == teacher for any lambda.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, lam: torch.Tensor) -> torch.Tensor:
        # lam: [B] or [B,1] in [0,1] -> [B, hidden_size]
        lam = lam.reshape(-1, 1).to(self.net[0].weight.dtype)
        return self.net(lam)


def wrap_lambda_conditioned(model, mlp_dim: int = 128):
    """Attach a lambda-conditioner to `model` via a forward hook on its input embeddings.

    Returns (model, state). Set the active lambda with `set_lambda(state, lam)` before a
    forward pass; while `state["lam"]` is None the hook is a no-op, so existing (non-Di4C)
    forwards are unaffected. The conditioner is stored on the model as `._lambda_conditioner`
    (its parameters are the only new trainable weights besides LoRA).
    """
    hidden = model.config.hidden_size
    device = next(model.parameters()).device
    conditioner = LambdaConditioner(hidden, mlp_dim).to(device)
    state = {"lam": None}

    def _hook(module, inputs, output):
        lam = state["lam"]
        if lam is None:
            return output
        bias = conditioner(lam).to(output.dtype)      # [B, hidden]
        return output + bias[:, None, :]              # broadcast over sequence length

    embeddings = model.get_input_embeddings()
    handle = embeddings.register_forward_hook(_hook)

    model._lambda_conditioner = conditioner
    model._lambda_state = state
    model._lambda_hook_handle = handle
    return model, state


def set_lambda(state: dict, lam) -> None:
    """Set the active lambda for subsequent forward passes (None disables conditioning)."""
    state["lam"] = lam
