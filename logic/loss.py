# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
import math

from flow_matching.path import MixtureDiscreteProbPath


class MixturePathGeneralizedKL(_Loss):
    r"""A generalized KL loss for discrete flow matching.
    A class that measures the generalized KL of a discrete flow model :math:`p_{1|t}` w.r.t. a probability path given by ``path``. Note: this class is assuming that the model is trained on the same path.

    For a model trained on a space :math:`\mathcal{S} = \mathcal{T}^d`, :math:`\mathcal{T} = [K] = \set{1,2,\ldots,K}`, the loss is given by

    .. math::
            \ell_i(x_1, x_t, t) = -\frac{\dot{\kappa}_t}{1-\kappa_t} \biggr[  p_{1|t}(x_t^i|x_t) -\delta_{x^i_1}(x_t^i) + (1-\delta_{x^i_1}(x_t^i))\left(\log p_{1|t}(x_1^i|x_t)\right)\biggr],

    where :math:`\kappa_t` is the scheduler associated with ``path``.

    Args:
        path (MixtureDiscreteProbPath): Probability path (x-prediction training).
        reduction (str, optional): Specify the reduction to apply to the output ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction is applied to the output, ``'mean'``: the output is reduced by mean over sequence elements, ``'sum'``: the output is reduced by sum over sequence elements. Defaults to 'mean'.
    """

    def __init__(self, path: MixtureDiscreteProbPath, reduction: str = "mean") -> None:
        super().__init__(None, None, reduction)
        self.path = path

    def forward(self, logits: Tensor, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        x_1_shape = x_1.shape
        dtype = logits.dtype
        device = logits.device

        # --- 1) compute stable log-probs in higher precision ---
        # cast logits to float32 (or float64) for stability only for loss path
        logits_for_loss = logits.to(torch.float32)
        log_p_1t = torch.log_softmax(logits_for_loss, dim=-1)         # shape (B, d, K)

        # small epsilons
        eps = 1e-12
        log_eps = math.log(eps)

        # --- 2) gather log-prob for x1 and clamp safely ---
        log_p_1t_x1 = torch.gather(log_p_1t, dim=-1, index=x_1.unsqueeze(-1))
        log_p_1t_x1 = log_p_1t_x1.view(*x_1_shape)                    # (B, d)

        # clamp log-probs for numerical safety (but keep them finite)
        log_p_1t_x1_clamped = torch.clamp(log_p_1t_x1, min=log_eps)

        # --- 3) gather probability at x_t, computed from clamped log-probs ---
        p_1t = torch.exp(torch.clamp(log_p_1t, min=log_eps))          # avoid exact zeros
        p_1t_xt = torch.gather(p_1t, dim=-1, index=x_t.unsqueeze(-1))
        p_1t_xt = p_1t_xt.view(*x_1_shape)

        # --- 4) compute scheduler/jump coefficient safely ---
        scheduler_output = self.path.scheduler(t)
        # ensure scheduler tensors are float32 and on same device
        alpha_t = scheduler_output.alpha_t.to(torch.float32).to(device)
        d_alpha_t = scheduler_output.d_alpha_t.to(torch.float32).to(device)

        denom = (1.0 - alpha_t).clamp(min=1e-6)                        # avoid division by tiny numbers
        jump_coefficient = (d_alpha_t / denom)[(...,) + (None,) * (x_1.dim() - 1)]
        jump_coefficient = jump_coefficient.repeat(1, *x_1_shape[1:]).to(torch.float32)

        # sanitize jump_coefficient (clip to a reasonable range)
        jump_coefficient = torch.clamp(jump_coefficient, min=-1e6, max=1e6)

        # --- 5) delta mask (same dtype as the loss computations) ---
        delta_x1_xt = (x_t == x_1).to(log_p_1t_x1_clamped.dtype)

        # --- 6) IMPORTANT: avoid 0 * -inf by using torch.where ---
        # if delta==1 -> we want to *ignore* log_p_1t_x1 term (use 0)
        safe_log_term = torch.where(
            delta_x1_xt.bool(),
            torch.zeros_like(log_p_1t_x1_clamped),
            log_p_1t_x1_clamped,   # only used when delta==0
        )

        # --- 7) final loss (compute in float32) ---
        loss = -jump_coefficient * (p_1t_xt - delta_x1_xt + (1.0 - delta_x1_xt) * safe_log_term)

        # optional: convert back to original dtype for reduction if needed
        if self.reduction == "mean":
            return torch.mean(loss.to(dtype))
        elif self.reduction == "sum":
            return torch.sum(loss.to(dtype))
        elif self.reduction == "none":
            return loss.to(dtype)
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")