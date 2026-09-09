import torch
import numpy as np
import torch.nn.functional as F

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def generate_and_cache_trajectory(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, target_step=0, mask_id=126336):
    """
    Truncated forward diffusion used to create distillation training pairs.
    Stops at target_step and returns the masked input plus teacher logits.
    """
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x, attention_mask=attention_mask).logits

            if i == target_step:
                # Capture the teacher's prediction at the midpoint; the student learns to recover
                # this distribution in fewer steps than the full diffusion trajectory.
                # Cast logits to float16 to halve cache size; KL divergence is robust to this precision loss.
                return x.clone(), logits.clone().to(torch.float16), attention_mask.clone() if attention_mask is not None else None

            logits_with_noise = add_gumbel_noise(logits, temperature=0.0)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x, None, None


def _select_transfer(x, logits, num_transfer, block_end, mask_id):
    """Pick which masked positions to denoise this step (top-k by teacher confidence).

    Mirrors the selection rule inside generate_and_cache_trajectory. ``num_transfer`` is the
    per-row token budget for this step (shape [B]). Returns (transfer_index[bool B,L], x0).
    """
    mask_index = (x == mask_id)
    x0 = torch.argmax(add_gumbel_noise(logits, temperature=0.0), dim=-1)
    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    x0_p[:, block_end:] = -np.inf
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer[j])
        transfer_index[j, select_index] = True
    return transfer_index, x0


@torch.no_grad()
def generate_rollout_target(model, prompt, attention_mask=None, steps=128, gen_length=128,
                            block_length=128, start_step=0, k=2, mask_id=126336):
    """Roll the teacher forward from a midpoint snapshot and assemble a per-position target.

    Runs the teacher denoising to ``start_step`` (the same partially-masked midpoint state
    progressive halving caches), snapshots it as the student input ``x_t``, then rolls the
    teacher forward and records, for each generated position, the teacher distribution at the
    step it was denoised. Any position still masked at the end takes the last rollout step's
    distribution.

    The rollout horizon ``k`` selects the distillation method:
      * ``k=2``    -> SDTT: match the teacher's 2-step-ahead distribution
                     (Deschenaux & Gulcehre, ICLR 2025).
      * ``k=None`` -> DUO / consistency: roll all the way to x0, so the student learns to jump
                     straight to the teacher's fully-denoised output (The Diffusion Duality,
                     Sahoo et al., ICML 2025, adapted to masked diffusion). No positions remain
                     masked, so the still-masked fallback is a no-op.

    Shape/return contract is identical to generate_and_cache_trajectory, so the caching and
    training stages and the KL loss consume it unchanged.

    NOTE: assumes ``start_step`` falls in the first block (true for the single-block configs
    used here); ``k`` is clamped so the rollout never runs past the block's step budget.
    """
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)],
            dim=-1,
        )

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            logits = model(x, attention_mask=attention_mask).logits

            if i == start_step:
                # Snapshot the student input, then roll the teacher k steps forward and record
                # the teacher distribution that denoises each position.
                x_t = x.clone()
                target_logits = logits.clone()          # base target: teacher distribution at x_t
                last_logits = logits
                # k=None -> roll to the end of the block (DUO/consistency); else k steps (SDTT).
                k_eff = (steps_per_block - i) if k is None else min(k, steps_per_block - i)
                for r in range(k_eff):
                    if r > 0:
                        logits = model(x, attention_mask=attention_mask).logits
                    transfer_index, x0 = _select_transfer(
                        x, logits, num_transfer_tokens[:, i + r], block_end, mask_id
                    )
                    # Teacher distribution that led to these tokens being denoised.
                    target_logits[transfer_index] = logits[transfer_index]
                    x[transfer_index] = x0[transfer_index]
                    last_logits = logits
                # Still-masked positions take the last rollout step's distribution.
                still_masked = (x == mask_id)
                target_logits[still_masked] = last_logits[still_masked]
                return (
                    x_t,
                    target_logits.to(torch.float16),
                    attention_mask.clone() if attention_mask is not None else None,
                )

            # Normal denoising step, advancing toward start_step.
            transfer_index, x0 = _select_transfer(
                x, logits, num_transfer_tokens[:, i], block_end, mask_id
            )
            x[transfer_index] = x0[transfer_index]

    return x, None, None


# Backward-compatible alias: SDTT is the k=2 rollout.
generate_sdtt_target = generate_rollout_target
