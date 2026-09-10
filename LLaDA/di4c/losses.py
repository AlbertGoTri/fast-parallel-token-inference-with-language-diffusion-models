"""Di4C loss functions.

- ``distillation_loss``: KL(teacher || student) single-step (Di4C Eq. 3).
- ``mixture_log_prob``: log-prob of a sampled sequence under the mixture-over-lambda of
  factorized student distributions (the log-sum-exp mixture the paper uses).
- ``consistency_loss``: the correlation-learning core (Di4C Eq. 4) -- trains the direct
  student's mixture to match samples drawn from the composed teacher-then-student path, with
  the composed path stop-gradiented and an optional factorized control variate.

The consistency loss here is faithful in structure (mixture + composition + stop-grad + control
variate); the exact estimator constants/decomposition follow the paper's intent and may need
reconciling against its appendix. Hyperparameters (lambda samples, loss weights) are tuning.
"""

import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, mask=None):
    """KL(teacher || student) over generated positions, per-position factorized.

    Matches the student's single-step denoising distribution to the teacher's at a given noise
    level (Di4C Eq. 3). Direction is KL(teacher || student): the teacher is the reference.

    Args:
        student_logits, teacher_logits: [B, L, V] logits.
        mask: optional [B, L] bool selecting positions to score (e.g. generated/masked
              positions). If None, all positions are averaged.
    """
    logp_student = F.log_softmax(student_logits, dim=-1)
    logp_teacher = F.log_softmax(teacher_logits, dim=-1)
    p_teacher = logp_teacher.exp()
    kl = (p_teacher * (logp_teacher - logp_student)).sum(dim=-1)   # [B, L]
    if mask is not None:
        kl = kl[mask]
    return kl.mean()


def mixture_log_prob(lambda_logits, tokens, mask=None):
    """Log-prob of ``tokens`` under a mixture over lambda of factorized distributions.

    p(x | ...) = mean_lambda prod_d p_d(x_d | ...; lambda), so
    log p(x) = logsumexp_lambda( sum_d log p_d(x_d | lambda) ) - log(N).

    Args:
        lambda_logits: [N, B, L, V] logits for N sampled lambdas.
        tokens: [B, L] token ids (a sampled sequence).
        mask: optional [B, L] bool; only these positions enter the product (others contribute
              log-prob 0, i.e. probability 1).
    Returns: [B] log-probs.
    """
    logp = F.log_softmax(lambda_logits, dim=-1)                                  # [N,B,L,V]
    idx = tokens.unsqueeze(0).unsqueeze(-1).expand(logp.shape[0], -1, -1, 1)     # [N,B,L,1]
    logp_tok = torch.gather(logp, dim=-1, index=idx).squeeze(-1)                 # [N,B,L]
    if mask is not None:
        logp_tok = logp_tok * mask.unsqueeze(0)                                  # drop excluded positions
    seq_logp = logp_tok.sum(dim=-1)                                              # [N,B]
    n = seq_logp.shape[0]
    return torch.logsumexp(seq_logp, dim=0) - torch.log(torch.tensor(float(n), device=seq_logp.device))


def consistency_loss(direct_lambda_logits, target_tokens, mask=None, baseline_lambda_logits=None):
    """Di4C consistency loss (Eq. 4), sampled form.

    The direct student's mixture p^theta(.|x_t) is trained to assign high probability to
    ``target_tokens`` sampled from the composed teacher-then-student path (drawn by the caller
    WITH stop-gradient), i.e. minimize -E[ log p^theta_mixture(target | x_t) ]. Gradients flow
    only through ``direct_lambda_logits`` -- matching the paper's stop-grad on the composed
    student.

    Args:
        direct_lambda_logits: [N, B, L, V] direct-student logits at x_t for N lambdas (grad).
        target_tokens: [B, L] samples from the composed path (should be detached by the caller).
        mask: optional [B, L] bool of positions to score (e.g. the denoised/generated span).
        baseline_lambda_logits: optional [N, B, L, V] factorized/product-model logits used as a
            control variate; the loss is centered on the mixture's gain over this baseline,
            reducing variance without changing the optimum.
    """
    logp = mixture_log_prob(direct_lambda_logits, target_tokens, mask)           # [B]
    loss = -logp
    if baseline_lambda_logits is not None:
        baseline = mixture_log_prob(baseline_lambda_logits, target_tokens, mask).detach()
        loss = loss + baseline                                                   # control variate (centering)
    return loss.mean()
