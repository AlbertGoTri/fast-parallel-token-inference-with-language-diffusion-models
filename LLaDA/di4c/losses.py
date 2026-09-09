"""Di4C loss functions.

Phase 0 implements the distillation loss only (KL between the teacher's and student's
single-step denoising distributions). The consistency loss (the correlation-learning core,
with stop-grad + product control variate + log-sum-exp) and the auxiliary marginal/data losses
are added in the next phase.
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
