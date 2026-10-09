"""Participant-only temperature-one Bernoulli KL to a detached frozen teacher."""


def preservation_losses(student_logits,teacher_logits,manual_mask,pseudo_mask):
    import torch
    if student_logits.shape!=teacher_logits.shape or student_logits.shape!=manual_mask.shape or manual_mask.shape!=pseudo_mask.shape:
        raise ValueError('Aligned logits and participation required')
    if manual_mask.dtype!=torch.bool or pseudo_mask.dtype!=torch.bool or bool((manual_mask&pseudo_mask).any()):
        raise ValueError('Original disjoint Boolean masks required')
    if not bool(manual_mask.any()) or not bool(pseudo_mask.any()): raise ValueError('Nonempty manual/pseudo support required')
    z=student_logits.double(); t=teacher_logits.detach().double()
    if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(t).all()): raise ValueError('Finite logits required')
    q=torch.sigmoid(t)
    # logaddexp avoids softplus's threshold approximation; equality gives exact zero.
    divergence=torch.logaddexp(z,torch.zeros_like(z))-torch.logaddexp(t,torch.zeros_like(t))-q*(z-t)
    manual=divergence[manual_mask].mean(); pseudo=divergence[pseudo_mask].mean()
    if not bool(torch.isfinite(manual)) or not bool(torch.isfinite(pseudo)) or min(float(manual.detach()),float(pseudo.detach())) < -1e-6:
        raise ValueError('KL finite/nonnegative-with-roundoff guard failed')
    return manual,pseudo
