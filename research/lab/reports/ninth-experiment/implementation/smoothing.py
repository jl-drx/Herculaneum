"""Participant-only BCE target smoothing; hard masks and group identity preserved."""
from training_mechanics import losses as hard_losses


def smoothed_losses(logits,manual_mask,manual_target,pseudo_mask,pseudo_target,epsilon=.5):
    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits as bce
    if epsilon not in [0.,.5]: raise ValueError('Only fixed epsilon0/0.5 control supported')
    hard=hard_losses(logits,manual_mask,manual_target,pseudo_mask,pseudo_target)
    if epsilon==0.: return hard
    def term(values,targets): return bce(values,.5*epsilon+(1.-epsilon)*targets)
    u=term(logits[manual_mask],manual_target[manual_mask])
    p=term(logits[pseudo_mask],pseudo_target[pseudo_mask]); groups=[]
    for region in range(2):
        for label in [0.,1.]:
            mask=manual_mask[region] & (manual_target[region]==label)  # Membership remains hard class.
            groups.append(term(logits[region][mask],manual_target[region][mask]))
    return u,p,torch.stack(groups).mean()  # R is diagnostic only in A_smooth/B_smooth.
