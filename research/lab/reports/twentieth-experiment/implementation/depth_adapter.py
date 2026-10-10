"""Manual objective unchanged from phase18; architecture imported from phase19."""
import torch
from depth_common import require
from fusion_adapter import ARMS, FusionAdapter, FrozenFusionModel, depth_features as features

def manual_terms(logits, teacher, mask, hard_target):
    """FP32 epsilon=.5 manual BCE plus FP64 KL(teacher || student), unit weight."""
    require(mask.dtype == torch.bool and mask.shape == logits.shape == teacher.shape == hard_target.shape
            and bool(mask.any()) and bool(((hard_target[mask] == 0) | (hard_target[mask] == 1)).all()),
            'Manual participation/target contract differs')
    soft = .25 + .5 * hard_target[mask]
    base = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], soft)
    z = logits[mask].double()
    t = teacher.detach()[mask].double()
    divergence = torch.logaddexp(z, torch.zeros_like(z)) - torch.logaddexp(t, torch.zeros_like(t)) - torch.sigmoid(t) * (z - t)
    kl = divergence.mean()
    total = base + kl
    require(bool(torch.isfinite(total)) and float(kl.detach()) >= -1e-6, 'Nonfinite/negative KL loss')
    with torch.no_grad():
        hard = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], hard_target[mask])
        drift = (torch.sigmoid(z) - torch.sigmoid(t)).abs().mean()
    return total, dict(base=float(base.detach()), full=float(total.detach()), kl=float(kl.detach()),
                      hard_bce=float(hard), probability_drift_manual=float(drift))
