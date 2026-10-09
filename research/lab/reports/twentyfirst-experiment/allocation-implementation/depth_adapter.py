"""Matched fixed-feature fusion arms differing only in declared KL allocation."""
import torch
from depth_common import require
from fusion_adapter import FusionAdapter,FrozenFusionModel,depth_features as features

ARMS=['manual_double','manual_retired']


def manual_terms(logits,teacher,mask,hard_target,retired,arm):
    require(arm in ARMS,'Unknown preservation arm')
    require(mask.dtype==retired.dtype==torch.bool
            and mask.shape==retired.shape==logits.shape==teacher.shape==hard_target.shape
            and bool(mask.any()) and bool(retired.any()) and not bool((mask&retired).any()),
            'Nonempty disjoint manual and retired reference masks required')
    require(logits.dtype==teacher.dtype==hard_target.dtype==torch.float32
            and bool(((hard_target[mask]==0)|(hard_target[mask]==1)).all()),'Explicit FP32 manual hard targets required')
    support=mask|retired
    require(bool(torch.isfinite(logits[support]).all()) and bool(torch.isfinite(teacher[support]).all()),
            'Nonfinite participating logits')
    base=torch.nn.functional.binary_cross_entropy_with_logits(logits[mask],.25+.5*hard_target[mask])
    def kl_and_drift(participation):
        z=logits[participation].double();t=teacher.detach()[participation].double()
        kl=(torch.logaddexp(z,torch.zeros_like(z))-torch.logaddexp(t,torch.zeros_like(t))-torch.sigmoid(t)*(z-t)).mean()
        with torch.no_grad():drift=(torch.sigmoid(z)-torch.sigmoid(t)).abs().mean()
        return kl,drift
    manual_kl,manual_drift=kl_and_drift(mask);retired_kl,retired_drift=kl_and_drift(retired)
    objective_A=base+2*manual_kl
    objective_B=base+manual_kl+retired_kl
    total=objective_A if arm=='manual_double' else objective_B
    require(bool(torch.isfinite(total)) and min(float(manual_kl.detach()),float(retired_kl.detach()))>=-1e-6,
            'Nonfinite/negative KL')
    with torch.no_grad():hard=torch.nn.functional.binary_cross_entropy_with_logits(logits[mask],hard_target[mask])
    return total,dict(base=float(base.detach()),full=float(total.detach()),
        kl_manual=float(manual_kl.detach()),kl_retired=float(retired_kl.detach()),
        objective_A=float(objective_A.detach()),objective_B=float(objective_B.detach()),
        hard_bce=float(hard),probability_drift_manual=float(manual_drift),probability_drift_retired=float(retired_drift))
