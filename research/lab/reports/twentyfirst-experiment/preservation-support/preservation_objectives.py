"""Synthetic-only objective candidates; no scientific support or choice is implied."""
import torch


def require(ok,message):
    if not ok:raise ValueError(message)


def objective(logits,teacher,manual,hard_target,support,mode='manual_mean',weight=1.,manual_stratum_weight=.5):
    """FP32 BCE mean on M plus FP64 teacher KL at an explicitly declared support.

    `support` may mean geometry-valid or supplied-reference membership in a future
    contract; this function authenticates neither. Synthetic fixtures alone use it.
    """
    require(mode in ['manual_mean','support_mean','stratified_mean'],'Unknown reduction')
    require(logits.dtype==teacher.dtype==hard_target.dtype==torch.float32,'Expected FP32 inputs')
    require(manual.dtype==support.dtype==torch.bool and logits.shape==teacher.shape==manual.shape==hard_target.shape==support.shape,'Aligned Boolean masks required')
    require(bool(manual.any()) and not bool((manual&~support).any()),'Manual support must be nonempty subset')
    require(weight>=0 and 0<=manual_stratum_weight<=1,'Invalid declared weights')
    require(bool(torch.isfinite(logits[support]).all()) and bool(torch.isfinite(teacher[support]).all()),'Nonfinite participating logit')
    require(bool(((hard_target[manual]==0)|(hard_target[manual]==1)).all()),'Manual targets must be explicit hard labels')
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits[manual],.25+.5*hard_target[manual])
    z=logits[support].double();t=teacher.detach()[support].double()
    divergence=torch.logaddexp(z,torch.zeros_like(z))-torch.logaddexp(t,torch.zeros_like(t))-torch.sigmoid(t)*(z-t)
    manual_within_support=manual[support];extra=~manual_within_support
    manual_kl=divergence[manual_within_support].mean()
    if mode=='manual_mean':kl=manual_kl
    elif mode=='support_mean':kl=divergence.mean()
    else:
        require(bool(extra.any()),'Stratified mean requires nonempty extra support')
        kl=manual_stratum_weight*manual_kl+(1-manual_stratum_weight)*divergence[extra].mean()
    total=bce+weight*kl
    require(bool(torch.isfinite(total)),'Nonfinite objective')
    return total,dict(bce=bce,kl=kl,manual_kl=manual_kl,
        support_kl=divergence.mean(),extra_kl=divergence[extra].mean() if bool(extra.any()) else None,
        manual_pixels=int(manual.sum()),support_pixels=int(support.sum()),extra_pixels=int(extra.sum()),
        mode=mode,kl_weight=weight,manual_stratum_weight=manual_stratum_weight if mode=='stratified_mean' else None)
