"""Root-proposed synthetic-only preservation reallocation; not a training runner."""
import torch


def objective(logits,teacher,manual,retired_validation,hard_target,arm):
    if arm not in ['manual_twice','manual_plus_retired_validation']:
        raise ValueError('Unknown declared arm')
    if not(logits.dtype==teacher.dtype==hard_target.dtype==torch.float32):
        raise ValueError('Expected FP32 logits and manual targets')
    if not(logits.shape==teacher.shape==manual.shape==retired_validation.shape==hard_target.shape):
        raise ValueError('Aligned shapes required')
    if manual.dtype!=torch.bool or retired_validation.dtype!=torch.bool:
        raise ValueError('Boolean role masks required')
    if not bool(manual.any()) or not bool(retired_validation.any()) or bool((manual&retired_validation).any()):
        raise ValueError('Both supplied role masks must be nonempty and disjoint')
    support=manual|retired_validation
    if not bool(torch.isfinite(logits[support]).all()) or not bool(torch.isfinite(teacher[support]).all()):
        raise ValueError('Finite participating logits required')
    if not bool(((hard_target[manual]==0)|(hard_target[manual]==1)).all()):
        raise ValueError('Explicit manual hard labels required')
    base=torch.nn.functional.binary_cross_entropy_with_logits(logits[manual],.25+.5*hard_target[manual])
    def mean_kl(mask):
        z=logits[mask].double();t=teacher.detach()[mask].double()
        return (torch.logaddexp(z,torch.zeros_like(z))-torch.logaddexp(t,torch.zeros_like(t))-torch.sigmoid(t)*(z-t)).mean()
    manual_kl=mean_kl(manual);retired_kl=mean_kl(retired_validation)
    coefficients=(2.,0.) if arm=='manual_twice' else (1.,1.)
    total=base+coefficients[0]*manual_kl+coefficients[1]*retired_kl
    if not bool(torch.isfinite(total)):raise ValueError('Nonfinite objective')
    return total,dict(base=base,manual_kl=manual_kl,retired_validation_kl=retired_kl,
        manual_kl_coefficient=coefficients[0],retired_validation_kl_coefficient=coefficients[1],
        manual_pixels=int(manual.sum()),retired_validation_pixels=int(retired_validation.sum()))
