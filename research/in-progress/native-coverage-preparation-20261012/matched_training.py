"""Deterministic stream scheduling and exact stream-loss accumulation.

No network, paid launch, data selection or training occurs on import.
"""
import hashlib
import json
import torch
from torch.nn import functional as F


def slot(seed, step, stream, index, eligible):
    """Counter-based draws keep source schedules independent of optimizer RNG."""
    if not eligible or len(eligible) != len(set(eligible)):
        raise ValueError('A nonempty set of unique, admitted patch IDs is required')
    ordered = sorted(eligible)
    key = json.dumps([int(seed), int(step), str(stream), int(index)], separators=(',', ':')).encode()
    digest = hashlib.sha256(key).digest()
    return dict(patch_id=ordered[int.from_bytes(digest[:8], 'little') % len(ordered)],
                augmentation_seed=int.from_bytes(digest[8:16], 'little'), stream=stream)


def stream_totals(targets, masks):
    if targets.shape != masks.shape or targets.ndim != 4 or targets.shape[1] != 1:
        raise ValueError('Expected matching N1HW targets and masks')
    if not torch.isfinite(targets).all() or not torch.all((targets == 0) | (targets == 1)):
        raise ValueError('Binary, finite targets required')
    if masks.dtype != torch.bool or torch.any(masks.flatten(1).sum(1) == 0):
        raise ValueError('Every sample needs a nonempty boolean supervision mask')
    return dict(pixels=int(masks.sum()), samples=len(targets))


def partial_stream_loss(logits, targets, masks, totals, *, bce_smoothing=.5, dice_weight=.25):
    """Contribution to one stream's loss, with totals fixed across microbatches.

    Matches the retained recipe: masked BCE / stream pixels plus 0.25 times
    negative per-image Dice averaged over stream images, Dice smoothing 1.
    BCE and Dice use different denominators; simply averaging microbatch losses
    is incorrect when valid-pixel counts differ.
    """
    current = stream_totals(targets, masks)
    if logits.shape != targets.shape or not torch.isfinite(logits).all():
        raise ValueError('Finite logits must match targets')
    if totals['pixels'] < current['pixels'] or totals['samples'] < current['samples']:
        raise ValueError('Invalid full-stream totals')
    z = logits.float(); y = targets.float(); mask = masks
    smoothed = y * (1-bce_smoothing) + .5*bce_smoothing
    bce = (F.binary_cross_entropy_with_logits(z, smoothed, reduction='none')*mask).sum()/totals['pixels']
    prob = z.sigmoid()
    axes = tuple(range(2, z.ndim))
    overlap = (prob*y*mask).sum(axes)
    pred_sum = (prob*mask).sum(axes)
    true_sum = (y*mask).sum(axes)
    dice = -((2*overlap+1)/(pred_sum+true_sum+1)).sum()/totals['samples']
    return bce+dice_weight*dice


def require_sample_independent_model(model):
    """Reject normalization/stochastic couplings absent from this qualification."""
    forbidden = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout,
                 torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.AlphaDropout)
    bad = [name for name, module in model.named_modules() if isinstance(module, forbidden)]
    running = [name for name,module in model.named_modules()
               if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm) and module.track_running_stats]
    if bad or running:
        raise ValueError(f'Unqualified batch/stochastic model coupling: {bad + running}')


def tensor_state_hash(state):
    """Hash tensor values/names without depending on torch container byte order."""
    h=hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode()+str(value.dtype).encode()+str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()
