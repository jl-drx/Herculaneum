"""Matched schedule, XY transforms and explicitly masked unsmoothed losses."""
import numpy as np


def build_schedule(tile_lists, steps=32, seed=20261008):
    if len(tile_lists) != 2 or any(len(tiles) != 6 for tiles in tile_lists):
        raise ValueError('Exactly six supported tiles per adaptation region required')
    rng = np.random.default_rng(seed)
    result = []
    for step in range(steps):
        if step % 6 == 0:
            permutations = [rng.permutation(6) for _ in tile_lists]
        result.append(dict(step=step + 1, patches=[dict(region=r,
            bbox=tile_lists[r][int(permutations[r][step % 6])], augmentation=int(rng.integers(8)))
            for r in range(2)]))
    return result


def augment(array, code):
    result = np.rot90(array, code % 4, axes=(-2, -1))
    if code >= 4:
        result = np.flip(result, axis=-1)
    return np.ascontiguousarray(result)


def losses(logits, manual_mask, manual_target, pseudo_mask, pseudo_target):
    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits as bce
    if logits.shape != manual_mask.shape or logits.shape[0] != 2:
        raise ValueError('Require batch2 aligned targets/masks')
    if bool((manual_mask & pseudo_mask).any()) or not bool(manual_mask.any()) or not bool(pseudo_mask.any()):
        raise ValueError('Manual and pseudo support must be nonempty and disjoint')
    for mask, target in [(manual_mask, manual_target), (pseudo_mask, pseudo_target)]:
        if not bool(((target[mask] == 0) | (target[mask] == 1)).all()):
            raise ValueError('Only explicit hard targets may participate')
    uniform_manual = bce(logits[manual_mask], manual_target[manual_mask])
    uniform_pseudo = bce(logits[pseudo_mask], pseudo_target[pseudo_mask])
    group_losses = []
    for region in range(2):
        for label in [0., 1.]:
            mask = manual_mask[region] & (manual_target[region] == label)
            if not bool(mask.any()):
                raise ValueError('Each region needs both manual classes')
            group_losses.append(bce(logits[region][mask], manual_target[region][mask]))
    return uniform_manual, uniform_pseudo, torch.stack(group_losses).mean()


def arm_loss(arm, uniform_manual, uniform_pseudo, balanced_manual):
    if arm == 'A': return uniform_manual
    if arm == 'B': return uniform_manual + .25 * uniform_pseudo
    if arm == 'C': return .5 * uniform_manual + .5 * balanced_manual + .25 * uniform_pseudo
    raise ValueError('Unknown fixed arm')
