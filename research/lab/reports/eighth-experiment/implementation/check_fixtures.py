"""Synthetic mechanism checks; never loads a model, checkpoint, or scroll data."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

from training_mechanics import arm_loss, augment, build_schedule, losses


HERE = Path(__file__).resolve().parent
OPERATING_POINT = HERE.parents[1] / 'seventh-experiment/benchmark/operating_point.py'
OPERATING_SHA = '38a0ebbab0835845ee2213c2be9505cf9b41a129ab13dd6668ff797bfacacb73'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(call, message):
    try:
        call()
    except ValueError:
        return
    raise AssertionError(message)


def scalar_bce(logit, target):
    # Independent scalar softplus, without using Torch's BCE implementation.
    return max(logit, 0.) + math.log1p(math.exp(-abs(logit))) - logit * target


def close(actual, expected, message):
    value = float(actual.detach()) if isinstance(actual, torch.Tensor) else float(actual)
    require(math.isclose(value, float(expected), rel_tol=1e-12, abs_tol=1e-12), message)


def fixture_arrays():
    values = np.linspace(-1.7, 2.1, 32, dtype=np.float64).reshape(2, 4, 4)
    manual = np.zeros(values.shape, dtype=bool)
    pseudo = np.zeros(values.shape, dtype=bool)
    manual_target = np.full(values.shape, np.nan, dtype=np.float64)
    pseudo_target = np.full(values.shape, np.nan, dtype=np.float64)
    for region, indices, targets in [
        (0, [0, 1, 2, 4, 5, 8], [1, 1, 0, 0, 0, 0]),
        (1, [0, 2, 3, 5, 6, 7, 8, 10, 11], [1, 0, 0, 0, 0, 0, 0, 0, 0]),
    ]:
        manual[region].flat[indices] = True
        manual_target[region].flat[indices] = targets
    for region, indices, targets in [
        (0, [3, 6, 9], [0, 1, 0]),
        (1, [1, 4, 12, 15], [1, 0, 0, 1]),
    ]:
        pseudo[region].flat[indices] = True
        pseudo_target[region].flat[indices] = targets
    require(not (manual & pseudo).any(), 'Synthetic masks must be disjoint')
    require((~(manual | pseudo)).any(), 'Fixture must contain unknown pixels')
    return values, manual, manual_target, pseudo, pseudo_target


def independent_means(values, manual, manual_target, pseudo, pseudo_target):
    manual_losses = [scalar_bce(x, y) for x, y in zip(values[manual], manual_target[manual])]
    pseudo_losses = [scalar_bce(x, y) for x, y in zip(values[pseudo], pseudo_target[pseudo])]
    group_means = []
    for region in range(2):
        for target in [0., 1.]:
            mask = manual[region] & (manual_target[region] == target)
            group = [scalar_bce(x, y) for x, y in zip(values[region][mask], manual_target[region][mask])]
            group_means.append(math.fsum(group) / len(group))
    return math.fsum(manual_losses) / len(manual_losses), math.fsum(pseudo_losses) / len(pseudo_losses), math.fsum(group_means) / 4


def tensors(arrays):
    return [torch.from_numpy(np.array(array, copy=True)) for array in arrays]


def check_losses_and_gradients():
    arrays = fixture_arrays()
    values, manual, manual_target, pseudo, pseudo_target = arrays
    expected_u, expected_p, expected_r = independent_means(*arrays)
    require(abs(expected_u - expected_r) > .01, 'Fixture must distinguish pixel and balanced means')
    totals = {}
    for arm in ['A', 'B', 'C']:
        logits, manual_tensor, manual_targets, pseudo_tensor, pseudo_targets = tensors(arrays)
        logits.requires_grad_(True)
        uniform_manual, uniform_pseudo, balanced_manual = losses(logits, manual_tensor, manual_targets, pseudo_tensor, pseudo_targets)
        close(uniform_manual, expected_u, f'{arm}: U must be a global manual pixel mean')
        close(uniform_pseudo, expected_p, f'{arm}: P must be a global pseudo pixel mean')
        close(balanced_manual, expected_r, f'{arm}: R must equally weight four manual groups')
        expected_total = {'A': expected_u, 'B': expected_u + .25 * expected_p,
                          'C': .5 * expected_u + .5 * expected_r + .25 * expected_p}[arm]
        total = arm_loss(arm, uniform_manual, uniform_pseudo, balanced_manual)
        close(total, expected_total, f'{arm}: exact fixed coefficient formula')
        total.backward()
        gradient = logits.grad.numpy()
        expected_gradient = np.zeros_like(values)
        for region in range(2):
            for row in range(4):
                for col in range(4):
                    index = region, row, col
                    if manual[index]:
                        target = manual_target[index]
                        coefficient = 1. / int(manual.sum())
                        if arm == 'C':
                            group_size = int((manual[region] & (manual_target[region] == target)).sum())
                            coefficient = .5 / int(manual.sum()) + .5 / (4 * group_size)
                    elif pseudo[index] and arm != 'A':
                        target = pseudo_target[index]
                        coefficient = .25 / int(pseudo.sum())
                    else:
                        continue
                    expected_gradient[index] = coefficient * (1. / (1. + math.exp(-values[index])) - target)
        require(np.allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12), f'{arm}: independently computed participant gradients')
        require(np.isfinite(gradient).all(), f'{arm}: ignored NaN targets must not contaminate gradients')
        require(np.count_nonzero(gradient[~(manual | pseudo)]) == 0, f'{arm}: unknown output gradients must be exactly zero')
        if arm == 'A':
            require(np.count_nonzero(gradient[~manual]) == 0, 'A: all nonmanual output gradients must be exactly zero')
        else:
            require(np.all(gradient[pseudo] != 0), f'{arm}: pseudo participants must affect loss')
        require(np.all(gradient[manual] != 0), f'{arm}: manual participants must affect loss')
        totals[arm] = float(total.detach())

    logits, manual_tensor, manual_targets, pseudo_tensor, pseudo_targets = tensors(arrays)
    overlap = pseudo_tensor.clone()
    overlap[0, 0, 0] = True
    rejects(lambda: losses(logits, manual_tensor, manual_targets, overlap, pseudo_targets), 'Overlapping supports must be rejected')
    for mask, target, role in [(manual_tensor, manual_targets, 'manual'), (pseudo_tensor, pseudo_targets, 'pseudo')]:
        invalid = target.clone()
        invalid[mask] = .5
        args = [logits, manual_tensor, manual_targets, pseudo_tensor, pseudo_targets]
        args[2 if role == 'manual' else 4] = invalid
        rejects(lambda: losses(*args), f'{role}: smoothed participating targets must be rejected')
        invalid[mask] = float('nan')
        rejects(lambda: losses(*args), f'{role}: nonfinite participating targets must be rejected')
    rejects(lambda: losses(logits, torch.zeros_like(manual_tensor), manual_targets, pseudo_tensor, pseudo_targets), 'Empty manual support must be rejected')
    rejects(lambda: losses(logits, manual_tensor, manual_targets, torch.zeros_like(pseudo_tensor), pseudo_targets), 'Empty pseudo support must be rejected')
    missing_class = manual_tensor.clone()
    missing_class[1] &= manual_targets[1] != 1
    rejects(lambda: losses(logits, missing_class, manual_targets, pseudo_tensor, pseudo_targets), 'A missing region/manual class must be rejected')
    rejects(lambda: losses(logits[:1], manual_tensor[:1], manual_targets[:1], pseudo_tensor[:1], pseudo_targets[:1]), 'Batch size must be exactly two')
    rejects(lambda: arm_loss('D', 1., 1., 1.), 'Unspecified arms must be rejected')
    return dict(U=expected_u, P=expected_p, R=expected_r, arm_losses=totals,
                manual_pixels=int(manual.sum()), pseudo_pixels=int(pseudo.sum()), unknown_pixels=int((~(manual | pseudo)).sum()))


def support_counts(arrays):
    _, manual, manual_target, pseudo, pseudo_target = arrays
    return [dict(manual_positive=int((manual[r] & (manual_target[r] == 1)).sum()),
                 manual_negative=int((manual[r] & (manual_target[r] == 0)).sum()),
                 pseudo_positive=int((pseudo[r] & (pseudo_target[r] == 1)).sum()),
                 pseudo_negative=int((pseudo[r] & (pseudo_target[r] == 0)).sum()),
                 unknown=int((~(manual[r] | pseudo[r])).sum())) for r in range(2)]


def check_augmentations_and_schedule():
    arrays = fixture_arrays()
    counts = support_counts(arrays)
    means = independent_means(*arrays)
    image_variants = set()
    for code in range(8):
        transformed = tuple(augment(array, code) for array in arrays)
        require(all(array.flags.c_contiguous for array in transformed), f'D4 {code}: arrays must be contiguous')
        require(support_counts(transformed) == counts, f'D4 {code}: region/class exposures must remain equal')
        results = losses(*tensors(transformed))
        for actual, expected in zip(results, means):
            close(actual, expected, f'D4 {code}: aligned transforms must preserve loss means')
        image_variants.add(transformed[0].tobytes())
    require(len(image_variants) == 8, 'Asymmetric fixture must distinguish all eight XY D4 transforms')
    tiles = [[[64 * index, 64 * region, 128, 128] for index in range(6)] for region in range(2)]
    schedules = [build_schedule(tiles, steps=32, seed=20261008) for _ in ['A', 'B', 'C']]
    encoded = [json.dumps(schedule, sort_keys=True, separators=(',', ':')).encode() for schedule in schedules]
    require(encoded[0] == encoded[1] == encoded[2], 'All arms must receive the same schedule bytes')
    require(build_schedule(tiles, steps=32, seed=20261009) != schedules[0], 'A changed random seed must change schedule')
    schedule = schedules[0]
    require(len(schedule) == 32 and [entry['step'] for entry in schedule] == list(range(1, 33)), 'Exactly 32 numbered steps')
    for start in range(0, 32, 6):
        block = schedule[start:start + 6]
        for region in range(2):
            selected = []
            for entry in block:
                require(len(entry['patches']) == 2, 'Each batch contains one tile from each region')
                patch = entry['patches'][region]
                require(patch['region'] == region, 'Region membership must be stable')
                require(0 <= patch['augmentation'] < 8, 'Only the eight frozen D4 transforms are allowed')
                require(patch['bbox'] in tiles[region], 'Scheduled tile must be from supported list')
                selected.append(tuple(patch['bbox']))
            require(len(set(selected)) == len(block), 'Each cycle samples without replacement')
            if len(block) == 6:
                require(set(selected) == {tuple(tile) for tile in tiles[region]}, 'Every complete cycle exposes all six tiles')
    rejects(lambda: build_schedule([tiles[0]], steps=32), 'Exactly two adaptation regions required')
    rejects(lambda: build_schedule([tiles[0], tiles[1][:5]], steps=32), 'Exactly six supported tiles per region required')
    return dict(d4_transforms=8, schedule_steps=32, schedule_sha256=hashlib.sha256(encoded[0]).hexdigest(), region_class_counts=counts)


def check_operating_cutoffs():
    require(hashlib.sha256(OPERATING_POINT.read_bytes()).hexdigest() == OPERATING_SHA, 'The seventh operating-point helper must remain unchanged')
    spec = importlib.util.spec_from_file_location('unchanged_seventh_operating_point', OPERATING_POINT)
    operating = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operating)
    tied = np.array([.99] + [.95] * 3 + [.2] * 36, dtype=np.float32)
    threshold = operating.choose_development_threshold(tied)
    require(threshold['maximum_false_positives'] == 2, '40 negatives allow exactly two false positives')
    require(threshold['boundary_tie_count'] == 3, 'Fixture must exercise a crossing three-way tie')
    require(threshold['development_false_positives'] == 1, 'Conservative tie exclusion must retain only the .99 negative')
    require(threshold['threshold'] > float(np.float32(.95)), 'Threshold must advance above the entire crossing tie')
    require(int((tied.astype(np.float64) >= float(np.float32(.95))).sum()) == 4, 'Including the crossing tie would violate 95% specificity')
    require(threshold['development_specificity'] >= .95, 'Tie cutoff must meet fixed development specificity')
    ones = np.ones(100, dtype=np.float32)
    all_one_threshold = operating.choose_development_threshold(ones)
    require(all_one_threshold['threshold'] > 1., 'All-one scores require a finite threshold above one')
    require(math.isfinite(all_one_threshold['threshold']), 'Threshold above one must remain finite')
    require(all_one_threshold['development_false_positives'] == 0, 'All-one tie must be entirely excluded')
    operating_result = operating.operating_metrics(np.array([True, False], dtype=bool), np.array([1., 1.], dtype=np.float32), all_one_threshold['threshold'])
    require(operating_result['confusion'] == dict(tp=0, fp=0, tn=1, fn=1), 'Threshold above one must predict no positives')
    rejects(lambda: operating.choose_development_threshold(np.array([.1], dtype=np.float64)), 'Calibration inputs must remain float32')
    rejects(lambda: operating.choose_development_threshold(np.array([np.nan], dtype=np.float32)), 'Nonfinite calibration negatives must be rejected')
    return dict(tie_false_positives=1, tie_specificity=threshold['development_specificity'], all_one_threshold=all_one_threshold['threshold'])


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = dict(status='PASS', scope='Synthetic logits and masks only; no model, checkpoint, or scroll data loaded.',
                  loss_and_gradients=check_losses_and_gradients(),
                  augmentation_and_schedule=check_augmentations_and_schedule(),
                  operating_cutoffs=check_operating_cutoffs())
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
