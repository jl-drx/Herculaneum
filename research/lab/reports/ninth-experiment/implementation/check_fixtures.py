"""Tiny synthetic smoothing checks; no model, data, checkpoint, or network."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

import control_common  # Verify immutable eighth helpers and initialize their import path.
from smoothing import smoothed_losses

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'eighth-experiment/implementation/training_mechanics.py'
spec = importlib.util.spec_from_file_location('unchanged_eighth_mechanics', OLD)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def fixture():
    values = np.linspace(-2.2, 2.4, 20).reshape(2, 2, 5)
    manual, pseudo = np.zeros_like(values, dtype=bool), np.zeros_like(values, dtype=bool)
    mt, pt = np.full_like(values, np.nan), np.full_like(values, -1.)
    for region, indices, targets in [(0, [0, 1, 2, 3], [0, 0, 0, 1]),
                                     (1, [0, 1, 2], [0, 1, 1])]:
        manual[region].flat[indices] = True
        mt[region].flat[indices] = targets
    for region, indices, targets in [(0, [4, 5], [0, 1]), (1, [3, 4, 5], [1, 0, 0])]:
        pseudo[region].flat[indices] = True
        pt[region].flat[indices] = targets
    require(not (manual & pseudo).any() and (~(manual | pseudo)).any(), 'Nonempty disjoint supports and unknown pixels')
    return values, manual, mt, pseudo, pt


def analytic(arrays, epsilon):
    values, manual, mt, pseudo, pt = arrays
    def mean(mask, targets):
        pairs = zip(values[mask], targets[mask])
        return math.fsum(max(x, 0.) + math.log1p(math.exp(-abs(x))) - x * ((1 - epsilon) * y + epsilon / 2)
                         for x, y in pairs) / int(mask.sum())
    groups = []
    for region in range(2):
        for hard_class in [0., 1.]:
            mask = np.zeros_like(manual)
            mask[region] = manual[region] & (mt[region] == hard_class)
            groups.append(mean(mask, mt))
    return mean(manual, mt), mean(pseudo, pt), math.fsum(groups) / 4


def expected_gradient(arrays, epsilon, coefficients):
    values, manual, mt, pseudo, pt = arrays
    result = np.zeros_like(values)
    for region in range(2):
        for row in range(2):
            for col in range(5):
                ix = region, row, col
                if manual[ix]:
                    target = mt[ix]
                    group_size = int((manual[region] & (mt[region] == target)).sum())
                    weight = coefficients[0] / int(manual.sum()) + coefficients[2] / (4 * group_size)
                elif pseudo[ix]:
                    target, weight = pt[ix], coefficients[1] / int(pseudo.sum())
                else:
                    continue
                soft_target = (1 - epsilon) * target + epsilon / 2
                result[ix] = weight * (1 / (1 + math.exp(-values[ix])) - soft_target)
    return result


def execute(arrays, epsilon, component=None, arm=None, legacy=False):
    tensors = [torch.from_numpy(array.copy()) for array in arrays]
    tensors[0].requires_grad_(True)
    before = [tensor.detach().numpy().copy() for tensor in tensors[1:]]
    parts = old.losses(*tensors) if legacy else smoothed_losses(*tensors, epsilon=epsilon)
    total = old.arm_loss(arm, *parts) if arm else parts[component]
    total.backward()
    for tensor, original in zip(tensors[1:], before):
        require(np.array_equal(tensor.numpy(), original, equal_nan=True), 'Masks and hard targets must remain unchanged')
    return tuple(float(part.detach()) for part in parts), tensors[0].grad.detach().clone()


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    arrays = fixture()
    for component in range(3):
        original, original_grad = execute(arrays, 0., component=component, legacy=True)
        actual, actual_grad = execute(arrays, 0., component=component)
        require(actual == original and torch.equal(actual_grad, original_grad), 'Epsilon zero must exactly preserve all losses and gradients')
    expected = analytic(arrays, .5)
    require(abs(expected[0] - expected[2]) > .001, 'Unequal group sizes must distinguish U from R')
    for component in range(3):
        actual, gradient = execute(arrays, .5, component=component)
        require(np.allclose(actual, expected, rtol=1e-12, atol=1e-12), 'Independent scalar softplus loss means')
        coefficient = tuple(float(index == component) for index in range(3))
        require(np.allclose(gradient.numpy(), expected_gradient(arrays, .5, coefficient), rtol=1e-12, atol=1e-12), 'Independent sigmoid-minus-soft-target component derivatives')
    for arm, coefficient in [('A', (1., 0., 0.)), ('B', (1., .25, 0.))]:
        original, original_grad = execute(arrays, 0., arm=arm, legacy=True)
        actual_zero, zero_grad = execute(arrays, 0., arm=arm)
        require(actual_zero == original and torch.equal(zero_grad, original_grad), f'{arm}: exact epsilon-zero legacy arm gradient')
        _, gradient = execute(arrays, .5, arm=arm)
        require(np.allclose(gradient.numpy(), expected_gradient(arrays, .5, coefficient), rtol=1e-12, atol=1e-12), f'{arm}: unchanged arm coefficients and smoothed derivative')
        inactive = ~arrays[1] if arm == 'A' else ~(arrays[1] | arrays[3])
        require(torch.count_nonzero(gradient[torch.from_numpy(inactive)]).item() == 0, f'{arm}: exact zero outside participating support')
        require(torch.isfinite(gradient).all().item(), 'Ignored NaN and -1 targets must not contaminate gradients')
    print(json.dumps(dict(status='PASS', scope=__doc__, manual_pixels=int(arrays[1].sum()),
                         pseudo_pixels=int(arrays[3].sum()), unknown_pixels=int((~(arrays[1] | arrays[3])).sum()),
                         epsilon_half_targets=[.25, .75], epsilon_half_losses=dict(zip(['U', 'P', 'R'], expected)),
                         fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         unchanged_mechanics_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest()), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
