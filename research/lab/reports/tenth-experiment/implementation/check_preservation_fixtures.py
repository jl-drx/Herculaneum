"""Tiny independent preservation fixtures; no model, checkpoint, data, or network."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import torch
from preservation import preservation_losses

HERE = Path(__file__).resolve().parent
EIGHTH = HERE.parents[1] / 'eighth-experiment/implementation/training_mechanics.py'
SMOOTHING = HERE.parents[1] / 'ninth-experiment/implementation/smoothing.py'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sigmoid(value):
    return 1. / (1. + math.exp(-value))


def close(actual, expected, tolerance=1e-12):
    require(torch.allclose(torch.as_tensor(actual).double(), torch.as_tensor(expected).double(),
                           rtol=tolerance, atol=tolerance), 'Independent value/gradient mismatch')


def fixture(dtype=torch.float64):
    z = torch.linspace(-2.5, 2.7, 12, dtype=dtype).reshape(2, 6).requires_grad_()
    t = torch.linspace(2.1, -2.3, 12, dtype=dtype).reshape(2, 6).requires_grad_()
    m = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    p = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0]], dtype=torch.bool)
    return z, t, m, p


def reference(z, t, m, p):
    values, gradients = [], []
    for mask in [m, p]:
        terms, grad = [], torch.zeros_like(z, dtype=torch.float64)
        for index in zip(*mask.nonzero(as_tuple=True)):
            a, b = sigmoid(float(z[index].detach())), sigmoid(float(t[index].detach()))
            terms.append(b * math.log(b / a) + (1 - b) * math.log((1 - b) / (1 - a)))
            grad[index] = (a - b) / int(mask.sum())
        values.append(math.fsum(terms) / len(terms)); gradients.append(grad)
    return values, gradients


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    for dtype, tolerance in [(torch.float64, 1e-12), (torch.float32, 2e-7)]:
        for component in range(2):
            z, t, m, p = fixture(dtype)
            expected, gradient = reference(z, t, m, p)
            parts = preservation_losses(z, t, m, p); parts[component].backward()
            close(torch.stack(parts), torch.tensor(expected, dtype=torch.float64))
            close(z.grad, gradient[component], tolerance)
            require(z.grad.dtype == dtype and bool(torch.isfinite(z.grad).all()), 'Finite input-dtype backcast')
            require(t.grad is None, 'Teacher must be detached')
            require(int(torch.count_nonzero(z.grad[~(m | p)])) == 0, 'Ignored student gradients exactly zero')
    z, _, m, p = fixture(); t = z.detach().clone().requires_grad_()
    parts = preservation_losses(z, t, m, p); sum(parts).backward()
    close(torch.stack(parts), torch.zeros(2)); close(z.grad, torch.zeros_like(z), 1e-15)
    require(t.grad is None, 'Equal-logit teacher still detached')
    z = torch.tensor([[-1000., 1000., -1000., 1000.], [0., 0., 1000., -1000.]], requires_grad=True)
    t = torch.tensor([[1000., -1000., 0., 0.], [1000., -1000., 1000., -1000.]], requires_grad=True)
    m = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=torch.bool); p = ~m
    parts = preservation_losses(z, t, m, p); sum(parts).backward()
    close(torch.stack(parts), torch.full((2,), 375., dtype=torch.float64))
    close(z.grad, torch.tensor([[-.25, .25, -.125, .125], [-.125, .125, 0., 0.]]))
    require(bool(torch.isfinite(z.grad).all()) and t.grad is None, 'Extreme-logit finite gradients and detached teacher')
    # This tiny negative cancellation is accepted and returned, rather than clamped.
    z = torch.full((2,), .3 + 1e-9, dtype=torch.float64)
    t = torch.full((2,), .3, dtype=torch.float64)
    m = torch.tensor([True, False]); p = ~m
    roundoff = preservation_losses(z, t, m, p)
    require(all(-1e-6 <= float(value) < 0. for value in roundoff), 'Allowed negative roundoff must remain unclipped')
    for path, expected in [(EIGHTH, '83dc4482381d4efeb79382e4698844116c90b02f3370e15940a912021ddf7e1b'),
                           (SMOOTHING, 'bfb66a0323a50850ebaa13c93ef4d02b6ee70289aa35bae57fa1d87b49f1ea56')]:
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Unchanged helper hash')
    sys.path.insert(0, str(EIGHTH.parent))
    spec = importlib.util.spec_from_file_location('pinned_ninth_smoothing', SMOOTHING)
    smoothing = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoothing)
    objectives = {}
    for arm, pseudo_coefficient in [('A', 0.), ('B', .25)]:
        z, t, m, p = fixture(); mt = torch.full_like(z, float('nan')); pt = torch.full_like(z, -1.)
        mt[m] = torch.tensor([0., 0., 1., 0., 1.], dtype=z.dtype)
        pt[p] = torch.tensor([0., 1., 1.], dtype=z.dtype)
        u, pseudo, _ = smoothing.smoothed_losses(z, m, mt, p, pt, epsilon=.5)
        dm, dp = preservation_losses(z, t, m, p)
        total = u + dm if arm == 'A' else u + .25 * pseudo + dm + .25 * dp
        expected_kl, expected_grad = reference(z, t, m, p); expected = expected_kl[0]
        gradient = expected_grad[0] + pseudo_coefficient * expected_grad[1]
        for mask, target, coefficient in [(m, mt, 1.), (p, pt, pseudo_coefficient)]:
            terms = []
            for index in zip(*mask.nonzero(as_tuple=True)):
                x = float(z[index].detach()); y = .25 + .5 * float(target[index])
                terms.append(max(x, 0.) + math.log1p(math.exp(-abs(x))) - x * y)
                gradient[index] += coefficient * (sigmoid(x) - y) / int(mask.sum())
            expected += coefficient * (math.fsum(terms) / len(terms))
        expected += pseudo_coefficient * expected_kl[1]; total.backward()
        close(total.detach(), torch.tensor(expected, dtype=torch.float64)); close(z.grad, gradient)
        require(t.grad is None and int(torch.count_nonzero(z.grad[~(m | p)])) == 0, 'Objective participation and teacher detachment')
        objectives[arm] = float(total.detach())
    print(json.dumps(dict(status='PASS', scope=__doc__, objectives=objectives,
                         roundoff_kl=float(roundoff[0]), extreme_kl=375.,
                         fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), sort_keys=True))


if __name__ == '__main__':
    main()
