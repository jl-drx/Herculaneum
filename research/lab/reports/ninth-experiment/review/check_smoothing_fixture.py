"""One independent batch-two, logits-only check of the ninth loss control."""
import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPLEMENTATION = HERE.parent / 'implementation'
EIGHTH = HERE.parents[1] / 'eighth-experiment' / 'implementation'
OLD_MECHANICS_SHA = '83dc4482381d4efeb79382e4698844116c90b02f3370e15940a912021ddf7e1b'
OLD_COMMON_SHA = '8d933dae02248d7b5fbbd26370c95fca53191bcff4e1dc51e984c40d8095b8cf'
ABSOLUTE_TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve(strict=True)
    require(path.stat().st_size < 65536, 'Fixture source unexpectedly large')
    return dict(path=str(path), size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def sigmoid(value):
    if value >= 0:
        return 1. / (1. + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1. + exponential)


def expected(logits, manual, manual_targets, pseudo, pseudo_targets, epsilon):
    """Scalar Python arithmetic, independent of Torch's BCE and autograd."""
    size = len(logits)
    terms = {}
    for name, mask, targets in [('U', manual, manual_targets),
                                ('P', pseudo, pseudo_targets)]:
        count = sum(mask)
        value = 0.
        gradient = [0.] * size
        for index, participating in enumerate(mask):
            if participating:
                target = .5 * epsilon + (1. - epsilon) * targets[index]
                z = logits[index]
                value += (max(z, 0.) + math.log1p(math.exp(-abs(z))) - z * target) / count
                gradient[index] = (sigmoid(z) - target) / count
        terms[name] = (value, gradient)
    balanced_value = 0.
    balanced_gradient = [0.] * size
    for region in range(2):
        for hard_class in [0., 1.]:
            selected = [i for i in range(region * 8, (region + 1) * 8)
                        if manual[i] and manual_targets[i] == hard_class]
            require(selected, 'Fixture needs both manual classes in each region')
            for index in selected:
                target = .5 * epsilon + (1. - epsilon) * hard_class
                z = logits[index]
                weight = 1. / (4 * len(selected))
                balanced_value += weight * (max(z, 0.) + math.log1p(math.exp(-abs(z))) - z * target)
                balanced_gradient[index] = weight * (sigmoid(z) - target)
    terms['R'] = (balanced_value, balanced_gradient)
    terms['A'] = terms['U']
    terms['B'] = (terms['U'][0] + .25 * terms['P'][0],
                  [u + .25 * p for u, p in zip(terms['U'][1], terms['P'][1])])
    return terms


def verify_fixture():
    paths = [Path(__file__), IMPLEMENTATION / 'control_common.py',
             IMPLEMENTATION / 'smoothing.py', EIGHTH / 'experiment_common.py',
             EIGHTH / 'training_mechanics.py']
    before = [record(path) for path in paths]
    require(before[3]['sha256'] == OLD_COMMON_SHA and before[4]['sha256'] == OLD_MECHANICS_SHA,
            'Immutable eighth helper identity differs')
    require('training_mechanics' not in sys.modules and 'smoothing' not in sys.modules,
            'Run this fixture in a fresh Python process')
    sys.path.insert(0, str(IMPLEMENTATION))
    common = importlib.import_module('control_common')
    smoothing = importlib.import_module('smoothing')
    old = importlib.import_module('training_mechanics')
    require(Path(common.__file__).resolve() == IMPLEMENTATION / 'control_common.py',
            'Unexpected control_common module')
    require(Path(smoothing.__file__).resolve() == IMPLEMENTATION / 'smoothing.py'
            and Path(old.__file__).resolve() == EIGHTH / 'training_mechanics.py',
            'Unexpected loss module')
    import torch
    torch.set_num_threads(4)
    values = [-100., -3., -.5, 0., .25, 2., 100., -1.,
              -100., -2., -.25, .5, 1., 3., 100., -4.]
    manual = [True, True, False, False, True, False, False, False,
              True, False, True, True, False, True, False, False]
    pseudo = [False, False, True, False, False, True, True, False,
              False, True, False, False, True, False, False, False]
    manual_targets = [float('nan')] * 16
    pseudo_targets = [float('nan')] * 16
    for index, target in {0: 0., 1: 1., 4: 1., 8: 1., 10: 0., 11: 1., 13: 0.}.items():
        manual_targets[index] = target
    for index, target in {2: 0., 5: 1., 6: 0., 9: 1., 12: 0.}.items():
        pseudo_targets[index] = target
    require(not any(m and p for m, p in zip(manual, pseudo)), 'Fixture masks overlap')
    shape = (2, 1, 2, 4)
    manual_tensor = torch.tensor(manual, dtype=torch.bool).reshape(shape)
    pseudo_tensor = torch.tensor(pseudo, dtype=torch.bool).reshape(shape)
    manual_target = torch.tensor(manual_targets, dtype=torch.float64).reshape(shape)
    pseudo_target = torch.tensor(pseudo_targets, dtype=torch.float64).reshape(shape)
    ignored = ~(manual_tensor | pseudo_tensor)
    proofs = []
    exact_proofs = []
    for epsilon in [0., .5]:
        logits = torch.tensor(values, dtype=torch.float64).reshape(shape).requires_grad_()
        u, p, r = smoothing.smoothed_losses(logits, manual_tensor, manual_target,
                                            pseudo_tensor, pseudo_target, epsilon=epsilon)
        terms = dict(U=u, P=p, R=r, A=old.arm_loss('A', u, p, r),
                     B=old.arm_loss('B', u, p, r))
        analytic = expected(values, manual, manual_targets, pseudo, pseudo_targets, epsilon)
        gradients = {}
        for name, term in terms.items():
            gradient, = torch.autograd.grad(term, logits, retain_graph=True)
            require(bool(torch.isfinite(term)) and bool(torch.isfinite(gradient).all()),
                    'Nonfinite fixture value/gradient: ' + name)
            expected_value, expected_gradient = analytic[name]
            value_error = abs(term.item() - expected_value)
            gradient_error = max(abs(a - b) for a, b in
                                 zip(gradient.reshape(-1).tolist(), expected_gradient))
            require(value_error <= ABSOLUTE_TOLERANCE and gradient_error <= ABSOLUTE_TOLERANCE,
                    'Independent loss/gradient mismatch: ' + name)
            require(bool((gradient[ignored] == 0).all()), 'Ignored-logit gradient: ' + name)
            if name == 'A':
                require(bool((gradient[pseudo_tensor] == 0).all()), 'A used pseudo logits')
            gradients[name] = gradient
            proofs.append(dict(epsilon=epsilon, term=name, observed_value=term.item(),
                               expected_value=expected_value, absolute_value_error=value_error,
                               maximum_absolute_gradient_error=gradient_error,
                               ignored_logit_gradients_exactly_zero=True))
        if epsilon == 0.:
            old_logits = torch.tensor(values, dtype=torch.float64).reshape(shape).requires_grad_()
            old_u, old_p, old_r = old.losses(old_logits, manual_tensor, manual_target,
                                          pseudo_tensor, pseudo_target)
            old_terms = dict(U=old_u, P=old_p, R=old_r,
                             A=old.arm_loss('A', old_u, old_p, old_r),
                             B=old.arm_loss('B', old_u, old_p, old_r))
            for name, term in old_terms.items():
                gradient, = torch.autograd.grad(term, old_logits, retain_graph=True)
                require(torch.equal(terms[name].detach(), term.detach())
                        and torch.equal(gradients[name], gradient),
                        'Epsilon-zero retained value/gradient differs: ' + name)
                exact_proofs.append(dict(term=name, loss_exactly_equal=True,
                                         gradient_exactly_equal=True))
    guard_proofs = []
    guard_logits = torch.tensor(values, dtype=torch.float64).reshape(shape)
    for epsilon in [-.1, .25, 1., float('nan'), float('inf')]:
        try:
            smoothing.smoothed_losses(guard_logits, manual_tensor, manual_target,
                                      pseudo_tensor, pseudo_target, epsilon=epsilon)
        except ValueError:
            guard_proofs.append(dict(kind='invalid_fixed_epsilon', value=repr(epsilon), rejected=True))
        else:
            raise ValueError('Invalid fixed epsilon accepted')
    for kind, index, value in [('manual_fractional_target', 0, .25),
                               ('pseudo_nonfinite_target', 2, float('nan'))]:
        bad_manual, bad_pseudo = manual_target.clone(), pseudo_target.clone()
        target = bad_manual if kind.startswith('manual') else bad_pseudo
        target.reshape(-1)[index] = value
        try:
            smoothing.smoothed_losses(guard_logits, manual_tensor, bad_manual,
                                      pseudo_tensor, bad_pseudo, epsilon=.5)
        except ValueError:
            guard_proofs.append(dict(kind=kind, rejected=True))
        else:
            raise ValueError('Invalid participating hard target accepted')
    after = [record(path) for path in paths]
    require(before == after, 'Fixture sources changed during verification')
    return dict(status='PASS', fixture_kind='artificial-logits-only', shape=list(shape),
                dtype='torch.float64', manual_count=sum(manual), pseudo_count=sum(pseudo),
                ignored_count=sum(not m and not p for m, p in zip(manual, pseudo)),
                hard_targets=[0., 1.], epsilon_half_targets=[.25, .75],
                ignored_target_payload='NaN outside each participating mask',
                coefficients=dict(A=dict(U=1., P=0.), B=dict(U=1., P=.25)),
                independent_formula='max(z,0)+log1p(exp(-abs(z)))-z*t',
                independent_logit_derivative='sigmoid(z)-t, divided by mask count and weighted by objective',
                absolute_tolerance=ABSOLUTE_TOLERANCE, analytic_checks=proofs,
                epsilon_zero_retained_eighth_exact=exact_proofs, guard_checks=guard_proofs,
                source_receipts_before=before, source_receipts_after=after,
                torch_version=torch.__version__, torch_threads=torch.get_num_threads(),
                model_inputs=0, checkpoint_inputs=0, model_constructions=0,
                model_forward_calls=0, optimizer_steps=0, inference_calls=0, network_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, default=HERE / 'smoothing-fixture-receipt.json')
    args = parser.parse_args()
    require(args.receipt.resolve() == HERE / 'smoothing-fixture-receipt.json',
            'Only the assigned fixture receipt may be written')
    require(not args.receipt.exists(), 'Preserve an existing fixture receipt')
    result = verify_fixture()
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], receipt=record(args.receipt),
                         analytic_check_count=len(result['analytic_checks']),
                         exact_eighth_check_count=len(result['epsilon_zero_retained_eighth_exact']),
                         guard_check_count=len(result['guard_checks']))))


if __name__ == '__main__':
    main()
