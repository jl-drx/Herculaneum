"""Independent NumPy-only loss identities and finite differences; synthetic data."""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parent


def sigmoid(z):
    return np.exp(-np.logaddexp(0., -np.asarray(z, dtype=np.float64)))


def bce(z, target):
    return np.logaddexp(0., z)-target*z


def divergence(z, teacher):
    return np.logaddexp(0., z)-np.logaddexp(0., teacher)-sigmoid(teacher)*(z-teacher)


def losses(z, teacher, manual, labels, pseudo, pseudo_labels, coefficient, lam=1.):
    if manual.dtype != np.bool_ or pseudo.dtype != np.bool_ or np.any(manual & pseudo):
        raise ValueError('Disjoint Boolean masks required')
    if not manual.any() or not pseudo.any():
        raise ValueError('Nonempty groups required')
    s = .25+.5*labels
    sp = .25+.5*pseudo_labels
    base = bce(z[manual], s[manual]).mean()+coefficient*bce(z[pseudo], sp[pseudo]).mean()
    preserve = divergence(z[manual], teacher[manual]).mean()+coefficient*divergence(z[pseudo], teacher[pseudo]).mean()
    grad = np.zeros_like(z)
    grad[manual] = ((sigmoid(z[manual])-s[manual])+lam*(sigmoid(z[manual])-sigmoid(teacher[manual])))/manual.sum()
    grad[pseudo] = coefficient*((sigmoid(z[pseudo])-sp[pseudo])+lam*(sigmoid(z[pseudo])-sigmoid(teacher[pseudo])))/pseudo.sum()
    return base+lam*preserve, grad, base, preserve


def finite_difference(function, values, h=1e-5):
    result = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        high, low = values.copy(), values.copy()
        high[index] += h
        low[index] -= h
        result[index] = (function(high)-function(low))/(2*h)
    return result


def main():
    cases = []
    def passed(name, **details):
        cases.append(dict(name=name, status='PASS', **details))

    z = np.array([-3., -.7, .2, 2.4])
    y = np.array([0., 1., 0., 1.])
    s = .25+.5*y
    gradient = sigmoid(z)-s
    numerical = finite_difference(lambda zz: bce(zz, s).sum(), z)
    np.testing.assert_allclose(gradient, numerical, rtol=0, atol=1e-10)
    passed('smoothed_BCE_gradient', maximum_error=float(np.max(np.abs(gradient-numerical))))

    t = np.array([2., -.4, -.5, 1.2])
    numerical = finite_difference(lambda zz: divergence(zz, t).sum(), z)
    np.testing.assert_allclose(sigmoid(z)-sigmoid(t), numerical, rtol=0, atol=1e-10)
    q, p = sigmoid(t), sigmoid(z)
    probability_KL = q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p))
    np.testing.assert_allclose(divergence(z, t), probability_KL, rtol=0, atol=1e-14)
    assert not np.allclose(divergence(z, t), divergence(t, z))
    passed('teacher_to_student_KL_direction_and_gradient')

    values = np.linspace(-2., 2., 30).reshape(2, 1, 3, 5)
    teachers = np.cos(np.arange(30)).reshape(values.shape)
    manual = np.zeros_like(values, dtype=bool)
    pseudo = np.zeros_like(values, dtype=bool)
    manual.reshape(2, -1)[0, [0, 1, 2, 3, 4, 5]] = True
    manual.reshape(2, -1)[1, [0, 1, 2]] = True
    pseudo.reshape(2, -1)[0, [6, 7, 8]] = True
    pseudo.reshape(2, -1)[1, [3, 4, 5, 6]] = True
    labels = np.full_like(values, -1.)
    pseudo_labels = np.full_like(values, -1.)
    labels[manual] = np.arange(manual.sum()) % 2
    pseudo_labels[pseudo] = np.arange(pseudo.sum()) % 2
    for arm, coefficient in [('A', 0.), ('B', .25)]:
        value, analytic, base, preserve = losses(values, teachers, manual, labels, pseudo, pseudo_labels, coefficient)
        numerical = finite_difference(lambda zz: losses(zz, teachers, manual, labels, pseudo, pseudo_labels, coefficient)[0], values)
        np.testing.assert_allclose(analytic, numerical, rtol=0, atol=1e-10)
        active = manual if arm == 'A' else manual | pseudo
        assert np.all(analytic[~active] == 0) and np.all(numerical[~active] == 0)
        changed = values.copy()
        changed[~active] = 1e8
        assert losses(changed, teachers, manual, labels, pseudo, pseudo_labels, coefficient)[0] == value
        passed('masked_'+arm+'_total_gradient_and_unknown_invariance', manual_count=int(manual.sum()),
               pseudo_count=int(pseudo.sum()), maximum_error=float(np.max(np.abs(analytic-numerical))))

    manual_element = bce(values[manual], (.25+.5*labels)[manual])
    pooled = manual_element.mean()
    regional = [bce(values[r][manual[r]], (.25+.5*labels[r])[manual[r]]).mean() for r in range(2)]
    np.testing.assert_allclose(pooled, (6*regional[0]+3*regional[1])/9, rtol=0, atol=1e-15)
    assert abs(pooled-np.mean(regional)) > .01
    grouped = [bce(values[r][manual[r] & (labels[r] == c)], .25+.5*c).mean()
               for r in range(2) for c in [0., 1.]]
    assert abs(pooled-np.mean(grouped)) > .01
    passed('pooled_participant_mean_not_equal_region_or_class_mean', pooled=float(pooled),
           equal_region_mean=float(np.mean(regional)), diagnostic_R=float(np.mean(grouped)))

    initial = losses(teachers, teachers, manual, labels, pseudo, pseudo_labels, .25)
    unregularized = losses(teachers, teachers, manual, labels, pseudo, pseudo_labels, .25, lam=0.)
    assert initial[3] == 0 and np.array_equal(initial[1], unregularized[1])
    assert np.array_equal(divergence(teachers, teachers), np.zeros_like(teachers))
    passed('equal_teacher_zero_KL_and_identical_initial_gradient')

    for coefficient in [0., .25]:
        current = losses(values, teachers, manual, labels, pseudo, pseudo_labels, coefficient)
        initial = losses(teachers, teachers, manual, labels, pseudo, pseudo_labels, coefficient)
        linear = ((sigmoid(teachers[manual])-(.25+.5*labels[manual]))
                  *(values[manual]-teachers[manual])).mean()
        linear += coefficient*((sigmoid(teachers[pseudo])-(.25+.5*pseudo_labels[pseudo]))
                                *(values[pseudo]-teachers[pseudo])).mean()
        np.testing.assert_allclose(current[2]-initial[2], current[3]+linear, rtol=0, atol=1e-14)
        np.testing.assert_allclose(current[0]-initial[0], 2*current[3]+linear, rtol=0, atol=1e-14)
    passed('exact_weighted_base_and_full_delta_decomposition')

    for lam in [0., .7, 1.]:
        smooth_target = .25+.5*y
        q = sigmoid(t)
        r = (smooth_target+lam*q)/(1+lam)
        entropy = np.logaddexp(0., t)-q*t
        left = bce(z, smooth_target)+lam*divergence(z, t)
        right = (1+lam)*bce(z, r)-lam*entropy
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-14)
        optimum = np.log(r)-np.log1p(-r)
        np.testing.assert_allclose((1+lam)*sigmoid(optimum)-smooth_target-lam*q, 0., rtol=0, atol=3e-16)
        assert np.all(bce(optimum, smooth_target) <= bce(t, smooth_target)+1e-14)
    passed('effective_target_identity_and_unconstrained_optimum')

    # A genuine hard-label worsening can coincide with smooth AND full improvement.
    old = np.log(.95/.05)
    new = old-.1*(.95-.75)
    hard_change = float(bce(new, 1.)-bce(old, 1.))
    smooth_change = float(bce(new, .75)-bce(old, .75))
    full_change = smooth_change+float(divergence(new, old))
    assert hard_change > 0 and smooth_change < 0 and full_change < 0
    passed('hard_loss_increase_is_not_smoothed_loss_failure', hard_delta=hard_change,
           smoothed_delta=smooth_change, full_delta=full_change)

    # Conversely, a positive SAME smoothed base delta plus KL cannot improve total.
    wrong = old+.2
    smooth_change = float(bce(wrong, .75)-bce(old, .75))
    full_change = smooth_change+float(divergence(wrong, old))
    assert smooth_change > 0 and full_change >= smooth_change
    passed('positive_matched_smoothed_delta_implies_positive_full_delta',
           smoothed_delta=smooth_change, full_delta=full_change)

    # First fresh AdamW step, weight_decay=0, after a single global norm clip.
    parameter_gradient = np.array([-3., -.002, 1e-12, 0., 2.])
    clip = min(1., 1./(np.linalg.norm(parameter_gradient)+1e-6))
    clipped = parameter_gradient*clip
    beta1, beta2, epsilon, learning_rate = .9, .999, 1e-8, 1e-4
    m = (1-beta1)*clipped
    v = (1-beta2)*clipped**2
    step = -learning_rate*(m/(1-beta1))/(np.sqrt(v/(1-beta2))+epsilon)
    expected = -learning_rate*clipped/(np.abs(clipped)+epsilon)
    np.testing.assert_allclose(step, expected, rtol=0, atol=1e-19)
    assert float(np.dot(parameter_gradient, step)) < 0
    passed('first_fresh_AdamW_clipped_step_is_a_local_descent_direction',
           directional_derivative=float(np.dot(parameter_gradient, step)))

    extreme = np.array([-100., -80., 0., 80., 100.])
    d = divergence(extreme, extreme[::-1])
    assert np.isfinite(d).all() and np.min(d) >= -1e-12
    assert np.all(divergence(extreme, extreme) == 0)
    passed('finite_extreme_logits_without_clamp')

    overlapped = pseudo | manual
    try:
        losses(values, teachers, manual, labels, overlapped, pseudo_labels, .25)
    except ValueError:
        passed('fixture_rejects_overlap')
    else:
        raise AssertionError('Overlapping support accepted')
    receipt = {'status': 'PASS', 'cases': cases, 'case_count': len(cases),
               'numpy_version': np.__version__, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'synthetic_numpy_only': True, 'real_arrays_read': 0, 'torch_imports': 0,
               'model_calls': 0, 'tensor_forward_backward_calls': 0, 'checkpoint_loads': 0, 'network_calls': 0}
    target = ROOT/'numpy-fixtures.json'
    target.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'status': 'PASS', 'cases': len(cases),
                      'receipt_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
