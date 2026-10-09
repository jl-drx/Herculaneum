"""Independent NumPy checks of retained phase21-allocation clipping/AdamW arithmetic.

This module never imports the producer implementation or Torch, decodes a
checkpoint, calls a model, or replays training. Gradients remain producer
autograd outputs: checking their optimizer consequences does not independently
establish convolutional derivatives or the claimed execution history.
"""
import copy
import hashlib
import json
import math

import numpy as np


PARAMETER_ORDER = [
    dict(name='stem.weight', shape=[8, 3, 3, 3, 3], numel=648),
    dict(name='stem.bias', shape=[8], numel=8),
    dict(name='projection.weight', shape=[32, 16, 3, 3], numel=4608),
    dict(name='projection.bias', shape=[32], numel=32),
]
TRACE_KEYS = (
    'parameters_before', 'parameters_after',
    'gradients_before_clip', 'gradients_after_clip',
    'exp_avg_before', 'exp_avg_after',
    'exp_avg_sq_before', 'exp_avg_sq_after',
)
EXPECTED_CONTRACT = dict(
    parameter_order=PARAMETER_ORDER, parameter_count=5296,
    optimizer_class='torch.optim.AdamW', learning_rate=.001,
    betas=[.9, .999], eps=1e-8, weight_decay=0, foreach=False,
    fused=False, optimizer_membership_exact=True, original_backbone_members=0,
    clip_foreach=False, clip_epsilon=1e-6, step_counters=list(range(1, 33)),
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact_float32(actual, expected, label):
    require(actual.dtype == expected.dtype == np.dtype('float32')
            and actual.shape == expected.shape
            and np.array_equal(actual.view(np.uint32), expected.view(np.uint32)),
            'Optimizer bitwise mismatch: ' + label)


def scalar_tolerance(before, after, expected):
    """Frozen absolute floor plus eight largest corresponding FP32 ULPs."""
    spacings = [np.abs(np.spacing(np.asarray(x, dtype=np.float32))).astype(np.float64)
                for x in (before, after, expected)]
    return 2e-8 + 8 * np.maximum.reduce(spacings)


def check_update(actual, expected, before, label):
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    tolerance = scalar_tolerance(before, actual, expected)
    require(np.all(np.isfinite(error)) and np.all(error <= tolerance),
            'Optimizer arithmetic mismatch: ' + label)
    return dict(maximum_absolute_error=float(error.max(initial=0)),
                maximum_tolerance_fraction=float((error / tolerance).max(initial=0)),
                scalar_comparisons=int(error.size))


def global_norm(gradient):
    """The four FP32 tensor norms, followed by their FP32 global L2 norm."""
    norms = []
    offset = 0
    for tensor in PARAMETER_ORDER:
        block = gradient[offset:offset + tensor['numel']]
        norms.append(np.sqrt(np.sum(np.square(block, dtype=np.float32), dtype=np.float32)))
        offset += tensor['numel']
    return np.sqrt(np.sum(np.square(np.asarray(norms, dtype=np.float32)), dtype=np.float32))


def clipping_coefficient(norm):
    return np.minimum(np.float32(1), np.float32(1) / (np.float32(norm) + np.float32(1e-6)))


def adamw_step(parameters, first_moment, second_moment, clipped_gradient, step):
    """Independent nonfused FP32 recurrence; scalar bias corrections are FP64.

    No weight-decay term is present because the frozen decay is exactly zero.
    Each array operation rounds to FP32. The numerical contract allows bounded
    differences from producer fused scalar multiply-adds inside CPU kernels.
    """
    first = first_moment + (clipped_gradient - first_moment) * np.float32(1 - .9)
    second = second_moment * np.float32(.999)
    second = second + np.square(clipped_gradient, dtype=np.float32) * np.float32(1 - .999)
    denominator = np.sqrt(second) / np.float32(math.sqrt(1 - .999 ** step))
    denominator = denominator + np.float32(1e-8)
    parameter = parameters - (first / denominator) * np.float32(.001 / (1 - .9 ** step))
    return parameter, first, second


def _finite_scalar(value, label):
    require(isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value), 'Finite optimizer scalar required: ' + label)
    return float(value)


def verify_arrays(traces, contract, steps):
    """Verify already-loaded arrays; caller owns authorization and input pins."""
    require(set(traces) == set(TRACE_KEYS), 'Optimizer trace keys differ')
    for name in TRACE_KEYS:
        value = traces[name]
        require(isinstance(value, np.ndarray) and value.dtype == np.dtype('float32')
                and value.shape == (32, 5296) and np.isfinite(value).all(),
                'Optimizer trace dtype/shape/finiteness differs: ' + name)
    for key, expected in EXPECTED_CONTRACT.items():
        require(key in contract and contract[key] == expected,
                'Optimizer contract differs: ' + key)
        if isinstance(expected, bool):
            require(type(contract[key]) is bool, 'Optimizer boolean differs: ' + key)
    require(len(steps) == 32 and [row['step'] for row in steps] == list(range(1, 33)),
            'Optimizer row steps differ')
    for stem in ('parameters', 'exp_avg', 'exp_avg_sq'):
        exact_float32(traces[stem + '_before'][1:], traces[stem + '_after'][:-1],
                      stem + ' adjacent boundaries')
    zeros = np.zeros(5296, dtype=np.float32)
    exact_float32(traces['exp_avg_before'][0], zeros, 'initial first moment')
    exact_float32(traces['exp_avg_sq_before'][0], zeros, 'initial second moment')
    exact_float32(traces['parameters_before'][0, 656:], zeros[656:], 'initial zero projection')
    require(np.all(traces['exp_avg_sq_before'] >= 0)
            and np.all(traces['exp_avg_sq_after'] >= 0), 'Negative second moment')
    require(np.all(traces['gradients_before_clip'][0, :656] == 0)
            and np.any(traces['gradients_before_clip'][0, 656:] != 0),
            'First-step hidden/head gradient path differs')
    require(np.any(traces['gradients_before_clip'][1, :656] != 0),
            'Second-step hidden gradients inactive')

    maxima = {}
    norm_error = coefficient_error = 0.
    clipped_steps = 0
    checks = 0
    for index, row in enumerate(steps):
        gradient = traces['gradients_before_clip'][index]
        expected_norm = float(global_norm(gradient))
        norm = _finite_scalar(row['preclip_global_gradient_norm'], 'global norm')
        require(norm > 0 and math.isfinite(expected_norm), 'Nonpositive/nonfinite global norm')
        difference = abs(norm - expected_norm)
        require(difference <= 1e-6 + 2e-7 * abs(expected_norm), 'Global gradient norm differs')
        norm_error = max(norm_error, difference)
        coefficient = _finite_scalar(row['clipping_coefficient'], 'clipping coefficient')
        expected_coefficient = clipping_coefficient(expected_norm)
        require(0 < coefficient <= 1, 'Clipping coefficient outside (0,1]')
        difference = abs(coefficient - float(expected_coefficient))
        require(difference <= 2e-7, 'Clipping coefficient differs')
        coefficient_error = max(coefficient_error, difference)
        require(coefficient == float(clipping_coefficient(norm)),
                'Recorded clipping coefficient does not follow its recorded FP32 norm')
        exact_float32(traces['gradients_after_clip'][index],
                      gradient * np.float32(coefficient), 'saved gradient multiplication')
        clipped = gradient * expected_coefficient
        expected_parameter, expected_first, expected_second = adamw_step(
            traces['parameters_before'][index], traces['exp_avg_before'][index],
            traces['exp_avg_sq_before'][index], clipped, index + 1)
        for key, expected in [('gradients', clipped), ('parameters', expected_parameter),
                              ('exp_avg', expected_first), ('exp_avg_sq', expected_second)]:
            before_key = key + ('_before_clip' if key == 'gradients' else '_before')
            after_key = key + ('_after_clip' if key == 'gradients' else '_after')
            checked = check_update(traces[after_key][index], expected,
                                   traces[before_key][index], f'{key} step {index + 1}')
            old = maxima.setdefault(key, dict(maximum_absolute_error=0., maximum_tolerance_fraction=0.))
            for field in old:
                old[field] = max(old[field], checked[field])
            checks += checked['scalar_comparisons']
        require(np.any(traces['parameters_after'][index] != traces['parameters_before'][index]),
                'Parameter update is identically zero')
        clipped_steps += int(coefficient < 1)
    return dict(status='PASS', steps=32, ordered_parameters=5296,
                verified_scalar_update_comparisons=checks,
                exact_adjacent_boundary_scalars=31 * 5296 * 3,
                maximum_gradient_norm_absolute_error=norm_error,
                maximum_clipping_coefficient_absolute_error=coefficient_error,
                update_errors=maxima, clipped_steps=clipped_steps,
                initial_parameters_sha256=hashlib.sha256(traces['parameters_before'][0].tobytes()).hexdigest(),
                final_parameters_sha256=hashlib.sha256(traces['parameters_after'][-1].tobytes()).hexdigest(),
                scope='Independent clipping and AdamW arithmetic on author-produced autograd gradients; '
                      'no independent convolutional derivative or execution-history proof',
                model_calls=0, training_replays=0, checkpoint_decodes=0)


def semantic_state_hash(vector):
    """Reconstruct producer state-dict identity from retained ordered FP32 vectors."""
    values, offset = {}, 0
    for spec in PARAMETER_ORDER:
        count = spec['numel']
        values[spec['name']] = vector[offset:offset+count].reshape(spec['shape'])
        offset += count
    require(offset == 5296 and vector.shape == (5296,), 'State-vector shape differs')
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        digest.update(json.dumps([name, list(value.shape), 'torch.float32'], separators=(',', ':')).encode() + b'\0')
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def verify_trace(review, training_result, steps):
    """Pinned-file adapter for review_io.Review; call only after root GO."""
    contract = review.read_json(training_result['optimizer_contract'])
    pin = training_result['optimizer_traces']
    path = review.verify(pin)
    require(path.suffix == '.npz', 'Optimizer trace must be a NumPy archive')
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(TRACE_KEYS), 'Optimizer archive keys differ')
        traces = {key: archive[key] for key in TRACE_KEYS}
    review.array_reads += len(traces)
    report = verify_arrays(traces, contract, steps)
    require(semantic_state_hash(traces['parameters_before'][0]) == training_result['initial_adapter_state_sha256'],
            'Initial semantic state/trace identity differs')
    require(semantic_state_hash(traces['parameters_after'][-1]) == training_result['final_adapter_state_sha256'] ==
            training_result['checkpoint_reload']['state_sha256'], 'Final semantic state/trace/reload identity differs')
    maximum_norm_error = 0.
    for index, row in enumerate(steps):
        gradient = traces['gradients_before_clip'][index].astype(np.float64)
        delta = traces['parameters_after'][index].astype(np.float64) - traces['parameters_before'][index].astype(np.float64)
        expected = dict(stem_gradient_l2=float(np.sqrt(np.square(gradient[:656]).sum())),
                        head_gradient_l2=float(np.sqrt(np.square(gradient[656:]).sum())),
                        parameter_update_l2=float(np.sqrt(np.square(delta).sum())))
        for key, value in expected.items():
            error = abs(row[key] - value)
            require(error <= 1e-12, 'FP64 trace gradient/update norm differs: ' + key)
            maximum_norm_error = max(maximum_norm_error, error)
    report.update(semantic_state_trace_linkage_exact=True, maximum_reported_norm_error=maximum_norm_error,
                  initial_semantic_state_sha256=semantic_state_hash(traces['parameters_before'][0]),
                  final_semantic_state_sha256=semantic_state_hash(traces['parameters_after'][-1]))
    review.verify(pin)
    return report


def _synthetic_trace():
    """Scalar FP64 oracle with explicit FP32 rounding, independent of vector step.

    Fixed gradients cover signed values, un/clipped updates, momentum after a
    zero gradient, tiny epsilon-dominated coordinates, and 32 bias corrections.
    This is synthetic optimizer arithmetic, not a model/training replay.
    """
    traces = {name: np.zeros((32, 5296), dtype=np.float32) for name in TRACE_KEYS}
    parameters = np.asarray([((-1) ** j) * (j % 13 + 1) / 128 for j in range(5296)], dtype=np.float32)
    parameters[656:] = 0
    first = np.zeros(5296, dtype=np.float32)
    second = first.copy()
    steps = []
    for index in range(32):
        gradient = np.asarray([((-1) ** (j + index // 4)) * (1 / 8 if index % 3 == 2 else 1 / 128)
                               for j in range(5296)], dtype=np.float32)
        gradient[::7] = 0
        gradient[13] = np.float32(1e-12)
        gradient[700] = np.float32(-1e-12)
        if index % 5 == 0:
            gradient[29] = 0
        if index == 0:
            gradient[:656] = 0
        norms = []
        offset = 0
        for count in (648, 8, 4608, 32):
            norms.append(np.float32(math.sqrt(math.fsum(float(g) ** 2 for g in gradient[offset:offset + count]))))
            offset += count
        norm = np.float32(math.sqrt(math.fsum(float(n) ** 2 for n in norms)))
        coefficient = np.float32(min(1., 1. / float(np.float32(float(norm) + float(np.float32(1e-6))))))
        clipped = np.asarray([np.float32(float(g) * float(coefficient)) for g in gradient], dtype=np.float32)
        for name, vector in [('parameters_before', parameters), ('exp_avg_before', first),
                             ('exp_avg_sq_before', second), ('gradients_before_clip', gradient),
                             ('gradients_after_clip', clipped)]:
            traces[name][index] = vector
        for j, g in enumerate(clipped):
            first[j] = np.float32(float(first[j]) +
                                  float(np.float32(float(g) - float(first[j]))) * float(np.float32(.1)))
            second[j] = np.float32(float(np.float32(float(second[j]) * float(np.float32(.999)))) +
                                   float(g) * float(g) * float(np.float32(.001)))
            denom = np.float32(float(np.float32(math.sqrt(float(second[j])))) /
                               float(np.float32(math.sqrt(1 - .999 ** (index + 1)))))
            denom = np.float32(float(denom) + float(np.float32(1e-8)))
            parameters[j] = np.float32(float(parameters[j]) -
                                       float(first[j]) / float(denom) * (.001 / (1 - .9 ** (index + 1))))
        traces['parameters_after'][index] = parameters
        traces['exp_avg_after'][index] = first
        traces['exp_avg_sq_after'][index] = second
        steps.append(dict(step=index + 1, preclip_global_gradient_norm=float(norm),
                          clipping_coefficient=float(coefficient)))
    return traces, copy.deepcopy(EXPECTED_CONTRACT), steps


def fixture_tests():
    """Run positive analytic fixtures and deliberate corruption rejections."""
    fixtures = []
    traces, contract, rows = _synthetic_trace()
    report = verify_arrays(traces, contract, rows)
    require(0 < report['clipped_steps'] < 32, 'Fixture must exercise both clipping branches')
    fixtures.append('32-step signed/clipped/unclipped/zero/tiny-gradient scalar oracle')

    # Direct first-update identities and epsilon placement, independent of traces.
    p = np.asarray([0., .25, -.25, 1.], dtype=np.float32)
    g = np.asarray([0., .5, -.5, 1e-12], dtype=np.float32)
    new, first, second = adamw_step(p, np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32), g, 1)
    np.testing.assert_allclose(first, np.asarray([0., .05, -.05, 1e-13]), rtol=1e-6, atol=1e-20)
    np.testing.assert_allclose(second, np.asarray([0., .00025, .00025, 1e-27]), rtol=1e-6, atol=1e-34)
    analytic = p.astype(np.float64) - .001 * g.astype(np.float64) / (np.abs(g.astype(np.float64)) + 1e-8)
    require(np.all(np.abs(new.astype(np.float64) - analytic) <= 2e-8 + 8 * np.abs(np.spacing(p))),
            'Analytic first-step epsilon/bias correction differs')
    fixtures.append('analytic first-step zero/sign/bias-correction/epsilon identities')

    grouped = np.zeros(5296, dtype=np.float32)
    grouped[647] = 3
    grouped[648] = -4
    require(global_norm(grouped) == np.float32(5), 'Grouped 3-4-5 norm differs')
    require(clipping_coefficient(np.float32(.5)) == 1
            and clipping_coefficient(np.float32(1)) == np.float32(1 / float(np.float32(1 + 1e-6)))
            and clipping_coefficient(np.float32(1)) < 1,
            'Norm boundary/epsilon clipping differs')
    fixtures.append('grouped 3-4-5 norm and epsilon clipping at norm one')

    constant = np.asarray([.125], dtype=np.float32)
    parameter = np.asarray([1.], dtype=np.float32)
    first = np.zeros(1, dtype=np.float32)
    second = first.copy()
    for step in range(1, 33):
        parameter, first, second = adamw_step(parameter, first, second, constant, step)
    check_update(first, np.asarray([(1 - .9 ** 32) * .125], dtype=np.float32),
                 np.zeros(1, dtype=np.float32), 'constant-gradient analytic first moment')
    check_update(second, np.asarray([(1 - .999 ** 32) * .125 ** 2], dtype=np.float32),
                 np.zeros(1, dtype=np.float32), 'constant-gradient analytic second moment')
    check_update(parameter, np.asarray([1 - 32 * .001 * .125 / (.125 + 1e-8)], dtype=np.float32),
                 np.ones(1, dtype=np.float32), 'constant-gradient analytic parameter')
    fixtures.append('32-step constant-gradient closed-form moments and parameter')

    def rejected(label, mutate):
        changed = {key: value.copy() for key, value in traces.items()}
        changed_contract = copy.deepcopy(contract)
        changed_rows = copy.deepcopy(rows)
        mutate(changed, changed_contract, changed_rows)
        try:
            verify_arrays(changed, changed_contract, changed_rows)
        except AssertionError:
            fixtures.append(label)
        else:
            raise AssertionError('Corruption was accepted: ' + label)

    rejected('reject float64 traces', lambda t, c, r: t.__setitem__('parameters_before', t['parameters_before'].astype(np.float64)))
    rejected('reject wrong shape', lambda t, c, r: t.__setitem__('parameters_before', t['parameters_before'][:, :-1]))
    rejected('reject nonfinite gradient', lambda t, c, r: t['gradients_before_clip'].__setitem__((8, 0), np.nan))
    rejected('reject altered optimizer rate', lambda t, c, r: c.__setitem__('learning_rate', .002))
    rejected('reject fused optimizer', lambda t, c, r: c.__setitem__('fused', True))
    rejected('reject parameter ordering change', lambda t, c, r: c['parameter_order'].reverse())
    rejected('reject counter discontinuity', lambda t, c, r: c['step_counters'].__setitem__(8, 8))
    rejected('reject row step discontinuity', lambda t, c, r: r[8].__setitem__('step', 8))
    rejected('reject altered parameter update', lambda t, c, r: t['parameters_after'].__setitem__((31, 20), t['parameters_after'][31, 20] + .001))
    rejected('reject altered first moment', lambda t, c, r: t['exp_avg_after'].__setitem__((31, 20), t['exp_avg_after'][31, 20] + .001))
    rejected('reject altered second moment', lambda t, c, r: t['exp_avg_sq_after'].__setitem__((31, 20), t['exp_avg_sq_after'][31, 20] + .001))
    rejected('reject signed-zero continuity break', lambda t, c, r: t['exp_avg_before'].__setitem__((1, 0), np.float32(-0.)))
    rejected('reject nonzero initial first moment', lambda t, c, r: t['exp_avg_before'].__setitem__((0, 0), np.float32(1e-12)))
    rejected('reject nonzero initial second moment', lambda t, c, r: t['exp_avg_sq_before'].__setitem__((0, 0), np.float32(1e-12)))
    rejected('reject nonzero initial projection', lambda t, c, r: t['parameters_before'].__setitem__((0, 656), np.float32(1e-12)))
    rejected('reject clipped-gradient mutation', lambda t, c, r: t['gradients_after_clip'].__setitem__((2, 20), 0.))
    rejected('reject norm beyond frozen tolerance', lambda t, c, r: r[2].__setitem__('preclip_global_gradient_norm', r[2]['preclip_global_gradient_norm'] + 1e-4))
    rejected('reject coefficient beyond frozen tolerance', lambda t, c, r: r[2].__setitem__('clipping_coefficient', r[2]['clipping_coefficient'] + 1e-5))
    rejected('reject unexpected initial hidden gradient', lambda t, c, r: t['gradients_before_clip'].__setitem__((0, 0), np.float32(1e-12)))
    rejected('reject inactive second-step hidden gradient', lambda t, c, r: t['gradients_before_clip'].__setitem__((1, slice(0, 656)), 0.))

    unit = np.ones(1, dtype=np.float32)
    spacing = np.spacing(unit)
    check_update(unit + 8 * spacing, unit, unit, 'eight ULP acceptance')
    try:
        check_update(unit + 9 * spacing, unit, unit, 'nine ULP rejection')
    except AssertionError:
        fixtures.append('frozen eight-ULP boundary acceptance and nine-ULP rejection')
    else:
        raise AssertionError('Nine-ULP corruption was accepted')
    zero = np.zeros(1, dtype=np.float32)
    check_update(np.asarray([2e-8], dtype=np.float32), zero, zero, 'absolute floor acceptance')
    try:
        check_update(np.asarray([2.01e-8], dtype=np.float32), zero, zero, 'absolute floor rejection')
    except AssertionError:
        fixtures.append('frozen absolute-floor boundary acceptance and rejection')
    else:
        raise AssertionError('Absolute-floor corruption was accepted')
    return dict(status='PASS', synthetic_only=True, fixtures=fixtures, fixture_count=len(fixtures),
                scalar_oracle_verification=report, scientific_array_reads=0,
                torch_imports=0, model_calls=0, checkpoint_decodes=0, training_replays=0)


if __name__ == '__main__':
    print(json.dumps(fixture_tests(), indent=2, sort_keys=True, allow_nan=False))
