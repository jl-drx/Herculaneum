"""Reconstruct the declared preprocessing and masks without historical helpers.

Robust-normalization specification: retained MIT-licensed villa source at
vendor/villa-ink-baseline/vesuvius/image_proc/intensity/normalization.py.
The full acquisition canvas is observed; no masked-normalization branch is used.
"""
import hashlib
import json
import numpy as np
from numerical_kernels import d4, require


def normalize_patch(image):
    array = np.array(image, dtype=np.float32, copy=True)
    require(array.size > 0 and np.isfinite(array).all(), "Finite nonempty normalization input required")
    values = array.ravel()
    original_std = float(np.std(values))
    lower, upper = [float(np.percentile(values, p)) for p in (1., 99.)]
    np.clip(array, lower, upper, out=array)
    median = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - median)))
    if not np.isfinite(scale) or scale < 1e-6:
        candidates = [abs(original_std), abs(upper - lower) / 2.]
        scale = next((v for v in candidates if np.isfinite(v) and v >= 1e-6), 1.)
    array -= median
    array /= scale
    np.nan_to_num(array, copy=False, nan=0., posinf=0., neginf=0.)
    return array


def fingerprint(values):
    digest = hashlib.sha256()
    names = {np.dtype('float32'): 'torch.float32', np.dtype('bool'): 'torch.bool'}
    for name, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        require(array.dtype in names, "Unexpected fingerprint dtype")
        digest.update(json.dumps([name, list(array.shape), names[array.dtype]], separators=(',', ':')).encode() + b'\0')
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_batches(base_plan, schedule, load_array):
    regions = []
    require(len(base_plan['adaptation']) == 2, "Exactly two adaptation regions required")
    for case in base_plan['adaptation']:
        require(case['physical_group'] == 'PHerc0139', "Unexpected adaptation specimen")
        arrays = {k: load_array(v) for k, v in case['inputs'].items()}
        surface = arrays['surface']
        observed = arrays['acquisition_observed_mask']
        require(surface.shape == (21, 384, 384) and surface.dtype == np.uint8, "Surface shape or dtype changed")
        require(observed.shape == (384, 384) and observed.dtype == np.bool_ and observed.all(), "Acquisition support changed")
        for k in ('inklabels', 'supervision_mask', 'validation_mask'):
            require(arrays[k].shape == (384, 384) and arrays[k].dtype == np.uint8 and
                    np.isin(arrays[k], [0, 1]).all(), "Annotation contract changed")
        manual = observed & (arrays['supervision_mask'] == 1) & (arrays['validation_mask'] == 0)
        score, evaluated = load_array(case['teacher_scores']), load_array(case['teacher_evaluated'])
        require(score.shape == evaluated.shape == (384, 384) and score.dtype == np.float32 and
                evaluated.dtype == np.bool_, "Pseudo-source contract changed")
        finite = np.isfinite(score)
        require(np.all((score[finite] >= 0) & (score[finite] <= 1)), "Pseudo probabilities invalid")
        pseudo = observed & evaluated & finite & ~manual
        manual_y = np.full((384, 384), -1., dtype=np.float32)
        pseudo_y = np.full((384, 384), -1., dtype=np.float32)
        manual_y[manual] = arrays['inklabels'][manual]
        pseudo_y[pseudo] = (score[pseudo] >= .5).astype(np.float32)
        regions.append((surface, manual, manual_y, pseudo, pseudo_y))
    require([r['step'] for r in schedule] == list(range(1, 33)), "Exactly 32 ordered scheduled batches required")
    result = []
    for step in schedule:
        require([p['region'] for p in step['patches']] == [0, 1], "Batch region order changed")
        fields = {k: [] for k in ('input', 'manual_mask', 'manual_hard_target', 'pseudo_mask', 'pseudo_hard_target')}
        exposures = []
        for patch in step['patches']:
            r = patch['region']
            surface, manual, manual_y, pseudo, pseudo_y = regions[r]
            x, y, w, h = patch['bbox']
            require(w == h == 128 and 0 <= x <= 256 and 0 <= y <= 256 and x % 64 == y % 64 == 0 and
                    patch['bbox'] in base_plan['exposure'][r]['eligible_tiles'], "Frozen tile changed")
            cut = np.s_[y:y+h, x:x+w]
            code = patch['augmentation']
            image = normalize_patch(surface[2:19, y:y+h, x:x+w])
            fields['input'].append(d4(image, code)[None])
            for k, value in [('manual_mask', manual), ('manual_hard_target', manual_y),
                             ('pseudo_mask', pseudo), ('pseudo_hard_target', pseudo_y)]:
                fields[k].append(d4(value[cut], code)[None])
            exposure = {'region': r}
            for name, mask, target in [('manual', manual, manual_y), ('pseudo', pseudo, pseudo_y)]:
                exposure[name] = {label: int((mask[cut] & (target[cut] == number)).sum())
                                  for label, number in [('positive', 1), ('negative', 0)]}
            require(min(exposure['manual'].values()) >= 256, "Manual class participation gate changed")
            exposures.append(exposure)
        batch = {k: np.stack(v) for k, v in fields.items()}
        digest = fingerprint(batch)
        del batch['input']
        require(not (batch['manual_mask'] & batch['pseudo_mask']).any(), "Supports overlap")
        result.append({'step': step['step'], 'arrays': batch, 'input_target_sha256': digest,
                       'patches': step['patches'], 'exposures': exposures})
    return result


def fixture_checks():
    constant = normalize_patch(np.full((2, 3, 3), 17, dtype=np.uint8))
    require(constant.dtype == np.float32 and np.array_equal(constant, np.zeros_like(constant)), "Constant normalization differs")
    input_array = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    normalized = normalize_patch(input_array)
    hand = np.array([.04, 1., 2., 3., 3.96], dtype=np.float32)
    hand -= 2.
    hand /= 1.4826
    require(np.array_equal(normalized.view(np.uint32), hand.view(np.uint32)), "Median/MAD fixture differs")
    require(np.array_equal(input_array, np.arange(5, dtype=np.uint8)), "Normalization mutated source")
    value = {'z': np.array([[[0., -1.]]], dtype=np.float32), 'a': np.array([[[True, False]]], dtype=np.bool_)}
    require(fingerprint(value) == fingerprint(dict(reversed(list(value.items())))), "Fingerprint depends on dictionary insertion order")
    return {'status': 'PASS', 'constant_normalization_cases': 1, 'hand_computed_MAD_cases': 1,
            'normalization_source_immutability_cases': 1, 'fingerprint_order_cases': 1,
            'actual_retained_array_reads': 0}
