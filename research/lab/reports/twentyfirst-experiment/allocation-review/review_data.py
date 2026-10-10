"""Separate reconstruction of authorized historical batches and tiled development."""
import math
import numpy as np
from review_math import bit_equal, d4, require, rank_metrics, threshold_metrics
from review_features import ARMS, array_hash, features, fingerprint, normalize_patch

CUTOFF = 0.7558947205543518
BOXES = [[x, y, 128, 128] for y in range(0, 257, 64) for x in range(0, 257, 64)]


def training_batches(plan, review):
    base = review.read_json(plan['training_base_plan'])
    require([case['id'] for case in base['adaptation']] == ['pherc0139-w016-rank01', 'pherc0139-w016-rank02'] and
            all(case['physical_group'] == 'PHerc0139' for case in base['adaptation']), 'Training parent/region scope differs')
    require(base['schedule'] == plan['schedule'] and base['historical_initial_state_sha256'] == plan['backbone_state_sha256'],
            'Historical schedule/initializer lineage differs')
    require(base['initializer'] == plan['backbone_checkpoint_lineage'] and
            set(plan['teacher_caches']) == set(plan['historical_step_logs']) == {'A_preserve', 'B_preserve'},
            'Teacher/initializer identity differs')
    for arm in ('A_preserve', 'B_preserve'):
        require(plan['teacher_caches'][arm] == base['historical'][arm]['teacher'] and
                plan['historical_step_logs'][arm] == base['historical'][arm]['steps'] and
                plan['historical_training_results'][arm] == base['historical'][arm]['result'],
                'Historical cache/source plan linkage differs')
        historical = review.read_json(plan['historical_training_results'][arm])
        require(historical['status'] == 'PASS' and historical['initial_state_sha256'] == plan['backbone_state_sha256'] and
                historical['training']['preupdate_logits']['teacher'] == plan['teacher_caches'][arm] and
                historical['training']['steps_log'] == plan['historical_step_logs'][arm],
                'Historical teacher execution receipt differs')
    regions = []
    for case in base['adaptation']:
        arrays = {key: review.array(pin) for key, pin in case['inputs'].items()}
        surface, observed = arrays['surface'], arrays['acquisition_observed_mask']
        require(surface.shape == (21, 384, 384) and surface.dtype == np.uint8 and
                observed.shape == (384, 384) and observed.dtype == np.bool_ and observed.all(),
                'Training acquisition schema differs')
        for key in ('inklabels', 'supervision_mask', 'validation_mask'):
            require(arrays[key].shape == (384, 384) and arrays[key].dtype == np.uint8 and
                    np.isin(arrays[key], [0, 1]).all(), 'Training annotation schema differs')
        manual = observed & (arrays['supervision_mask'] == 1) & (arrays['validation_mask'] == 0)
        # Pseudo inputs are reconstructed solely to authenticate the historical full-batch hash.
        scores = review.array(case['teacher_scores'], shape=(384, 384), dtype=np.float32)
        evaluated = review.array(case['teacher_evaluated'], shape=(384, 384), dtype=np.bool_)
        finite = np.isfinite(scores)
        require(((scores[finite] >= 0) & (scores[finite] <= 1)).all(), 'Pseudo identity probability range differs')
        pseudo = observed & evaluated & finite & ~manual
        manual_target, pseudo_target = [np.full((384, 384), -1., dtype=np.float32) for _ in range(2)]
        manual_target[manual] = arrays['inklabels'][manual]
        pseudo_target[pseudo] = (scores[pseudo] >= .5).astype(np.float32)
        regions.append((surface, manual, manual_target, pseudo, pseudo_target, observed & (arrays['validation_mask'] == 1)))
    schedule = review.read_json(plan['schedule'])
    histories = {arm: review.read_jsonl(pin) for arm, pin in plan['historical_step_logs'].items()}
    require(set(histories) == {'A_preserve', 'B_preserve'} and all(len(rows) == 32 for rows in histories.values()),
            'Both historical step records required')
    teachers = [review.array(pin, shape=(32, 2, 1, 128, 128), dtype=np.float32, finite=True)
                for pin in plan['teacher_caches'].values()]
    require(len(teachers) == 2 and bit_equal(*teachers), 'Teacher caches differ')
    require([row['step'] for row in schedule] == list(range(1, 33)), 'Scheduled step order differs')
    batches = []
    for index, step in enumerate(schedule):
        require([patch['region'] for patch in step['patches']] == [0, 1], 'Batch region order differs')
        fields = {name: [] for name in ('input', 'manual_mask', 'manual_hard_target', 'pseudo_mask', 'pseudo_hard_target')}
        exposures = []
        retired_fields, source_fields = [], []
        for patch in step['patches']:
            region = patch['region']
            surface, manual, target, pseudo, pseudo_target, retired = regions[region]
            x, y, width, height = patch['bbox']
            require(width == height == 128 and patch['bbox'] in base['exposure'][region]['eligible_tiles'],
                    'Training tile differs from frozen schedule')
            code = patch['augmentation']
            retired_fields.append(d4(retired[y:y+height, x:x+width], code)[None])
            source_fields.append(d4(np.arange(384*384, dtype=np.int32).reshape(384,384)[y:y+height,x:x+width], code)[None])
            fields['input'].append(d4(normalize_patch(surface[2:19, y:y+height, x:x+width]), code)[None])
            exposure = dict(region=region)
            for name, value in [('manual_mask', manual), ('manual_hard_target', target),
                                ('pseudo_mask', pseudo), ('pseudo_hard_target', pseudo_target)]:
                fields[name].append(d4(value[y:y+height, x:x+width], code)[None])
            for name, mask, hard in [('manual', manual, target), ('pseudo', pseudo, pseudo_target)]:
                selected, labels = mask[y:y+height, x:x+width], hard[y:y+height, x:x+width]
                exposure[name] = {key: int((selected & (labels == value)).sum())
                                  for key, value in [('positive', 1), ('negative', 0)]}
            require(min(exposure['manual'].values()) >= 256, 'Manual participation gate differs')
            exposures.append(exposure)
        arrays = {key: np.stack(values) for key, values in fields.items()}
        retired_batch = np.stack(retired_fields)
        source_batch = np.stack(source_fields)
        require(retired_batch.any() and not (retired_batch & arrays['manual_mask']).any(), 'Retired support empty or overlapping manual')
        identity = fingerprint(arrays)
        row = dict(step=index+1, patches=step['patches'], exposures=exposures, input_target_sha256=identity,
                   input_sha256=array_hash(arrays['input']),
                   manual_sha256=fingerprint({key: arrays[key] for key in ('manual_mask', 'manual_hard_target')}),
                   feature_sha256={arm: array_hash(features(arrays['input'], arm)) for arm in ARMS},
                   mask=arrays['manual_mask'], target=arrays['manual_hard_target'], retired_mask=retired_batch,
                   retired_sha256=array_hash(retired_batch), retired_pixels=int(retired_batch.sum()),
                   participation_sha256=fingerprint({'manual_mask':arrays['manual_mask'],'retired_mask':retired_batch}),
                   source_index_sha256=array_hash(source_batch), source_index=source_batch)
        for history in histories.values():
            for key in ('step', 'patches', 'exposures', 'input_target_sha256'):
                require(history[index][key] == row[key], 'Independent historical batch mismatch: ' + key)
        batches.append(row)
    support = review.read_json(plan['preservation_support'])
    require(support['broader_valid_support_ready'] is False and support['geometry_validity_status'] == 'not_acquired', 'Supplied reference support misrepresented as geometry valid')
    for name, expected in [('manual', np.stack([batch['mask'] for batch in batches])),
                           ('retired_validation', np.stack([batch['retired_mask'] for batch in batches])),
                           ('source_index', np.stack([batch['source_index'] for batch in batches]))]:
        item = support['batch_arrays'][name]
        pin = {key: item[key] for key in ('path','size_bytes','sha256')}
        actual = review.array(pin, shape=expected.shape, dtype=expected.dtype)
        require(np.array_equal(actual, expected) and item['raw_sha256'] == array_hash(expected), 'Sealed support/source index differs: ' + name)
    return batches, teachers[0]


def hann_weights():
    # Separate analytic construction. The frozen comparison permits normal FP32 cosine differences.
    coordinate = np.arange(128, dtype=np.float64)
    window = (.5 - .5 * np.cos(2. * math.pi * coordinate / 127.)).astype(np.float32)
    weight = window[:, None] * window[None, :]
    weight /= weight.max()
    return np.maximum(weight, np.float32(.001))


def blend_logits(logits):
    require(logits.shape == (25, 128, 128) and logits.dtype == np.float32 and np.isfinite(logits).all(),
            'Development tile logits differ')
    weight = hann_weights()
    numerator = np.zeros((384, 384), dtype=np.float32)
    denominator = np.zeros_like(numerator)
    for tile, (x, y, width, height) in zip(logits, BOXES):
        probability = np.exp(-np.logaddexp(0., -tile.astype(np.float64))).astype(np.float32)
        numerator[y:y+height, x:x+width] += probability * weight
        denominator[y:y+height, x:x+width] += weight
    require((denominator > 0).all(), 'Coverage contains unscored locations')
    return numerator / denominator, denominator


def development_case(case, outputs, review):
    metadata = review.read_json(case['baseline_metadata'])
    baseline = review.array(metadata['logits'], shape=(25, 128, 128), dtype=np.float32, finite=True)
    baseline_scores = review.array(metadata['scores'], shape=(384, 384), dtype=np.float32, finite=True)
    baseline_evaluated = review.array(metadata['evaluated'], shape=(384, 384), dtype=np.bool_)
    require(baseline_evaluated.all(), 'Cached baseline coverage differs')
    require(metadata['adapter_layer_start'] == 0 and metadata['adapter_direction'] == 'forward' and
            metadata['fixed_primary_mask'] == case['arrays']['primary_mask'] and
            [tile['bbox_roi_xywh'] for tile in metadata['captured_tiles']] == BOXES,
            'Baseline tile/depth/support contract differs')
    surface = review.array(case['arrays']['surface'], shape=(17, 384, 384), dtype=np.float32, finite=True)
    primary = review.array(case['arrays']['primary_mask'], shape=(384, 384), dtype=np.bool_)
    known = review.array(case['arrays']['known_mask'], shape=(384, 384), dtype=np.bool_)
    surface_mask = review.array(case['arrays']['surface_mask'], shape=(384, 384), dtype=np.bool_)
    border = np.zeros((384, 384), dtype=np.bool_)
    border[32:-32, 32:-32] = True
    require(np.array_equal(primary, known & border) and not (known & ~surface_mask).any(),
            'Independent primary/known/surface support relation differs')
    labels = review.array(case['arrays']['labels'], shape=(384, 384), dtype=np.uint8)
    require(np.isin(labels[primary], [0, 1]).all() and int((labels[primary] == 1).sum()) == case['primary_positive'] and
            int((labels[primary] == 0).sum()) == case['primary_negative'], 'Development label/count contract differs')
    hashes = []
    for index, (x, y, width, height) in enumerate(BOXES):
        image = normalize_patch(surface[:, y:y+height, x:x+width])[None, None]
        digest = array_hash(image)
        require(digest == metadata['captured_tiles'][index]['input_sha256'], 'Historical tile input fingerprint differs')
        hashes.append(dict(input_sha256=digest, feature_sha256={arm: array_hash(features(image, arm)) for arm in ARMS}))
    reconstructed_baseline, coverage = blend_logits(baseline)
    baseline_error = float(np.max(np.abs(reconstructed_baseline.astype(np.float64) - baseline_scores)))
    require(baseline_error <= 2e-6, 'Baseline sigmoid/blend reconstruction differs')
    maps = {'unchanged_seed42': baseline_scores}
    errors = {'unchanged_seed42': baseline_error}
    for arm in ARMS:
        item = outputs[arm]
        require(item['baseline_metadata'] == case['baseline_metadata'] and item['fixed_primary_mask'] == case['arrays']['primary_mask'] and
                item['input_identity_exact'] is True and item['role'] == 'reused development diagnostic',
                'New prediction lineage/support differs')
        require([tile['bbox_roi_xywh'] for tile in item['captured_tiles']] == BOXES, 'Development tile order differs')
        for actual, expected in zip(item['captured_tiles'], hashes):
            require(actual['input_sha256'] == expected['input_sha256'] and
                    actual['feature_sha256'] == expected['feature_sha256'][arm], 'Independent development features differ')
        logits = review.array(item['logits'], shape=(25, 128, 128), dtype=np.float32, finite=True)
        differences = review.array(item['logit_differences'], shape=(25, 128, 128), dtype=np.float32, finite=True)
        require(bit_equal(differences, logits - baseline), 'Development diagnostic logit difference differs')
        evaluated = review.array(item['evaluated'], shape=(384, 384), dtype=np.bool_)
        require(np.array_equal(evaluated, baseline_evaluated), 'Development evaluated support differs')
        scores = review.array(item['scores'], shape=(384, 384), dtype=np.float32, finite=True)
        require(((scores >= 0) & (scores <= 1)).all(), 'Scores outside probability range')
        rebuilt, arm_coverage = blend_logits(logits)
        require(np.array_equal(arm_coverage, coverage), 'Coverage weights differ across arms')
        error = float(np.max(np.abs(rebuilt.astype(np.float64) - scores)))
        require(error <= 2e-6, 'Independent sigmoid/blending mismatch')
        maps[arm], errors[arm] = scores, error
    result = dict(case=case['id'], fragment=case['fragment'], parent=case['parent_papyrus'],
                  score_reconstruction_max_error=errors, coverage_pixels=int((coverage > 0).sum()), models={})
    for name, scores in maps.items():
        result['models'][name] = dict(rank=rank_metrics(labels[primary], scores[primary]),
                                     operating=threshold_metrics(labels[primary], scores[primary], CUTOFF))
    return result


def fixture_checks():
    zeros = np.zeros((25, 128, 128), dtype=np.float32)
    scores, coverage = blend_logits(zeros)
    require(np.array_equal(scores, np.full((384, 384), .5, dtype=np.float32)) and
            (coverage > 0).all(), 'Zero-logit blend/coverage mismatch')
    require(len(BOXES) == 25 and BOXES[0] == [0, 0, 128, 128] and BOXES[-1] == [256, 256, 128, 128],
            'Tile placement fixture mismatch')
    weights = hann_weights()
    require(weights.min() == np.float32(.001) and weights.max() == 1 and
            np.array_equal(weights, weights.T), 'Hann weight fixture mismatch')
    return dict(status='PASS', blend_identity_cases=1, coverage_cases=1, tile_schedule_cases=1,
                hann_weight_cases=1, scientific_array_reads=0)
