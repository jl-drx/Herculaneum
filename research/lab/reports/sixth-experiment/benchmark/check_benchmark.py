#!/usr/bin/env python3
"""Small synthetic metric/mask guards only; no checkpoint or real-data inference."""
import json
from pathlib import Path
import tempfile

import numpy as np

from binary_metrics import binary_metrics
from comparison_common import INPUT_NAMES, load_data, load_plan, record


def main():
    checks = []
    cases = [
        ('all_tied', [1, 0], [.5, .5], .5, .5),
        ('perfect', [1, 0, 1, 0], [.9, .1, .8, .2], 1., 1.),
        ('reversed', [1, 0, 1, 0], [.2, .9, .1, .8], 0., 5/12),
        ('mixed_tie', [1, 0, 0, 1], [.8, .8, .3, .2], .375, .5),
    ]
    for name, labels, scores, auc, ap in cases:
        labels, scores = np.array(labels, dtype=bool), np.array(scores)
        result, _ = binary_metrics(labels, scores)
        # Enumerated pair wins independently define the ROC probability interpretation.
        pair_auc = np.mean([float(p > n) + .5 * float(p == n)
                            for p in scores[labels] for n in scores[~labels]])
        assert np.isclose(result['roc_auc'], auc) and np.isclose(result['roc_auc'], pair_auc)
        assert np.isclose(result['average_precision'], ap)
        checks.append(dict(name=name, status='pass', roc_auc=result['roc_auc'], average_precision=result['average_precision']))
    for name, labels in [('empty', []), ('all_positive', [1, 1]), ('all_negative', [0, 0])]:
        result, _ = binary_metrics(np.array(labels, dtype=bool), np.full(len(labels), .5))
        assert result['roc_auc'] is None and result['balanced_accuracy'] is None
        assert result['average_precision'] == (1. if name == 'all_positive' else None)
        checks.append(dict(name=name, status='pass'))
    result, _ = binary_metrics(np.array([True, False]), np.array([.5, .5]))
    assert result['confusion'] == dict(tp=1, fp=1, tn=0, fn=0)
    checks.append(dict(name='midpoint_greater_equal', status='pass'))
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='synthetic-mask-check-', dir=here) as temporary:
        directory = Path(temporary)
        surface = np.zeros((21, 256, 256), dtype=np.uint8)  # Observed zeros are intentionally retained.
        labels = np.zeros((256, 256), dtype=np.uint8)
        labels[:, :128] = 255
        values = dict(surface=surface, inklabels=labels,
            supervision_mask=np.ones((256, 256), dtype=np.uint8),
            validation_mask=np.ones((256, 256), dtype=np.uint8),
            acquisition_observed_mask=np.ones((256, 256), dtype=bool),
            source_observed_nonzero_mask=np.zeros((256, 256), dtype=bool))
        plan = dict(inputs={}, data=dict(shape=list(surface.shape),
            binary_foreground=dict(inklabels=255, supervision_mask=1, validation_mask=1)))
        for name in INPUT_NAMES:
            path = directory/(name+'.npy')
            np.save(path, values[name], allow_pickle=False)
            plan['inputs'][name] = record(path)
        _, _, primary, _, summary = load_data(plan)
        assert primary.sum() == 16384 and summary['primary_observed_zero_profile_pixels'] == 16384
        assert summary['raw_supervision_validation_overlap_pixels'] == 65536
        assert summary['primary_before_predictions']['supplied_positive_pixels'] == 8192
        checks.append(dict(name='raw_supervision_overlap_and_observed_zeros_retained', status='pass'))
        values['acquisition_observed_mask'][0, 0] = False
        np.save(directory/'acquisition_observed_mask.npy', values['acquisition_observed_mask'], allow_pickle=False)
        plan['inputs']['acquisition_observed_mask'] = record(directory/'acquisition_observed_mask.npy')
        try:
            load_data(plan)
        except ValueError as error:
            assert 'actually acquired support' in str(error)
        else:
            raise AssertionError('Missing acquisition support should stop')
        checks.append(dict(name='missing_acquisition_support_stops', status='pass'))
        values['acquisition_observed_mask'][:] = True
        np.save(directory/'acquisition_observed_mask.npy', values['acquisition_observed_mask'], allow_pickle=False)
        plan['inputs']['acquisition_observed_mask'] = record(directory/'acquisition_observed_mask.npy')
        values['validation_mask'][:] = 0
        np.save(directory/'validation_mask.npy', values['validation_mask'], allow_pickle=False)
        plan['inputs']['validation_mask'] = record(directory/'validation_mask.npy')
        try:
            load_data(plan)
        except ValueError as error:
            assert 'STOP' in str(error)
        else:
            raise AssertionError('Insufficient fixed primary support should stop')
        checks.append(dict(name='inadequate_fixed_primary_stops', status='pass'))
        draft = directory/'draft.json'
        draft.write_text(json.dumps(dict(schema_version=1, kind='fixed-w029-cpu-comparison', ready_for_execution=False)))
        try:
            load_plan(draft, execute=True)
        except ValueError as error:
            assert 'not frozen' in str(error)
        else:
            raise AssertionError('Draft plan should not execute')
        checks.append(dict(name='unfrozen_plan_rejects_execution_before_model_calls', status='pass'))
    print(json.dumps(dict(status='pass', synthetic_only=True, inference_calls=0, checks=checks), indent=2))


if __name__ == '__main__':
    main()
