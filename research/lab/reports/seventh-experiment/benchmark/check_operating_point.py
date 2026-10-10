#!/usr/bin/env python3
"""Small independent-count fixtures for conservative ties, not model tests."""
import json
import numpy as np
from operating_point import choose_development_threshold, operating_metrics


def main():
    checks = []
    cases = [
        ('one_allowed_fp', [.9, .8] + [.1]*18, 1, 1, .95),
        ('top_tie_excluded_whole', [.9, .9] + [.1]*18, 1, 0, 1.),
        ('two_allowed_fp', [.9, .9, .8] + [.1]*37, 2, 2, .95),
        ('less_than20_negatives_zero_allowed', [.8]+[.1]*18, 0, 0, 1.),
        ('all_scores_one_threshold_above1', [1.]*20, 1, 0, 1.),
        ('all_zero_ties_excluded', [0.]*20, 1, 0, 1.),
    ]
    for name, values, maximum, expected_fp, specificity in cases:
        values = np.asarray(values, dtype=np.float32)
        result = choose_development_threshold(values)
        assert result['maximum_false_positives'] == maximum
        measured = operating_metrics(np.zeros(values.size, dtype=bool), values, result['threshold'])
        assert measured['confusion']['fp'] == expected_fp and measured['specificity'] == specificity
        assert (values.size-measured['confusion']['fp'])*100 >= 95*values.size
        permuted = choose_development_threshold(values[::-1])
        assert permuted['threshold'] == result['threshold']
        boundary = np.float32(result['excluded_negative_boundary'])
        above = np.nextafter(boundary, np.float32(np.inf), dtype=np.float32)
        # Independent two-score operational check: whole boundary excluded,
        # next representable score included, with no float32 downcast leak.
        if above <= 1:
            metrics = operating_metrics(np.array([True, False]), np.array([above,boundary],dtype=np.float32), result['threshold'])
            assert metrics['confusion'] == dict(tp=1,fp=0,tn=1,fn=0)
        else:
            assert result['threshold'] > 1
        checks.append(dict(name=name,status='pass',threshold=result['threshold'],false_positives=expected_fp))
    for values in [np.array([],dtype=np.float32),np.array([np.nan],dtype=np.float32),np.array([1.1],dtype=np.float32)]:
        try:
            choose_development_threshold(values)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid development scores must stop')
    print(json.dumps(dict(status='pass',synthetic_only=True,model_calls=0,checks=checks),indent=2))


if __name__ == '__main__':
    main()
