"""Fixed empirical 95% specificity on development negatives; no test tuning."""
from __future__ import annotations

import numpy as np


def choose_development_threshold(negative_scores):
    """Lowest float32 threshold excluding the entire crossing negative-score tie.

    Classification is float64(score) >= threshold; source scores are retained
    float32. The exact allowable FP count is floor(n_negative/20), avoiding
    floating rounding of the rational95/100 requirement. Positive labels play
    no role in threshold choice. A finite threshold above1 is allowed.
    """
    values = np.asarray(negative_scores)
    if values.dtype != np.float32 or values.ndim != 1 or not values.size:
        raise ValueError('Expected a nonempty one-dimensional float32 development-negative score vector')
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError('Development scores must be finite raw values in [0,1]')
    maximum_fp = int(values.size // 20)
    boundary = np.sort(values)[::-1][maximum_fp]
    threshold = float(np.nextafter(boundary, np.float32(np.inf), dtype=np.float32))
    false_positive = int(np.count_nonzero(values.astype(np.float64) >= threshold))
    if false_positive > maximum_fp:
        raise ValueError('Conservative tie cutoff failed exact false-positive bound')
    return dict(target_specificity_numerator=95, target_specificity_denominator=100,
        negative_count=int(values.size), maximum_false_positives=maximum_fp,
        excluded_negative_boundary=float(boundary), threshold=threshold,
        threshold_construction='nextafter(float32 kth-descending negative boundary, float32 +inf)',
        threshold_operator='float64(score) >= threshold',
        boundary_tie_count=int(np.count_nonzero(values == boundary)),
        development_false_positives=false_positive,
        development_specificity=(values.size - false_positive) / values.size,
        scope='Chosen only from development negatives; never refitted on reserved scores or labels.')


def operating_metrics(labels, scores, threshold):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.dtype != np.bool_ or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError('Validated Boolean labels and aligned scores required')
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1) or not np.isfinite(threshold):
        raise ValueError('Finite raw scores in [0,1] and finite threshold required')
    predicted = scores >= threshold
    positive, negative = int(labels.sum()), int((~labels).sum())
    tp, fp = int((labels & predicted).sum()), int((~labels & predicted).sum())
    tn, fn = negative - fp, positive - tp
    return dict(threshold=float(threshold), threshold_operator='float64(score) >= threshold',
        pixel_count=int(labels.size), supplied_positive_pixels=positive, supplied_negative_pixels=negative,
        confusion=dict(tp=tp, fp=fp, tn=tn, fn=fn), sensitivity=tp/positive if positive else None,
        specificity=tn/negative if negative else None,
        precision=tp/(tp+fp) if tp+fp else None,
        balanced_accuracy=(tp/positive+tn/negative)/2 if positive and negative else None)
