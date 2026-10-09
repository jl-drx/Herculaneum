"""Descriptive binary-label metrics with exact score-tie grouping.

No threshold fitting, weighting, bootstrap or score calibration is performed.
ROC AUC integrates the grouped ROC polygon; average precision uses the
noninterpolated recall increments at complete equal-score threshold groups.
"""

from __future__ import annotations

import numpy as np


def binary_metrics(labels, scores, *, threshold=0.5):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError('Labels and scores must be aligned one-dimensional arrays.')
    if labels.dtype != np.bool_:
        raise ValueError('Labels must be independently validated and converted to Boolean.')
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('Scores must be finite in [0,1]; never clip scores silently.')
    if threshold != 0.5:
        raise ValueError('This benchmark fixes the diagnostic threshold at0.5.')
    count = labels.size
    positives = int(labels.sum())
    negatives = count - positives
    predicted = scores >= threshold
    tp = int(np.count_nonzero(labels & predicted))
    fp = int(np.count_nonzero(~labels & predicted))
    fn = positives - tp
    tn = negatives - fp
    sensitivity = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    precision = tp / (tp + fp) if tp + fp else None
    metrics = dict(pixel_count=int(count), supplied_positive_pixels=positives,
        supplied_negative_pixels=negatives, supplied_positive_fraction=positives / count if count else None,
        roc_auc=None, average_precision=None, distinct_score_groups=0,
        tied_score_pixel_count=0, tied_score_pixel_fraction=None,
        fixed_threshold=threshold, threshold_rule='score >=0.5',
        confusion=dict(tp=tp, fp=fp, tn=tn, fn=fn), sensitivity=sensitivity,
        specificity=specificity, precision=precision,
        balanced_accuracy=(sensitivity + specificity) / 2 if positives and negatives else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        interpretation='Descriptive supplied-label agreement; no calibration, independence or ink authentication.')
    empty = np.empty(0, dtype=np.float64)
    curves = dict(thresholds_descending=empty, true_positive_count=np.empty(0, dtype=np.int64),
        false_positive_count=np.empty(0, dtype=np.int64), tpr=empty, fpr=empty,
        recall=empty, precision=empty)
    if not count:
        return metrics, curves
    order = np.argsort(-scores, kind='stable')
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    group_ends = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), count - 1]
    group_sizes = np.diff(np.r_[0, group_ends + 1])
    cumulative_tp = np.cumsum(ordered_labels, dtype=np.int64)[group_ends]
    cumulative_fp = group_ends + 1 - cumulative_tp
    tied_count = int(group_sizes[group_sizes > 1].sum())
    metrics.update(distinct_score_groups=int(group_ends.size), tied_score_pixel_count=tied_count,
                   tied_score_pixel_fraction=tied_count / count)
    tpr = np.r_[0., cumulative_tp / positives] if positives else np.full(group_ends.size + 1, np.nan)
    fpr = np.r_[0., cumulative_fp / negatives] if negatives else np.full(group_ends.size + 1, np.nan)
    precision_curve = np.r_[1., cumulative_tp / (group_ends + 1)]
    if positives and negatives:
        metrics['roc_auc'] = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) * .5))
    if positives:
        metrics['average_precision'] = float(np.sum(np.diff(tpr) * precision_curve[1:]))
    curves = dict(thresholds_descending=ordered_scores[group_ends], true_positive_count=cumulative_tp,
        false_positive_count=cumulative_fp, tpr=tpr, fpr=fpr,
        recall=tpr.copy(), precision=precision_curve)
    return metrics, curves


def paired_score_diagnostics(first, second, labels):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if first.ndim != 1 or first.shape != second.shape or first.shape != labels.shape:
        raise ValueError('Paired scores and labels must align on exactly the same mask.')
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Paired scores must be finite.')
    difference = first - second
    first_positive, second_positive = first >= .5, second >= .5
    first_correct, second_correct = first_positive == labels, second_positive == labels
    result = dict(pixel_count=int(first.size), first_minus_second_score_mean=float(difference.mean()) if first.size else None,
        first_minus_second_score_median=float(np.median(difference)) if first.size else None,
        mean_absolute_score_difference=float(np.abs(difference).mean()) if first.size else None,
        fixed_midpoint_prediction_disagreements=int(np.count_nonzero(first_positive != second_positive)),
        first_correct_second_wrong=int(np.count_nonzero(first_correct & ~second_correct)),
        second_correct_first_wrong=int(np.count_nonzero(second_correct & ~first_correct)),
        both_correct=int(np.count_nonzero(first_correct & second_correct)),
        both_wrong=int(np.count_nonzero(~first_correct & ~second_correct)),
        pearson_r=None,
        interpretation='Paired descriptive behavior on the same supplied-label domain; no pixel independence assumption.')
    if first.size >= 2 and first.std() > 0 and second.std() > 0:
        result['pearson_r'] = float(np.corrcoef(first, second)[0, 1])
    return result
