"""Frozen descriptive statistics; no searches, fitted parameters, or models."""
import numpy as np


def support_geometry(masks):
    return {name: dict(mask=mask, horizontal=mask[:, :-1] & mask[:, 1:],
                       vertical=mask[:-1] & mask[1:]) for name, mask in masks.items()}


def plane_stats(plane, geometries, endpoint_max):
    values64 = np.asarray(plane, dtype=np.float64)
    horizontal = np.abs(np.diff(values64, axis=1))
    vertical = np.abs(np.diff(values64, axis=0))
    out = {}
    for name, geometry in geometries.items():
        values = values64[geometry["mask"]]
        count = int(values.size)
        lower = int(np.count_nonzero(values == 0))
        upper = int(np.count_nonzero(values == endpoint_max))
        h_values, v_values = horizontal[geometry["horizontal"]], vertical[geometry["vertical"]]
        quantiles = np.quantile(values, [0.1, 0.5, 0.9], method="linear") if count else [None] * 3
        out[name] = dict(count=count, mean=float(values.mean()) if count else None,
                         std=float(values.std(ddof=0)) if count else None,
                         q10=float(quantiles[0]) if count else None,
                         q50=float(quantiles[1]) if count else None,
                         q90=float(quantiles[2]) if count else None,
                         endpoint_min_count=lower, endpoint_max_count=upper,
                         endpoint_min_fraction=lower / count if count else None,
                         endpoint_max_fraction=upper / count if count else None,
                         horizontal_mean_absolute_difference=float(h_values.mean()) if h_values.size else None,
                         vertical_mean_absolute_difference=float(v_values.mean()) if v_values.size else None,
                         horizontal_edge_count=int(h_values.size), vertical_edge_count=int(v_values.size))
    return out


def rank_metrics(scores, labels):
    """Tie groups shared by ROC area and noninterpolated average precision."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    if scores.size != labels.size or not np.isfinite(scores).all():
        raise ValueError("Finite, matching scores/labels required")
    count = int(scores.size)
    positive = int(labels.sum())
    negative = count - positive
    if not count:
        return dict(count=0, positive=0, negative=0, auroc=None, average_precision=None)
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), count - 1]
    cumulative_true = np.cumsum(sorted_labels, dtype=np.int64)[ends]
    cumulative_count = ends + 1
    cumulative_false = cumulative_count - cumulative_true
    group_true = np.diff(np.r_[0, cumulative_true])
    average_precision = float(np.sum(group_true * (cumulative_true / cumulative_count), dtype=np.float64) / positive) if positive else None
    if positive and negative:
        tpr = np.r_[0.0, cumulative_true / positive]
        fpr = np.r_[0.0, cumulative_false / negative]
        auroc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) * 0.5, dtype=np.float64))
    else:
        auroc = None
    return dict(count=count, positive=positive, negative=negative, auroc=auroc,
                average_precision=average_precision, distinct_score_groups=int(ends.size))
