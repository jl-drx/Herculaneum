"""Tie-aware reference metrics from grouped concordance and suffix counts.

This module imports no experiment metric implementation or model code.
"""
import math
import numpy as np


def compute(labels, scores, thresholds):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or not labels.size:
        raise ValueError("Need equal nonempty one-dimensional label/score arrays")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be explicitly binary")
    if not np.isfinite(scores).all():
        raise ValueError("A fixed participating score is nonfinite")
    truth = labels.astype(bool, copy=False)
    values = scores.astype(np.float64, copy=False)
    positive = int(truth.sum())
    negative = truth.size - positive
    if not positive or not negative:
        raise ValueError("Both classes required for declared AUC/AP comparison")

    # Ascending score groups: count every concordant positive-negative pair,
    # giving half credit to ties. No ROC curve or average-rank implementation.
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    group_count = np.diff(np.r_[starts, len(ordered)]).astype(np.int64)
    group_positive = np.add.reduceat(truth[order].astype(np.int64), starts)
    group_negative = group_count - group_positive
    negative_below = np.cumsum(group_negative) - group_negative
    concordance = np.sum(group_positive * (negative_below + group_negative / 2), dtype=np.float64)
    auc = float(concordance / (positive * negative))

    # At a tied score threshold the entire group enters together. Suffixes
    # provide TP/predicted-positive counts, with group positives as recall mass.
    positive_at_or_above = np.cumsum(group_positive[::-1])[::-1]
    count_at_or_above = np.cumsum(group_count[::-1])[::-1]
    ap = float(np.sum(group_positive * positive_at_or_above / count_at_or_above,
                      dtype=np.float64) / positive)
    operating = {}
    for name, cutoff in thresholds.items():
        cutoff = float(cutoff)
        if not math.isfinite(cutoff):
            raise ValueError("Nonfinite frozen threshold")
        # Promote scores before comparison so a float64 cutoff is not silently
        # rounded down to the float32 array dtype at an equality boundary.
        predicted = values >= cutoff
        tp = int(np.count_nonzero(predicted & truth))
        fp = int(np.count_nonzero(predicted & ~truth))
        tn, fn = negative - fp, positive - tp
        operating[name] = {"threshold": cutoff, "tp": tp, "fp": fp,
                           "tn": tn, "fn": fn,
                           "sensitivity": tp / positive,
                           "specificity": tn / negative}
    return {"positive": positive, "negative": negative, "pixels": positive + negative,
            "auc": auc, "ap": ap, "operating": operating}
