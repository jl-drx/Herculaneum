"""Proposed source-normalized added-stream loss; no model or trainer changes.

Call validate_window on the complete post-augmentation optimizer window first.
Then call partial_objective for every microbatch using those fixed totals.
The caller applies the separately frozen original/added stream coefficients.
"""
import math


def validate_window(totals, coefficients):
    if not coefficients or set(totals) != set(coefficients):
        raise ValueError('Every configured source must occur in the full window')
    if any(not math.isfinite(v) or v <= 0 for v in coefficients.values()):
        raise ValueError('Source coefficients must be finite and positive')
    if not math.isclose(sum(coefficients.values()), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError('Source coefficients must sum to one')
    for record in totals.values():
        if set(record) != {'pixels', 'samples'} or any(type(v) is not int or v <= 0 for v in record.values()):
            raise ValueError('Positive integer window denominators are required')
        if record['pixels'] < record['samples']:
            raise ValueError('Each sample requires nonempty supervision')


def partial_objective(logits, targets, masks, source_ids, totals, coefficients,
                      partial_stream_loss, *, bce_smoothing=.5, dice_weight=.25):
    validate_window(totals, coefficients)
    if len(source_ids) != len(logits) or not source_ids:
        raise ValueError('Every microbatch sample needs a source ID')
    if set(source_ids) - set(coefficients):
        raise ValueError('Unknown microbatch source')
    result = logits.new_zeros(())
    for source in sorted(set(source_ids)):
        indices = [i for i, name in enumerate(source_ids) if name == source]
        result = result + coefficients[source] * partial_stream_loss(
            logits[indices], targets[indices], masks[indices], totals[source],
            bce_smoothing=bce_smoothing, dice_weight=dice_weight)
    return result
