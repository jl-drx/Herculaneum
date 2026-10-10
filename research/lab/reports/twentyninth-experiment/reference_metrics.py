"""Bounded supplied-reference diagnostics. No loaders, models, inference or CI.

Inspired by the separate ink/depth/surface diagnostics in Uri Kartoun's MIT
fragment benchmark; see UPSTREAM_LICENSE.txt and SOURCE_AUDIT.md. This local
implementation requires explicit coverage and unknown labels. It does not
authenticate references, physical coordinates or independence from training.
"""
from numbers import Integral
import numpy as np

MAX_ELEMENTS = 1_000_000
MAX_VOXELS = 4_000_000


def _mask(value, shape, name):
    a = np.asarray(value)
    if a.dtype != np.dtype(bool) or a.shape != shape:
        raise ValueError(f'{name} must be Boolean with shape {shape}')
    return a


def _bounded(value, limit):
    a = np.asarray(value)
    if a.size > limit:
        raise ValueError('input exceeds bounded diagnostic scope')
    return a


def decode_probabilities(values, valid, *, encoding, limit=MAX_ELEMENTS):
    """Return float64 values plus validity. Invalid output values are zero placeholders."""
    a = _bounded(values, limit)
    valid = _mask(valid, a.shape, 'valid').copy()
    if encoding == 'float01':
        if a.dtype.kind != 'f':
            raise ValueError('float01 requires a floating dtype')
        scale = 1.
    elif encoding in ('uint8', 'uint16'):
        width = 1 if encoding == 'uint8' else 2
        if a.dtype.kind != 'u' or a.dtype.itemsize != width:
            raise ValueError('integer encoding/dtype mismatch')
        scale = 255. if width == 1 else 65535.
    elif encoding == 'label012':
        if a.dtype.kind not in 'ui' or not np.isin(a[valid], [0, 1, 2]).all():
            raise ValueError('labels must be integer 0, 1 or unknown2')
        valid &= a != 2
        scale = 1.
    else:
        raise ValueError('explicit supported encoding required')
    chosen = a[valid].astype(np.float64) / scale
    if not np.isfinite(chosen).all() or np.any((chosen < 0) | (chosen > 1)):
        raise ValueError('valid probabilities must be finite and in [0,1]')
    result = np.zeros(a.shape, dtype=np.float64)
    result[valid] = chosen
    return result, valid


def row_coverage(height, rows):
    """Explicit half-open declared rows; absent metadata must not imply full coverage."""
    if isinstance(height, bool) or not isinstance(height, Integral) or not 0 <= height <= MAX_ELEMENTS:
        raise ValueError('invalid height')
    if not isinstance(rows, (tuple, list)) or len(rows) != 2:
        raise ValueError('coverage rows declaration required')
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in rows):
        raise ValueError('integer row bounds required')
    lo, hi = rows
    if not 0 <= lo <= hi <= height:
        raise ValueError('invalid row bounds')
    result = np.zeros(height, dtype=bool)
    result[lo:hi] = True
    return result


def _reference(values, labels, selected, valid):
    p = _bounded(values, MAX_ELEMENTS)
    y = np.asarray(labels)
    if y.shape != p.shape or y.dtype.kind not in 'ui' or not np.isin(y, [0, 1, 2]).all():
        raise ValueError('reference labels must be same-shape integer 0,1,2')
    selected = _mask(selected, p.shape, 'selected')
    valid = _mask(valid, p.shape, 'prediction_valid')
    p, valid = decode_probabilities(p, valid, encoding='float01')
    known = selected & (y != 2)
    use = known & valid
    counts = dict(selected=int(selected.sum()), unknown=int((selected & (y == 2)).sum()),
                  known=int(known.sum()), missing_prediction=int((known & ~valid).sum()),
                  scored_positive=int((use & (y == 1)).sum()),
                  scored_negative=int((use & (y == 0)).sum()))
    return p[use], y[use], counts


def rank_metrics(values, labels, *, selected, prediction_valid):
    """Exact tie-aware ranking, with both classes required by our diagnostic policy."""
    p, y, counts = _reference(values, labels, selected, prediction_valid)
    n_pos, n_neg = counts['scored_positive'], counts['scored_negative']
    if not n_pos or not n_neg:
        return dict(counts=counts, status='undefined_requires_both_classes', auc=None, ap=None)
    order = np.argsort(p, kind='stable')
    scores, labels = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    ends = np.r_[starts[1:], scores.size]
    pos = np.add.reduceat((labels == 1).astype(np.float64), starts)
    neg = (ends - starts) - pos
    auc = np.sum(pos * (np.cumsum(neg) - neg + .5 * neg)) / (n_pos * n_neg)
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    ap = np.sum((pos[::-1] / n_pos) * (tp / (tp + fp)))
    return dict(counts=counts, status='defined', auc=float(auc), ap=float(ap))


def surface_response_ratio(values, labels, *, selected, prediction_valid):
    """Clear/ink mean ratio on explicit complete-band support; no accuracy interpretation."""
    p, y, counts = _reference(values, labels, selected, prediction_valid)
    if not counts['scored_positive'] or not counts['scored_negative']:
        return dict(counts=counts, ratio=None, status='undefined_requires_both_classes')
    ink, clear = float(p[y == 1].mean()), float(p[y == 0].mean())
    ratio = clear / ink if ink > 0 else None
    status = 'defined' if ink > 0 else 'undefined_zero_ink_mean'
    if ratio is not None and not np.isfinite(ratio):
        ratio, status = None, 'undefined_ratio_overflow'
    return dict(counts=counts, mean_ink=ink, mean_clear=clear, ratio=ratio, status=status)


def depth_diagnostics(values, *, voxel_valid, surface_layers, surface_valid,
                      selected, z_offset, band=(-4, 5)):
    """Z-by-column diagnostics in declared layer coordinates. Arrays retain explicit masks."""
    p = _bounded(values, MAX_VOXELS)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] > MAX_ELEMENTS:
        raise ValueError('nonempty-depth Z-by-column block required')
    p, valid = decode_probabilities(p, voxel_valid, encoding='float01', limit=MAX_VOXELS)
    shape = (p.shape[1],)
    selected = _mask(selected, shape, 'selected')
    reliable = _mask(surface_valid, shape, 'surface_valid')
    surface = np.asarray(surface_layers)
    if surface.shape != shape or surface.dtype.kind not in 'ui':
        raise ValueError('integer absolute surface layer per column required')
    if isinstance(z_offset, bool) or not isinstance(z_offset, Integral):
        raise ValueError('integer z_offset required')
    if not isinstance(band, (tuple, list)) or len(band) != 2 or any(
            isinstance(v, bool) or not isinstance(v, Integral) for v in band):
        raise ValueError('integer band endpoints required')
    lo, hi = band
    # Layer coordinates are not physical units. Bound arithmetic well within exact float64 integers.
    if not -1_000_000_000 <= lo <= hi <= 1_000_000_000 or abs(z_offset) > 1_000_000_000:
        raise ValueError('layer coordinates exceed supported range')
    eligible = selected & reliable
    if np.any(surface[eligible] > 1_000_000_000) or np.any(surface[eligible] < -1_000_000_000):
        raise ValueError('surface coordinates exceed supported range')
    safe_surface = np.where(eligible, surface, 0).astype(np.float64)
    relative = (np.arange(p.shape[0], dtype=np.float64) + z_offset)[:, None] - safe_surface
    profile_valid = selected & valid.all(axis=0)
    full_depth = eligible & profile_valid
    inside = (relative >= lo) & (relative <= hi)
    band_valid = eligible & (inside.sum(axis=0) == hi - lo + 1) & ((~inside) | valid).all(axis=0)
    maximum = p.max(axis=0)
    mass = p.sum(axis=0)
    active = full_depth & (mass > 0)
    tied = active & ((p == maximum).sum(axis=0) != 1)
    peak_valid = active & ~tied
    centroid = np.full(shape, np.nan)
    centroid[active] = (p * relative).sum(axis=0)[active] / mass[active]
    peak = np.full(shape, np.nan)
    peak[peak_valid] = p.argmax(axis=0)[peak_valid] + z_offset - safe_surface[peak_valid]
    band_mean = np.full(shape, np.nan)
    band_mean[band_valid] = (p * inside).sum(axis=0)[band_valid] / (hi - lo + 1)
    maximum[~profile_valid] = np.nan
    return dict(maximum=maximum, profile_valid=profile_valid, band_mean=band_mean,
                band_valid=band_valid, centroid=centroid, centroid_valid=active,
                peak=peak, peak_valid=peak_valid,
                counts=dict(selected=int(selected.sum()),
                    surface_unknown=int((selected & ~reliable).sum()),
                    complete_profile=int(profile_valid.sum()),
                    incomplete_profile=int((selected & ~profile_valid).sum()),
                    complete_surface_band=int(band_valid.sum()),
                    incomplete_surface_band=int((eligible & ~band_valid).sum()),
                    zero_mass=int((full_depth & (mass == 0)).sum()),
                    tied_peak=int(tied.sum()), centroid_defined=int(active.sum()),
                    unique_peak_defined=int(peak_valid.sum())))


def parent_group_folds(fragment_groups):
    """Declare folds only; metadata is caller-supplied, not a proof of provenance."""
    if not fragment_groups or any(not isinstance(k, str) or not k or not isinstance(v, str)
                                  or not v for k, v in fragment_groups.items()):
        raise ValueError('nonempty fragment/group names required')
    return [dict(group=group,
                 test=sorted(k for k, v in fragment_groups.items() if v == group),
                 train=sorted(k for k, v in fragment_groups.items() if v != group))
            for group in sorted(set(fragment_groups.values()))]
