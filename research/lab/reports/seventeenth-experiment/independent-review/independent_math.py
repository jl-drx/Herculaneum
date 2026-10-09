"""Independent numerical kernels; no imports of producer helpers or model code."""
import math
import numpy as np


def quantiles(values):
    a = np.asarray(values, dtype=np.float64).ravel()
    if not len(a):
        return [None] * 3
    positions = [(len(a) - 1) * q for q in (0.1, 0.5, 0.9)]
    indices = sorted({int(math.floor(x)) for x in positions} | {int(math.ceil(x)) for x in positions})
    work = np.partition(a, indices)
    return [float(work[int(math.floor(p))] + (p-math.floor(p)) * (work[int(math.ceil(p))]-work[int(math.floor(p))])) for p in positions]


def profile(plane, mask, endpoint_max):
    a = np.asarray(plane, dtype=np.float64)
    b = np.asarray(mask)
    assert a.ndim == 2 and b.shape == a.shape and b.dtype == np.bool_
    assert np.isfinite(a).all() and (a >= 0).all() and (a <= endpoint_max).all()
    values = a[b]
    n = int(values.size)
    mean = float(np.add.reduce(values, dtype=np.float64)/n) if n else None
    std = float(np.sqrt(np.add.reduce(np.square(values-mean), dtype=np.float64)/n)) if n else None
    low = int(np.count_nonzero(values == 0))
    high = int(np.count_nonzero(values == endpoint_max))
    hmask = b[:, :-1] & b[:, 1:]
    vmask = b[:-1] & b[1:]
    nh, nv = int(hmask.sum()), int(vmask.sum())
    q = quantiles(values)
    return dict(count=n, mean=mean, std=std, q10=q[0], q50=q[1], q90=q[2],
                endpoint_min_count=low, endpoint_max_count=high,
                endpoint_min_fraction=low/n if n else None,
                endpoint_max_fraction=high/n if n else None,
                horizontal_edge_count=nh, vertical_edge_count=nv,
                horizontal_mean_absolute_difference=float(np.abs(a[:, 1:]-a[:, :-1])[hmask].sum(dtype=np.float64)/nh) if nh else None,
                vertical_mean_absolute_difference=float(np.abs(a[1:]-a[:-1])[vmask].sum(dtype=np.float64)/nv) if nv else None)


def exact_pool(planes):
    """Explicit strided gathers, independent of reshape/axis aggregation."""
    a = np.asarray(planes)
    assert a.dtype == np.uint16 and a.ndim == 3 and a.shape[0] == 3
    assert a.shape[1] % 3 == a.shape[2] % 3 == 0
    accum = np.zeros((a.shape[1]//3, a.shape[2]//3), dtype=np.uint32)
    for z in range(3):
        for y in range(3):
            for x in range(3):
                accum += a[z, y::3, x::3]
    return (accum.astype(np.float64)/6939.0).astype(np.float32)


def block_average(image):
    a = np.asarray(image, dtype=np.float64)
    assert a.ndim == 2 and a.shape[0] % 3 == a.shape[1] % 3 == 0
    out = np.zeros((a.shape[0]//3, a.shape[1]//3), dtype=np.float64)
    for y in range(3):
        for x in range(3):
            out += a[y::3, x::3]
    return out/9.0


def gaussian_reflect(image, sigma=16, radius=64):
    """Explicit separable normalized Gaussian; symmetric padding is half-sample reflect."""
    a = np.asarray(image, dtype=np.float64)
    axis = np.arange(-radius, radius+1, dtype=np.float64)
    kernel = np.exp(-0.5 * (axis/sigma)**2)
    kernel /= kernel.sum(dtype=np.float64)
    padded = np.pad(a, ((radius, radius), (0, 0)), mode='symmetric')
    vertical = np.zeros(a.shape, dtype=np.float64)
    for index, weight in enumerate(kernel):
        vertical += weight * padded[index:index+a.shape[0], :]
    padded = np.pad(vertical, ((0, 0), (radius, radius)), mode='symmetric')
    out = np.zeros(a.shape, dtype=np.float64)
    for index, weight in enumerate(kernel):
        out += weight * padded[:, index:index+a.shape[1]]
    return out


def ranking(values, labels):
    s = np.asarray(values, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel()
    assert s.shape == y.shape and np.isfinite(s).all() and np.isin(y, [0, 1]).all()
    positive = int(y.sum())
    negative = int(len(y)-positive)
    # Ascending group counts give pairwise wins without sorting labels stably.
    distinct, inverse = np.unique(s, return_inverse=True)
    pos = np.bincount(inverse, weights=y, minlength=len(distinct)).astype(np.int64)
    neg = np.bincount(inverse, minlength=len(distinct)).astype(np.int64)-pos
    twice_wins, below = 0, 0
    for p, n in zip(pos.tolist(), neg.tolist()):
        twice_wins += p*(2*below+n)
        below += n
    auc = twice_wins/(2*positive*negative) if positive and negative else None
    total_seen, positive_seen, pieces = 0, 0, []
    for p, n in zip(pos[::-1].tolist(), neg[::-1].tolist()):
        total_seen += p+n
        positive_seen += p
        if p:
            pieces.append((p/positive)*(positive_seen/total_seen))
    ap = math.fsum(pieces) if positive else None
    return dict(count=len(y), positive_count=positive, negative_count=negative, roc_auc=auc, average_precision=ap)


def geometry(neighborhood):
    a = np.asarray(neighborhood, dtype=np.float64)
    assert a.shape == (3, 3, 6)
    p, n = a[1, 1, :3], a[1, 1, 3:]
    result = dict(coordinate_finite=bool(np.isfinite(p).all()), normal_finite=bool(np.isfinite(n).all()))
    nn = float(np.linalg.norm(n)) if result['normal_finite'] else None
    result['normal_norm'] = nn
    if not np.isfinite(a[:, :, :3]).all():
        result.update(du_length=None, dv_length=None, cross_area=None, signed_normal_agreement=None, tangent_agreement=None)
        return result
    du = (a[1, 2, :3]-a[1, 0, :3])/2.0
    dv = (a[2, 1, :3]-a[0, 1, :3])/2.0
    lu, lv = math.sqrt(float(du@du)), math.sqrt(float(dv@dv))
    cross = np.array([du[1]*dv[2]-du[2]*dv[1], du[2]*dv[0]-du[0]*dv[2], du[0]*dv[1]-du[1]*dv[0]])
    area = math.sqrt(float(cross@cross))
    result.update(du_length=lu, dv_length=lv, cross_area=area,
                  signed_normal_agreement=float(cross@n)/(area*nn) if area and nn else None,
                  tangent_agreement=float(du@dv)/(lu*lv) if lu and lv else None)
    return result
