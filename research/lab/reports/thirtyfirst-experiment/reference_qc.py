"""Explicit supplied-label QC; unknown values never become negatives."""
import numpy as np


def classify(image):
    """Losslessly identify opaque black/white; return 0/1/2 (unknown)."""
    if image.mode == '1':
        image = image.convert('L')
    if image.mode == 'P':
        image = image.convert('RGBA')
    a = np.asarray(image)
    if a.dtype != np.uint8:
        raise ValueError('Only 8-bit/binary PNG pixels supported')
    if a.ndim == 2:
        gray, opaque = a, np.ones(a.shape, dtype=bool)
    elif image.mode in ('LA', 'RGB', 'RGBA'):
        gray = a[..., 0]
        opaque = np.ones(gray.shape, dtype=bool)
        if image.mode in ('RGB', 'RGBA'):
            opaque &= (a[..., 0] == a[..., 1]) & (a[..., 0] == a[..., 2])
        if image.mode in ('LA', 'RGBA'):
            opaque &= a[..., -1] == 255
    else:
        raise ValueError('Unsupported PNG mode: ' + image.mode)
    out = np.full(gray.shape, 2, dtype=np.uint8)
    out[opaque & (gray == 0)] = 0
    out[opaque & (gray == 255)] = 1
    return out


def tile_table(labels, mask):
    if labels.shape != mask.shape or labels.ndim != 2:
        raise ValueError('Equal two-dimensional images required')
    h, w = labels.shape
    table = []
    for y in range(32, h - 32 - 512 + 1, 512):
        for x in range(32, w - 32 - 512 + 1, 512):
            lab, m = labels[y:y+512, x:x+512], mask[y:y+512, x:x+512]
            positive = int(np.count_nonzero((lab == 1) & (m == 1)))
            negative = int(np.count_nonzero((lab == 0) & (m == 1)))
            support = positive + negative
            reasons = []
            if support / 512**2 < .95:
                reasons.append('common_support_below_95_percent')
            if positive < 256:
                reasons.append('supplied_positives_below_256')
            if negative < 256:
                reasons.append('supplied_negatives_below_256')
            table.append(dict(id=f'r{y}-c{x}', y=y, x=x, height=512, width=512,
                              vertical_band=min(3, (4*(y+256)) // h),
                              positive=positive, negative=negative, common_support=support,
                              excluded_or_unknown=512**2-support, eligible=not reasons,
                              exclusion_reasons=reasons))
    return table


def select_tiles(table):
    chosen = []
    for band in range(4):
        candidates = [t for t in table if t['vertical_band'] == band and t['eligible']]
        if candidates:
            chosen.append(min(candidates, key=lambda t: (t['y'], t['x'])))
    return chosen
