#!/usr/bin/env python3
"""build_concat.py - join the two spiral-fit meshes w047 and w048 into one tifxyz across the spiral seam (theta 360|0).

    python build_concat.py --w047 outputs/w047_R5B2_z9500-11000.tifxyz --w048 outputs/w048_R5B2_z9500-11000.tifxyz \
        --letter-boxes inputs/concat_w047-w048_R5B2_z9500-11000_letter_boxes.json --out-dir work/

Writes <out-dir>/concat_w047-w048_R5B2_z9500-11000.tifxyz (x/y/z.tif + meta.json), the node source map
<out-dir>/concat_w047-w048_R5B2_z9500-11000_node_source.npz (used by ink/merge_tile_phase.py) and build.json.
CPU only, a few seconds. The result is bit-identical to outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz.

How the join works:
  1. w048 is copied bit for bit (rows 32-122, columns 11-175) at row -32, column +160. Both shifts are multiples of
     16 nodes (= 320 render px), so w048 keeps the tile phase of its own render for the ink models.
  2. w047 (rows 3-119, columns 96-265) is placed at row -2, column -90: w048 row = w047 row + 30 and w048 column =
     w047 column - 250, the grid offsets measured across the seam in the text rows.
  3. The two meshes are rigid grids, and their true offset drifts along the seam (row offset 26.8-30.7, column offset
     246.9-251.5, measured by extending each grid across the gap). Per concat row R the remaining offset
     delta(R) (rows) and eps(R) (columns) is measured at the edge nodes, smoothed along the seam and tapered to 0 above
     and below the seam rows.
  4. That remaining offset is spread over a transition strip on the w047 side: t = smoothstep across the strip
     (width per row from STRIP_WIDTH_NODES, never touching a w047 letter box), and node (R, C) = the w047 surface at
     w047 grid position (r - t*delta, c + t*eps), bilinear between four valid w047 nodes, i.e. on the surface exactly
     as it is rendered. At the ragged sheet edge the cell may extrapolate by up to one node; where no cell is
     available the original node is kept. t = 0 gives the original node, bit for bit.
  5. Bridge: per row, linear 3D interpolation between the last w047 node and the first w048 node.
Node source codes: 0 none, 1 w047 (original), 2 w048 (original), 3 bridge, 4 w047 resampled (transition strip).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

NAME = 'concat_w047-w048_R5B2_z9500-11000'
PX = 20                   # render px per node (tifxyz scale 0.05)
K = 30                    # w048 row = w047 row + K (measured in the text rows: 30.1-30.7)
DELTA = 250               # w048 column offset: S48 - S47 (measured in the text rows: 249.5-250.2)
NR, NC = 128, 336         # node grid of the joined mesh (render 2560 x 6720 px)
PART = {                  # source rows/columns (end exclusive) and destination offsets, in nodes
    'w047': dict(row0=3, row1=120, col0=96, col1=266, A=-2, S=-90),
    'w048': dict(row0=32, row1=123, col0=11, col1=176, A=-32, S=160),
}
RAND_MAX = 12             # bridge only if the two edge nodes are at most 12 columns apart
D_MAX = 120.0             # and closer than 120 voxels in 3D
# transition strip width (nodes) as a function of the concat row; capped next to the letters (see below)
STRIP_WIDTH_NODES = [(0, 100.0), (28, 100.0), (42, 8.0), (76, 8.0), (80, 15.0), (100, 15.0), (106, 100.0), (NR, 100.0)]
TAPER_ROWS = 8            # above/below the seam rows the shift goes smoothly to 0
SMOOTH_SIGMA_ROWS = 1.5   # gaussian smoothing of the per-row offsets along the seam
SMOOTH_MEDIAN_ROWS = 3    # median filter before the gaussian
EXTRAPOLATE_MAX = 1.0     # strip: the bilinear cell may extrapolate by up to 1 node at the ragged edge
LETTER_MARGIN_NODES = 0.5 # the strip never enters a w047 letter box (+ margin)
W048_LETTER_IDS = {1, 2, 3, 4, 5}   # letters 1-5 lie on w048, 6-18 on w047


def sha(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_mesh(folder):
    """x/y/z as float32 (bit-exact), plus validity as in VC3D (x/y/z != -1, z > 0)."""
    import tifffile
    X = tifffile.imread(os.path.join(folder, 'x.tif'))
    Y = tifffile.imread(os.path.join(folder, 'y.tif'))
    Z = tifffile.imread(os.path.join(folder, 'z.tif'))
    assert X.dtype == np.float32 and X.shape == Y.shape == Z.shape
    ok = (X != -1) & (Y != -1) & (Z != -1) & (Z > 0) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    return X, Y, Z, ok


def stats(x):
    x = np.asarray([v for v in x if v is not None], float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0)
    return dict(n=int(x.size), med=round(float(np.median(x)), 2), p10=round(float(np.percentile(x, 10)), 2),
                p90=round(float(np.percentile(x, 90)), 2), min=round(float(x.min()), 2), max=round(float(x.max()), 2))


def tangents(P, ok, r, c, sgn):
    """median row and column tangent around (r, c), one-sided into the part (sgn +1: towards lower columns = w047 edge)."""
    ur, uc = [], []
    for i in range(-3, 4):
        for j in range(0, 4):
            jj, rr = c - sgn * j, r + i
            if 0 <= rr < P.shape[0] - 1 and 1 <= jj < P.shape[1] - 1:
                if ok[rr, jj] and ok[rr + 1, jj]:
                    ur.append(P[rr + 1, jj] - P[rr, jj])
                j2 = jj - sgn
                if ok[rr, jj] and ok[rr, j2]:
                    uc.append(sgn * (P[rr, jj] - P[rr, j2]))
    return np.median(ur, 0), np.median(uc, 0)


def bilinear(P, ok, rho, kap, ext_max=0.0):
    """bilinear in cell (floor rho, floor kap); with ext_max > 0 also in a neighbouring cell with 4 valid corners, as long
    as the extrapolation (distance outside [0,1] in u and v, in nodes) is <= ext_max. Returns (point, extrapolation) or
    (None, None)."""
    i0, j0 = int(np.floor(rho)), int(np.floor(kap))
    best = None
    for i in (i0, i0 - 1, i0 + 1):
        for j in (j0, j0 - 1, j0 + 1):
            if i < 0 or j < 0 or i + 1 >= P.shape[0] or j + 1 >= P.shape[1]:
                continue
            if not (ok[i, j] and ok[i + 1, j] and ok[i, j + 1] and ok[i + 1, j + 1]):
                continue
            u, v = rho - i, kap - j
            e = max(0.0, -u, u - 1) + max(0.0, -v, v - 1)
            if e > ext_max + 1e-9:
                continue
            if best is None or e < best[1]:
                q = (P[i, j] * (1 - u) * (1 - v) + P[i + 1, j] * u * (1 - v) + P[i, j + 1] * (1 - u) * v + P[i + 1, j + 1] * u * v)
                best = (q, e)
    return best if best is not None else (None, None)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def smooth(y, sigma, med=None):
    from scipy.ndimage import gaussian_filter1d, median_filter
    return gaussian_filter1d(median_filter(y, size=med or SMOOTH_MEDIAN_ROWS, mode='nearest'), sigma, mode='nearest')


def strip_width(R):
    xs, ys = zip(*STRIP_WIDTH_NODES)
    return float(np.interp(R, xs, ys))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--w047', required=True, help='w047_R5B2_z9500-11000.tifxyz')
    ap.add_argument('--w048', required=True, help='w048_R5B2_z9500-11000.tifxyz')
    ap.add_argument('--letter-boxes', required=True, help='letter boxes in the raw frame of the joined render')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    t0 = time.time()
    src = {}
    for w, d in (('w047', a.w047), ('w048', a.w048)):
        X, Y, Z, ok = read_mesh(d)
        dg = PART[w]
        okc = np.zeros_like(ok)
        okc[dg['row0']:dg['row1'], dg['col0']:dg['col1']] = ok[dg['row0']:dg['row1'], dg['col0']:dg['col1']]
        sums = {f: sha(os.path.join(d, f)) for f in ('x.tif', 'y.tif', 'z.tif')}
        src[w] = dict(X=X, Y=Y, Z=Z, ok=ok, okc=okc, P=np.stack([X, Y, Z], -1).astype(np.float64), sha=sums)
        print(f'{w}: {X.shape}, valid {ok.sum()} (in part {okc.sum()})', flush=True)
    a7, b8 = src['w047'], src['w048']
    d47, d48 = PART['w047'], PART['w048']
    A47, S47, A48, S48 = d47['A'], d47['S'], d48['A'], d48['S']
    assert A47 - A48 == K and S48 - S47 == DELTA

    # ------------------------------------------------------------------ 1. per-row offsets at the edge nodes
    meas = []
    for r47 in range(d47['row0'], d47['row1']):
        r48 = r47 + K
        if not (d48['row0'] <= r48 < d48['row1']):
            continue
        v7 = np.nonzero(a7['okc'][r47])[0]; v8 = np.nonzero(b8['okc'][r48])[0]
        if len(v7) == 0 or len(v8) == 0:
            continue
        L, F = int(v7.max()), int(v8.min())
        ur, uc = tangents(a7['P'], a7['ok'], r47, L, +1)
        s7 = np.linalg.lstsq(np.stack([ur, uc], 1), b8['P'][r48, F] - a7['P'][r47, L], rcond=None)[0]
        ur8, uc8 = tangents(b8['P'], b8['ok'], r48, F, -1)
        s8 = np.linalg.lstsq(np.stack([ur8, uc8], 1), a7['P'][r47, L] - b8['P'][r48, F], rcond=None)[0]
        Kc = 0.5 * ((r48 - (r47 + s7[0])) + ((r48 + s8[0]) - r47))
        Dc = 0.5 * (((L + s7[1]) - F) + (L - (F + s8[1])))
        meas.append(dict(R=r47 + A47, r47=r47, r48=r48, L=L, F=F, Kc=float(Kc), Dc=float(Dc)))
    Rm = np.array([m['R'] for m in meas]); assert (np.diff(Rm) == 1).all()
    Rall = np.arange(NR)
    Kc_r = np.interp(Rall, Rm, smooth(np.array([m['Kc'] for m in meas]), SMOOTH_SIGMA_ROWS))
    Dc_r = np.interp(Rall, Rm, smooth(np.array([m['Dc'] for m in meas]), SMOOTH_SIGMA_ROWS))
    g = np.ones(NR)
    g[Rall > Rm.max()] = 1 - smoothstep((Rall[Rall > Rm.max()] - Rm.max()) / TAPER_ROWS)
    g[Rall < Rm.min()] = 1 - smoothstep((Rm.min() - Rall[Rall < Rm.min()]) / TAPER_ROWS)
    delta = (Kc_r - K) * g                # rows
    eps = (Dc_r - DELTA) * g              # columns
    print(f"seam rows {Rm.min()}-{Rm.max()}: row offset {stats([m['Kc'] for m in meas])}; "
          f"column offset {stats([m['Dc'] for m in meas])}", flush=True)

    # ------------------------------------------------------------------ 2. rigid placement (validity only), seam line
    V7 = np.zeros((NR, NC), bool); V8 = np.zeros((NR, NC), bool)
    V7[d47['row0'] + A47:d47['row1'] + A47, d47['col0'] + S47:d47['col1'] + S47] = a7['okc'][d47['row0']:d47['row1'], d47['col0']:d47['col1']]
    V8[d48['row0'] + A48:d48['row1'] + A48, d48['col0'] + S48:d48['col1'] + S48] = b8['okc'][d48['row0']:d48['row1'], d48['col0']:d48['col1']]
    L7 = np.array([np.nonzero(V7[R])[0].max() if V7[R].any() else -1 for R in range(NR)], float)
    F8 = np.array([np.nonzero(V8[R])[0].min() if V8[R].any() else NC for R in range(NR)])
    has7 = L7 >= 0
    Cs = np.interp(Rall, Rall[has7], smooth(L7[has7], 4.0, med=5))   # smooth seam line (column) per row
    Cs = np.where(has7, np.maximum(Cs, L7), Cs)                       # t = 1 at the last valid w047 node
    Wd = np.array([strip_width(R) for R in Rall])
    # the strip starts only right of every w047 letter box (+ margin) in rows R +- 2
    boxes = json.load(open(a.letter_boxes))['letters']
    limit = np.full(NR, -1.0)
    for e in boxes:
        if e['id'] in W048_LETTER_IDS:
            continue
        x0, y0, x1, y1 = e['box']
        for R in range(int(np.floor(y0 / PX - 2)), int(np.ceil(y1 / PX + 2)) + 1):
            if 0 <= R < NR:
                limit[R] = max(limit[R], np.ceil(x1 / PX + LETTER_MARGIN_NODES))
    Wd = np.where(limit >= 0, np.minimum(Wd, Cs - limit), Wd)
    Wd = np.maximum(Wd, 1.0)
    active = (np.abs(delta) > 1e-3) | (np.abs(eps) > 1e-3)
    assert (Wd[active] >= 3).all(), [(int(R), round(float(Wd[R]), 2)) for R in np.nonzero(active & (Wd < 3))[0]]

    # ------------------------------------------------------------------ 3. grid
    Cg = np.full((NR, NC, 3), -1.0, np.float32)
    source = np.zeros((NR, NC), np.uint8)
    srow = np.full((NR, NC), -1, np.int16); scol = np.full((NR, NC), -1, np.int16)
    shift_px = np.zeros((NR, NC, 2), np.float32)
    for R in range(NR):                                       # w048 first (it wins), bit for bit
        for C in np.nonzero(V8[R])[0]:
            r, c = R - A48, C - S48
            Cg[R, C] = (b8['X'][r, c], b8['Y'][r, c], b8['Z'][r, c]); source[R, C] = 2; srow[R, C] = r; scol[R, C] = c
    count = dict(strip=0, extrapolated=0, kept_original=0)
    for R in range(NR):                                       # w047: unchanged outside the strip, resampled inside
        r0 = R - A47
        if not (d47['row0'] <= r0 < d47['row1']):
            continue
        c_lo, c_hi = d47['col0'] + S47, d47['col1'] + S47
        for C in range(max(c_lo, 0), min(c_hi, NC)):
            if C >= F8[R]:
                break                                         # w048 owns everything from its first valid node
            t = float(smoothstep((C - (Cs[R] - Wd[R])) / Wd[R]))
            r, c = r0, C - S47
            rigid = a7['okc'][r, c]
            dr, dc = -t * delta[R], t * eps[R]
            if t == 0.0 or (abs(dr) < 1e-9 and abs(dc) < 1e-9):
                if rigid:
                    Cg[R, C] = (a7['X'][r, c], a7['Y'][r, c], a7['Z'][r, c]); source[R, C] = 1; srow[R, C] = r; scol[R, C] = c
                continue
            rho, kap = r0 + dr, c + dc
            q, e = (None, None)
            if d47['row0'] - 1 <= rho <= d47['row1'] and d47['col0'] - 1 <= kap <= d47['col1']:
                q, e = bilinear(a7['P'], a7['okc'], rho, kap, ext_max=EXTRAPOLATE_MAX)
            if q is not None and e > 0 and not rigid:
                q = None                                      # extrapolate only to keep an existing w047 node
            if q is None:
                if rigid:                                     # ragged edge: keep the original node
                    Cg[R, C] = (a7['X'][r, c], a7['Y'][r, c], a7['Z'][r, c]); source[R, C] = 1; srow[R, C] = r; scol[R, C] = c
                    count['kept_original'] += 1
                continue
            Cg[R, C] = q.astype(np.float32); source[R, C] = 4
            shift_px[R, C] = (dr * PX, dc * PX)
            count['strip'] += 1
            if e > 0:
                count['extrapolated'] += 1
    ok = (Cg[..., 0] != -1) & (Cg[..., 1] != -1) & (Cg[..., 2] > 0)
    seam_rows = []                                            # bridge per row
    for R in range(NR):
        v7 = np.nonzero(ok[R] & ((source[R] == 1) | (source[R] == 4)))[0]; v8 = np.nonzero(ok[R] & (source[R] == 2))[0]
        if len(v7) == 0 or len(v8) == 0:
            continue
        cL, cR = int(v7.max()), int(v8.min())
        assert cL < cR
        pL, pR = Cg[R, cL].astype(np.float64), Cg[R, cR].astype(np.float64)
        D = float(np.linalg.norm(pR - pL))
        rec = dict(R=R, C_L=cL, C_R=cR, steps=cR - cL, d3_vox=round(D, 2), vox_per_step=round(D / (cR - cL), 2),
                   bridged=False)
        if D <= D_MAX and cR - cL <= RAND_MAX:
            n = cR - cL - 1
            for j in range(1, n + 1):
                Cg[R, cL + j] = (pL + (pR - pL) * (j / (n + 1))).astype(np.float32); source[R, cL + j] = 3
            rec['bridged'] = True
        seam_rows.append(rec)
    ok = (Cg[..., 0] != -1) & (Cg[..., 1] != -1) & (Cg[..., 2] > 0)

    # ------------------------------------------------------------------ 4. checks
    cols = np.arange(NC)[None, :].repeat(NR, 0)
    lost = V7 & (cols < F8[:, None]) & ~((source == 1) | (source == 4))
    assert not lost.any(), f'{int(lost.sum())} valid w047 nodes lost'
    for code, w in ((1, 'w047'), (2, 'w048')):
        s = src[w]; RR, CC = np.nonzero(source == code)
        org = np.stack([s['X'][srow[RR, CC], scol[RR, CC]], s['Y'][srow[RR, CC], scol[RR, CC]], s['Z'][srow[RR, CC], scol[RR, CC]]], -1)
        assert (Cg[RR, CC].view(np.uint32) == org.view(np.uint32)).all(), f'{w}: not bit-identical'
    assert (source[V8] == 2).all(), 'a valid w048 node is missing'
    for e in boxes:                                           # the strip and the bridge stay out of the w047 letters
        if e['id'] in W048_LETTER_IDS:
            continue
        x0, y0, x1, y1 = e['box']; m = LETTER_MARGIN_NODES
        r0, r1 = int(np.floor(y0 / PX - m)), int(np.ceil(y1 / PX + m)); c0, c1 = int(np.floor(x0 / PX - m)), int(np.ceil(x1 / PX + m))
        sub = source[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1]
        assert not ((sub == 3) | (sub == 4)).any(), f'strip or bridge inside letter box {e["id"]}'
    n48 = int(V8.sum()); n47 = int((source == 1).sum()); n47s = int((source == 4).sum())
    bridged = [x for x in seam_rows if x['bridged']]
    sp = stats([x['vox_per_step'] for x in seam_rows])
    print(f"strip: {n47s} resampled w047 nodes ({count}); w047 original {n47}; w048 {n48}; "
          f"bridge {int((source == 3).sum())} nodes in {len(bridged)} rows; seam voxels per step {sp}", flush=True)

    # ------------------------------------------------------------------ 5. write
    import tifffile
    os.makedirs(a.out_dir, exist_ok=True)
    seg = os.path.join(a.out_dir, f'{NAME}.tifxyz')
    os.makedirs(seg, exist_ok=True)
    for i, f in enumerate(('x.tif', 'y.tif', 'z.tif')):
        tifffile.imwrite(os.path.join(seg, f), np.ascontiguousarray(Cg[..., i]))
    V = Cg[ok].astype(np.float64)
    quads = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    maxshift = np.abs(shift_px[source == 4]).max(0)
    meta = {
        'uuid': NAME, 'name': NAME, 'type': 'seg', 'format': 'tifxyz', 'scale': [0.05, 0.05],
        'bbox': [[float(V[:, 0].min()), float(V[:, 1].min()), float(V[:, 2].min())],
                 [float(V[:, 0].max()), float(V[:, 1].max()), float(V[:, 2].max())]],
        'area_vx2': float(quads.sum() * 400.0),
        'source': ('PHerc. 343 (Vesuvius Challenge, First Letters). Concatenation of the nodes of the two official spiral-fit '
                   'meshes w047_R5B2_z9500-11000 and w048_R5B2_z9500-11000 (the two meshes of the submitted image), joined at '
                   'the spiral seam theta 360|0 with a smooth seam (version 3.2). w048 node rows 32-122 x columns 11-175 are '
                   'copied bit for bit at row - 32, column + 160 (same place and tile phase as in its own render). w047 node '
                   f'rows 3-119 x columns 96-265 are placed at row {A47:+d}, column {S47:+d} (w048 row = w047 row + {K}, w048 '
                   f'column = w047 column - {DELTA}: the grid offsets measured across the seam in the text rows). Because the '
                   'two meshes are rigid grids whose true offset drifts along the seam (row offset 26.8-30.7, column offset '
                   '246.9-251.5, measured by extending each grid across the gap), the remaining offset is spread over a '
                   'transition strip on the w047 side (5-8 nodes wide next to the letters of text row 1, 15 next to text row 2, up to 100 in the rows without letters, so that the texture distortion stays below 3 percent there): there the w047 '
                   'nodes are resampled on the w047 surface itself (bilinear between four original w047 nodes, i.e. on the '
                   'surface as it is rendered; at the ragged sheet edge the bilinear cell may extrapolate by up to one node, and '
                   'where no cell is available the original node is kept, so every valid node of both meshes is present) at '
                   'grid positions shifted smoothly by at most '
                   f'{maxshift.round(0).astype(int).tolist()} px (row, column); all other w047 '
                   f'nodes and all w048 nodes are unchanged. Bridge: {int((source == 3).sum())} nodes in {len(bridged)} rows are '
                   'linear 3D interpolations between the last w047 node and the first w048 node of a row; since the two grids '
                   f'now line up, the seam has a median {sp.get("med")} voxels per grid step. Columns = phi increasing '
                   '(theta, then theta + 360), rows = z increasing; 20 voxels per node; 8.64 um voxels.'),
        'concat': {
            'version': 3.2,
            'parts': [
                {'mesh': 'w047_R5B2_z9500-11000', 'source_rows': [3, 119], 'source_cols': [96, 265],
                 'dest_row_offset': A47, 'dest_col_offset': S47, 'sha256': dict(a7['sha'])},
                {'mesh': 'w048_R5B2_z9500-11000', 'source_rows': [32, 122], 'source_cols': [11, 175],
                 'dest_row_offset': A48, 'dest_col_offset': S48, 'sha256': dict(b8['sha'])}],
            'row_offset_w048_minus_w047': K, 'col_offset_w048_minus_w047': DELTA,
            'nodes': {'w047_original': n47, 'w047_resampled_in_transition_strip': n47s, 'w048_original': n48,
                      'bridge': int((source == 3).sum())},
            'transition_strip': {'side': 'w047', 'width_nodes': STRIP_WIDTH_NODES, 'profile': 'smoothstep',
                                 'max_shift_px_row_col': maxshift.round(1).tolist(),
                                 'rule': ('node (R, C) = w047 surface at w047 grid position (R - row_offset - t*d(R), C - '
                                          'col_offset + t*e(R)), t = smoothstep across the strip, d/e = measured residual '
                                          'row/column offset of the seam row, smoothed along the seam (sigma 3 rows)')},
            'bridge_rows': len(bridged), 'seam_voxels_per_grid_step': {k: sp[k] for k in ('med', 'p10', 'p90')},
            'bridge_rule': 'per row: linear 3D interpolation between the last w047 node and the first w048 node'},
    }
    json.dump(meta, open(os.path.join(seg, 'meta.json'), 'w'), indent=2)
    np.savez_compressed(os.path.join(a.out_dir, f'{NAME}_node_source.npz'), source=source, valid=ok)
    shas = {f: sha(os.path.join(seg, f)) for f in sorted(os.listdir(seg))}
    json.dump(dict(name=NAME, nodes_hw=[NR, NC], valid_nodes=int(ok.sum()), nodes=meta['concat']['nodes'],
                   strip_count=count, seam_rows=seam_rows, output_sha256=shas, seconds=round(time.time() - t0, 1)),
              open(os.path.join(a.out_dir, 'build.json'), 'w'), indent=1)
    print('done:', seg)
    for f, h in shas.items():
        print(f'  {h}  {f}')


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
