"""Offline qualification of complete cases in the unchanged eight-source intake.

Incomplete cases stay unresolved. This never selects a replacement cohort and
does not establish physical fine/native registration or authenticate ink.
"""
import argparse
import collections
import csv
from datetime import datetime, timezone
import functools
import hashlib
import json
from pathlib import Path
import shutil
import zlib

import cv2
import numpy as np
import tifffile


PLAN_SHA = 'ffd9817800e9ca6cc00287874bfe835e3e868a5643b39f0be857e89ff613b74a'


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def area_crop(src, x, y, n, dst_hw):
    """Exact integer-integral box areas, using global rational pixel edges.

    This arithmetic is independent of OpenCV and the earlier float weight matrix.
    It returns unrounded areas; the published recipe uses OpenCV uint8 output.
    """
    sh, sw = src.shape
    dh, dw = dst_hw
    xe = np.arange(x, x+n+1, dtype=np.int64) * sw
    ye = np.arange(y, y+n+1, dtype=np.int64) * sh
    xi, xr = xe // dw, xe % dw
    yi, yr = ye // dh, ye % dh
    x0, x1 = int(xi[0]), int((xe[-1]+dw-1)//dw)
    y0, y1 = int(yi[0]), int((ye[-1]+dh-1)//dh)
    a = src[y0:y1, x0:x1]
    px = np.zeros((a.shape[0], a.shape[1]+1), dtype=np.int64)
    np.cumsum(a, axis=1, dtype=np.int64, out=px[:, 1:])
    edge_values = np.pad(a, ((0, 0), (0, 1)))[:, xi-x0]
    horizontal = np.diff(px[:, xi-x0]*dw + edge_values*xr, axis=1)
    py = np.zeros((horizontal.shape[0]+1, n), dtype=np.int64)
    np.cumsum(horizontal, axis=0, dtype=np.int64, out=py[1:])
    edge_rows = np.pad(horizontal, ((0, 1), (0, 0)))[yi-y0]
    numerator = np.diff(py[yi-y0]*dh + edge_rows*yr[:, None], axis=0)
    return numerator.astype(np.float64) / (sw*sh)


def arithmetic_controls():
    rng = np.random.default_rng(1208142318)
    checks = []
    for sh, sw, dh, dw in [(41, 59, 13, 19), (64, 96, 16, 24), (29, 31, 29, 31)]:
        for constant in [None, 0, 64, 65, 127, 128, 255]:
            a = rng.integers(0, 256, (sh, sw), dtype=np.uint8) if constant is None else np.full((sh, sw), constant, np.uint8)
            cv = cv2.resize(a, (dw, dh), interpolation=cv2.INTER_AREA)
            n = min(8, dh, dw)
            for x, y in [(0, 0), (dw-n, dh-n), ((dw-n)//2, (dh-n)//2)]:
                exact = area_crop(a, x, y, n, (dh, dw))
                error = float(np.abs(exact-cv[y:y+n, x:x+n]).max())
                assert error <= .5002, error
                if constant is not None:
                    assert np.all(exact == constant)
                checks.append(dict(shape=[sh, sw, dh, dw], constant=constant, xy=[x, y], max_abs_error=error))
    return checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(1)
    a.output.mkdir(exist_ok=False)
    assert sha(a.work/'REQUEST_PLAN.json') == PLAN_SHA
    plan = json.loads((a.work/'REQUEST_PLAN.json').read_text())
    for rel, digest in plan['base_pins'].items():
        assert sha(a.base/rel) == digest, rel
    controls = arithmetic_controls()
    inputs = a.output/'reference'
    inputs.mkdir()
    for folder in ['teacher-metadata', 'source-fixtures']:
        shutil.copytree(a.base/folder, inputs/folder, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['SOURCE_CROP_GO.json', 'SOURCE_QUALIFICATION.json']:
        shutil.copyfile(a.base/name, inputs/name)
    shutil.copyfile(a.base/'support-reservation-v2/CANDIDATES.csv', inputs/'CANDIDATES.csv')
    shutil.copyfile(__file__, a.output/'qualify_added.py')
    candidates = list(csv.DictReader((inputs/'CANDIDATES.csv').open()))
    available = {}
    states = []
    receipt_pins = {}
    for r in plan['requests']:
        rp = a.work/'receipts'/(r['id']+'.json')
        if not rp.exists():
            continue
        rec = json.loads(rp.read_text())
        assert rec['request'] == r and rec['status'] in ['PRESENT', 'MISSING']
        available[r['id']] = rec
    for case in plan['cases']:
        requests = [r for r in plan['requests'] if r['source_id'] == case['id']]
        missing = sum(r['id'] in available and available[r['id']]['status'] == 'MISSING' for r in requests)
        unresolved = sum(r['id'] not in available for r in requests)
        states.append(dict(id=case['id'], candidates=case['candidates'], restricted=case['restricted'],
            status='SOURCE_MISSING' if missing else ('TRANSPORT_UNKNOWN' if unresolved else 'SOURCE_BYTES_COMPLETE'),
            confirmed_missing=missing, unresolved=unresolved))
        if missing or unresolved:
            continue
        for r in requests:
            rec = available[r['id']]
            assert (a.work/r['path']).stat().st_size == r['bytes'] == rec['bytes']
            assert sha(a.work/r['path']) == rec['sha256']
            receipt_pins[r['id']] = sha(a.work/'receipts'/(r['id']+'.json'))
    freeze = dict(utc=datetime.now(timezone.utc).isoformat(), implementation_sha256=sha(Path(__file__)),
        request_plan_sha256=PLAN_SHA, cases=states, source_receipt_sha256=receipt_pins,
        reference_sha256={str(f.relative_to(inputs)): sha(f) for f in sorted(inputs.rglob('*')) if f.is_file()},
        rule='Process every fully available case in fixed metadata order. Leave other cases unresolved; do not admit a smaller cohort.',
        thresholds=dict(positive_at_least=128, negative_at_most=64, minimum_supervised=1024, ct_centre_plane=14),
        teacher_rounding='Global OpenCV uint8 INTER_AREA, compared with independent unrounded integer-integral area arithmetic.',
        full_cohort_admitted=False, paid_compute=False, model_forwards=0)
    write(a.output/'FREEZE.json', freeze)
    write(a.output/'ARITHMETIC_CONTROLS.json', dict(status='PASS', checks=controls))
    header_ledger = json.loads((inputs/'teacher-metadata/LEDGER.json').read_text())['requests']
    smoke_plan = json.loads((inputs/'SOURCE_CROP_GO.json').read_text())
    smoke_result = json.loads((inputs/'SOURCE_QUALIFICATION.json').read_text())
    results = []
    for case, state in zip(plan['cases'], states, strict=True):
        if state['status'] != 'SOURCE_BYTES_COMPLETE':
            results.append(dict(state, supervision_qualified=False))
            continue
        ident = case['id']
        rows = [r for r in candidates if r['segment'] == case['segment']]
        assert len(rows) == case['candidates']
        fh, fw = case['teacher_shape']
        _, h, w = case['native_shape']
        header = json.loads((inputs/'teacher-metadata'/f'{ident}-HEADER.json').read_text())
        canvas = np.zeros((fh, fw), np.uint8)
        temp = a.output/(ident+'-header-only.tif')
        with temp.open('wb') as f:
            f.truncate(header['object_bytes'])
            for hr in header_ledger:
                if hr['id'] != ident:
                    continue
                data = (inputs/'teacher-metadata'/hr['path']).read_bytes()
                assert hashlib.sha256(data).hexdigest() == hr['sha256'] and len(data) == hr['bytes']
                f.seek(hr['offset']); f.write(data)
        teacher_requests = [r for r in plan['requests'] if r['source_id'] == ident and r['kind'] == 'teacher']
        with tifffile.TiffFile(temp) as tf:
            page = tf.pages[0]
            assert page.shape == (fh, fw)
            for r in teacher_requests:
                payload = (a.work/r['path']).read_bytes()
                raw = zlib.decompress(payload)
                assert len(raw) == 1024**2
                explicit = np.frombuffer(raw, np.uint8).reshape(1024, 1024)
                official = page.decode(payload, r['tile_index'])[0].reshape(1024, 1024)
                assert np.array_equal(explicit, official)
                ty, tx = r['ty']*1024, r['tx']*1024
                hh, ww = min(1024, fh-ty), min(1024, fw-tx)
                canvas[ty:ty+hh, tx:tx+ww] = explicit[:hh, :ww]
        temp.unlink()
        resized = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
        smoke = next(r for r in smoke_plan['cases'] if r['id'] == ident)
        smoke_verified = False
        if smoke['status'] == 'SOURCE_PRESENT':
            old = next(r for r in smoke_result['cases'] if r['id'] == ident)
            fixture = inputs/old['file']
            assert sha(fixture) == old['sha256']
            x, y, _, _ = smoke['native_crop_xy']
            with np.load(fixture, allow_pickle=False) as saved:
                assert np.array_equal(resized[y:y+128, x:x+128], saved['teacher'])
            smoke_verified = True
        @functools.lru_cache(maxsize=64)
        def ct(cy, cx):
            r = available[f'{ident}-ct-y{cy:04}-x{cx:04}']['request']
            return np.fromfile(a.work/r['path'], np.uint8).reshape(28, 128, 128)
        teachers = np.empty((len(rows), 128, 128), np.uint8)
        valids = np.empty((len(rows), 128, 128), bool)
        seen = np.zeros((h, w), bool)
        seen_a = np.zeros((h, w), bool)
        material = np.zeros((h, w), bool)
        support_rows = []
        max_error = 0.
        for i, row in enumerate(rows):
            x, y = int(row['x']), int(row['y'])
            # Require every tile influencing this crop; unpopulated canvas values
            # outside this footprint must never become a candidate's negative.
            x0=max(0,x*fw//w-2); x1=min(fw,((x+128)*fw+w-1)//w+2)
            y0=max(0,y*fh//h-2); y1=min(fh,((y+128)*fh+h-1)//h+2)
            for ty in range(y0//1024, (y1-1)//1024+1):
                for tx in range(x0//1024, (x1-1)//1024+1):
                    key=f'{ident}-teacher-{ty*((fw+1023)//1024)+tx:05}'
                    assert available[key]['status'] == 'PRESENT'
            teacher = resized[y:y+128, x:x+128]
            exact = area_crop(canvas, x, y, 128, (h, w))
            error = float(np.abs(exact-teacher.astype(np.float64)).max())
            assert error <= .5002, (row['id'], error)
            max_error = max(max_error, error)
            patch = np.empty((28, 128, 128), np.uint8)
            for cy in range(y//128, (y+127)//128+1):
                for cx in range(x//128, (x+127)//128+1):
                    ya, yb = max(y, cy*128), min(y+128, (cy+1)*128)
                    xa, xb = max(x, cx*128), min(x+128, (cx+1)*128)
                    patch[:, ya-y:yb-y, xa-x:xb-x] = ct(cy, cx)[:, ya-cy*128:yb-cy*128, xa-cx*128:xb-cx*128]
            valid = patch[14] > 0
            labelled = (teacher >= 128) | (teacher <= 64)
            mask = valid & labelled
            pos = mask & (teacher >= 128)
            count = int(mask.sum()); positives = int(pos.sum())
            teachers[i] = teacher; valids[i] = valid
            seen[y:y+128, x:x+128] = True
            if int(row['in_restricted_A']): seen_a[y:y+128, x:x+128] = True
            material[y:y+128, x:x+128] |= valid
            record = dict(row, supervised=count, positive=positives, negative=count-positives,
                ignored=128**2-count, valid_ct_pixels=int(valid.sum()),
                intermediate_teacher_pixels=int((~labelled).sum()),
                supervision_removed_by_ct=int((labelled & ~valid).sum()),
                technically_eligible=count >= 1024, independent_area_max_abs_error=error)
            for d in range(28): record[f'ct_nonzero_depth_{d:02}'] = int((patch[d] > 0).sum())
            support_rows.append(record)
        ct.cache_clear()
        derived = a.output/(ident+'-patches.npz')
        np.savez_compressed(derived, teacher=teachers, valid=valids)
        csv_path = a.output/(ident+'-support.csv')
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(support_rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(support_rows)
        counts = []
        for name, subset, uv in [('restricted_A', [r for r in support_rows if int(r['in_restricted_A'])], seen_a), ('broad_B', support_rows, seen)]:
            sup_uv = uv & material & ((resized >= 128) | (resized <= 64))
            counts.append(dict(arm=name, candidates=len(subset), technically_eligible=sum(r['technically_eligible'] for r in subset),
                zero_positive_patches=sum(r['positive']==0 for r in subset), zero_negative_patches=sum(r['negative']==0 for r in subset),
                supervised_occurrences=sum(r['supervised'] for r in subset), positive_occurrences=sum(r['positive'] for r in subset),
                unique_supervised_uv=int(sup_uv.sum()), unique_positive_uv=int((sup_uv & (resized >= 128)).sum())))
        result = dict(state, status='COMPLETE_CASE_MECHANICALLY_QUALIFIED', supervision_qualified=True,
            tiff_tiles_compared=len(teacher_requests), explicit_and_official_tiff_equal=True,
            independent_area_crops=len(rows), independent_area_max_abs_error=max_error,
            prior_smoke_teacher_byte_identical=smoke_verified, counts=counts,
            outputs={p.name:sha(p) for p in [csv_path, derived]},
            interpretation='Weak teacher and CT-support bookkeeping only; not physical registration, authenticated blanks/ink, independent sheet area or full-cohort admission.')
        results.append(result)
        print(json.dumps(result), flush=True)
        del canvas, resized, teachers, valids, seen, seen_a, material
    write(a.output/'RESULTS.json', dict(status='PARTIAL_POPULATION_QUALIFICATION',
        freeze_sha256=sha(a.output/'FREEZE.json'), cases=results, arithmetic_controls=len(controls),
        numpy_version=np.__version__, opencv_version=cv2.__version__, tifffile_version=tifffile.__version__,
        full_cohort_admitted=False, new_model_forwards=0, paid_compute=False))


if __name__ == '__main__':
    main()
