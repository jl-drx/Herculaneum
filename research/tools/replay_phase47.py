"""Restore and recompute the saved four-origin study without model execution.

Requires NumPy and SciPy. Verifies the archive and every member, all four
float32 blends, baseline crop, and all six pairs of ranking/distance metrics.
Does not recheck raw CT, model execution or physical target independence.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from workspace import checked, restore_bundle

ORIGINAL = Path('/workspace/herculaneum-lab')
INDEX = Path(__file__).resolve().parents[1] / 'workspace/PHASE47_BUNDLE.json'


def replay(root):
    def relocated(path):
        return root / Path(path).relative_to(ORIGINAL)

    phase = root / 'reports/fortyseventh-experiment'
    run = json.loads((phase / 'corrected-run/RUN_RESULT.json').read_text())
    go = json.loads((phase / 'GO_V3.json').read_text())
    plans = json.loads((phase / 'SCHEDULE.json').read_text())
    expected = json.loads((phase / 'ANALYSIS.json').read_text())
    assert run['status'] == expected['status'] == 'PASS'
    assert len(run['offsets']) == len(plans) == 4
    y, x, bottom, right = go['roi_yxYX']
    shape = (bottom - y, right - x)
    valid = np.load(root / 'reports/fortysixth-experiment/work/valid.npy', allow_pickle=False)[y:bottom, x:right]
    weights = np.load(phase / 'corrected-run/blend_weights.npy', allow_pickle=False)
    assert valid.dtype == np.bool_ and int(valid.sum()) == expected['valid_pixels']
    maps = []
    for record, plan in zip(run['offsets'], plans):
        assert record['offset_yx'] == plan['offset_yx']
        total = np.zeros(shape, np.float32)
        mass = np.zeros(shape, np.float32)
        assert len(record['batches']) == len(plan['retained_batches'])
        for batch, pin in zip(plan['retained_batches'], record['batches']):
            tilefile = relocated(pin['probabilities']['path'])
            assert checked(tilefile, pin['probabilities'])
            tiles = np.load(tilefile, allow_pickle=False)
            assert tiles.shape == (len(batch), 1, 128, 128)
            assert len(batch) == pin['batch'] and tiles.dtype == np.float32
            for blockid, tile in zip(batch, tiles[:, 0]):
                yy, xx, h, w = plan['blocks'][blockid]
                ya, xa, yb, xb = max(y, yy), max(x, xx), min(bottom, yy+h), min(right, xx+w)
                if ya >= yb or xa >= xb:
                    continue
                src = np.s_[ya-yy:yb-yy, xa-xx:xb-xx]
                dst = np.s_[ya-y:yb-y, xa-x:xb-x]
                total[dst] += tile[src] * weights[src]
                mass[dst] += weights[src]
        out = relocated(record['response']['path']).parent
        assert np.array_equal(total, np.load(out / 'probability_sum.npy', allow_pickle=False))
        assert np.array_equal(mass, np.load(out / 'weight_sum.npy', allow_pickle=False))
        assert (mass[valid] > 1e-6).all()
        np.divide(total, mass, out=total, where=mass > 1e-6)
        encoded = (np.clip(total, 0, 1) * 255).astype(np.uint8)
        actual = np.load(out / 'response.npy', allow_pickle=False)
        assert np.array_equal(encoded, actual)
        assert hashlib.sha256(actual.tobytes()).hexdigest() == record['decoded_sha256']
        maps.append(actual)
    assert np.array_equal(maps[0], np.load(phase / 'reference-seed42-roi.npy', allow_pickle=False))

    def close(actual, wanted):
        np.testing.assert_allclose(actual, wanted, rtol=0, atol=1e-12)

    def selection(array, q):
        values = array[valid]
        k = int(np.ceil(q * len(values)))
        cut = np.partition(values, len(values)-k)[len(values)-k]
        support = valid & (array >= cut)
        return support, dict(threshold=int(cut), selected=int(support.sum()), actual_fraction=float(support.sum()/valid.sum()))

    assert len(expected['comparisons']) == 6
    distance_checks = 0
    for (i, j), comparison in zip(itertools.combinations(range(4), 2), expected['comparisons']):
        assert comparison['offsets'] == [plans[i]['offset_yx'], plans[j]['offset_yx']]
        difference = np.abs(maps[i].astype(np.int16)-maps[j].astype(np.int16))
        close(difference[valid].mean(), comparison['mae'])
        assert int(difference[valid].max()) == comparison['max']
        close(np.corrcoef(maps[i][valid].astype(float), maps[j][valid].astype(float))[0, 1], comparison['pearson'])
        for rank in comparison['ranks']:
            a, ma = selection(maps[i], rank['q'])
            b, mb = selection(maps[j], rank['q'])
            assert [ma, mb] == rank['selections']
            close(2*(a & b).sum()/(a.sum()+b.sum()), rank['dice'])
            pa, pb = np.argwhere(a), np.argwhere(b)
            da = cKDTree(pb).query(pa, workers=1)[0]
            db = cKDTree(pa).query(pb, workers=1)[0]
            pooled = np.r_[da, db]
            close(pooled.mean(), rank['nearest_mean'])
            close(np.percentile(pooled, 95), rank['nearest_p95'])
            for radius, wanted in rank['within_radius'].items():
                radius = int(radius)
                close((da <= radius).mean(), wanted['A_to_B'])
                close((db <= radius).mean(), wanted['B_to_A'])
                close((pooled <= radius).mean(), wanted['pooled'])
            distance_checks += 1
        if 'phase8x8' in comparison:
            yy, xx = np.indices(shape); yy += y; xx += x
            for label, (dy, dx) in [('source', (0, 0)), ('baseline', plans[i]['offset_yx']), ('shifted', plans[j]['offset_yx'])]:
                phasebin = (((yy-dy) % 64)//8)*8 + ((xx-dx) % 64)//8
                count = np.bincount(phasebin[valid], minlength=64)
                sums = np.bincount(phasebin[valid], weights=difference[valid], minlength=64)
                assert count.tolist() == comparison['phase8x8'][label]['counts']
                close(sums/count, comparison['phase8x8'][label]['mae'])
    return dict(status='PASS',exact_blends=4,exact_original_baseline=True,pair_comparisons=6,rank_distance_comparisons=distance_checks,
                inference_run=False,raw_CT_reverified=False,target_physical_independence_established=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    meta = json.loads(INDEX.read_text())
    if not checked(args.bundle, meta['bundle']):
        raise ValueError('Bundle checksum/length mismatch')
    if args.destination.exists():
        raise ValueError('Choose a new replay destination; existing files are preserved')
    args.destination.mkdir(parents=True)
    restore = restore_bundle(args.bundle, args.destination, meta['members'])
    result = replay(args.destination)
    result.update(files_verified=restore['restored'],bundle_sha256=meta['bundle']['sha256'])
    (args.destination / 'REPLAY_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
