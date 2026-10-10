"""Independent whole-crop nearest-response check, no EDT or model execution."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument('--lab', type=Path, required=True)
ap.add_argument('--review', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
work = a.lab / 'reports/fortysixth-experiment/work'
valid = np.load(work / 'valid.npy', allow_pickle=False)[105:2560, 380:6021]
source = np.load(next((a.lab / 'reports/fortyfifth-experiment/ph343-mesh-replay').glob('*_node_source.npz')))['source']
own = np.isin(source.repeat(20, 0).repeat(20, 1), [1, 4])
expected = json.loads((a.review / 'spatial-persistence/metrics.json').read_text())['results'][0]['fractions'][1]
assert expected['q'] == .1
points = {}
for name, token in [('dnative', 'dnative'), ('seed42', 'hybrid_3d2d-seed42'), ('seed43', 'hybrid_3d2d-seed43')]:
    maps = {}
    for phase in ['plain', 'padded']:
        im = np.array(Image.open(next((work / ('maps_' + phase)).glob('*__' + token + '__*.tif'))))
        maps[phase] = im if phase == 'plain' else im[40:2600, 200:6920]
    maps['merged'] = np.where(own, maps['padded'], maps['plain'])
    for phase, im in maps.items():
        key = name + '_' + phase
        im = im[105:2560, 380:6021]
        values = im[valid]
        k = int(np.ceil(.1 * values.size))
        threshold = int(np.partition(values, values.size-k)[values.size-k])
        assert threshold == expected['selections'][key]['threshold_uint8']
        points[key] = np.argwhere(valid & (im >= threshold))
        assert len(points[key]) == expected['selections'][key]['selected']
pairs = [(n+' phase', n+'_plain', n+'_padded') for n in ['dnative','seed42','seed43']]
pairs += [(x+' / '+y, x+'_merged', y+'_merged') for x,y in [('dnative','seed42'),('dnative','seed43'),('seed42','seed43')]]
out = []
for label, ka, kb in pairs:
    pa, pb = points[ka], points[kb]
    ta, tb = cKDTree(pa), cKDTree(pb)
    assert np.all(ta.query(pa, workers=2)[0] == 0)
    da, db = tb.query(pa, workers=2)[0], ta.query(pb, workers=2)[0]
    both = np.concatenate([da, db])
    m = expected['comparisons'][label]
    for radius in [0, 2, 4, 8]:
        for direction, values in [('A_to_B', da), ('B_to_A', db), ('pooled_symmetric', both)]:
            assert abs(float((values <= radius).mean()) - m['within_radius_native_px'][str(radius)][direction]) < 1e-12
    assert abs(both.mean() - m['symmetric_nearest_mean_px']) < 1e-12
    assert abs(np.percentile(both,95) - m['symmetric_nearest_p95_px']) < 1e-12
    out.append({'pair':label, 'selected_A':len(pa), 'selected_B':len(pb),
                'mean':float(both.mean()), 'p95':float(np.percentile(both,95)),
                'within8':float((both<=8).mean()), 'status':'PASS'})
    print(label, 'PASS', flush=True)
with a.out.open('x') as f:
    json.dump({'status':'PASS','scope':'All six whole-crop top-10% pair distances verified independently with cKDTree; no inference or accuracy claim.', 'checks':out},f,indent=2)
    f.write('\n')
