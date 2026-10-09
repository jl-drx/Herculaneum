"""Independent integral-sum reconstruction of frozen label-only selection."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SELECTION = ROOT / 'reference-selection'


def record(path):
    body = path.read_bytes()
    return {'path': str(path), 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}


def neighborhood_sums(array, y, x):
    # Compute centered 33x33 sums through independent integer integral images.
    # Fixed zero padding matches the declared out-of-image surface semantics.
    padded = np.zeros((1184, 1184), dtype=np.uint8)
    y0, x0 = max(0, y-16), max(0, x-16)
    y1, x1 = min(array.shape[0], y+1168), min(array.shape[1], x+1168)
    py, px = y0-(y-16), x0-(x-16)
    padded[py:py+y1-y0, px:px+x1-x0] = array[y0:y1, x0:x1]
    integral = np.zeros((1185, 1185), dtype=np.int32)
    integral[1:, 1:] = padded.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
    low = 1 + 3*np.arange(384)
    high = low + 33
    return (integral[np.ix_(high, high)] - integral[np.ix_(low, high)]
            - integral[np.ix_(high, low)] + integral[np.ix_(low, low)])


def main():
    plan_path = SELECTION / 'selection-plan.json'
    result_path = SELECTION / 'selection-result.json'
    plan = json.loads(plan_path.read_text())
    result = json.loads(result_path.read_text())
    assert record(plan_path)['sha256'] == result['plan_sha256']
    assert record(SELECTION / 'select_contexts.py')['sha256'] == result['code_sha256']
    for item in plan['source_bindings']:
        assert record(Path(item['path'])) == item
    with Image.open(ROOT / 'optical-acquisition/inklabels.png') as im:
        labels = np.asarray(im).copy()
    with Image.open(ROOT / 'optical-acquisition/mask.png') as im:
        surface = np.asarray(im).copy()
    assert labels.shape == surface.shape == (9791, 6064)
    assert set(np.unique(labels)) == set(np.unique(surface)) == {0, 1}
    border = np.zeros((384, 384), dtype=bool)
    border[32:352, 32:352] = True
    reconstructed = []
    selected_by_xy = {(item['y'], item['x']): item for item in result['selected']}
    for y in range(0, 8640, 1152):
        for x in range(0, 4913, 1152):
            ink_count = neighborhood_sums(labels, y, x)
            surface_count = neighborhood_sums(surface, y, x)
            known = (surface_count == 1089) & ((ink_count == 0) | (ink_count == 1089))
            truth = labels[y+1:y+1152:3, x+1:x+1152:3]
            primary = known & border
            fraction = int(surface[y:y+1152, x:x+1152].sum()) / (1152*1152)
            positive = int(np.count_nonzero(primary & (truth == 1)))
            negative = int(np.count_nonzero(primary & (truth == 0)))
            item = dict(y=y, x=x, surface_fraction=fraction, primary_positive=positive,
                        primary_negative=negative,
                        eligible=fraction >= .9 and positive >= 512 and negative >= 8192)
            original = result['candidates'][len(reconstructed)]
            assert all(item[key] == original[key] for key in item)
            reconstructed.append(item)
            if (y, x) in selected_by_xy:
                selected = selected_by_xy[y, x]
                expected = {'labels.npy': truth, 'known-mask.npy': known,
                            'primary-mask.npy': primary,
                            'surface-mask.npy': surface[y+1:y+1152:3, x+1:x+1152:3].astype(bool)}
                for bound in selected['files']:
                    path = Path(bound['path'])
                    assert record(path) == bound
                    actual = np.load(path, allow_pickle=False)
                    assert actual.dtype == expected[path.name].dtype
                    assert np.array_equal(actual, expected[path.name])
    assert len(reconstructed) == len(result['candidates']) == 40
    eligible = [item for item in reconstructed if item['eligible']]
    chosen = [eligible[0], [item for item in eligible if item['y'] >= eligible[0]['y']+2304][-1]]
    assert [(item['y'], item['x']) for item in chosen] == [(item['y'], item['x']) for item in result['selected']]
    receipt = {'status': 'PASS', 'candidate_count': 40, 'eligible_count': len(eligible),
               'selected': chosen, 'saved_arrays_exactly_reconstructed': 8,
               'method': 'Independent integer integral-image centered 33x33 sums, not min/max filtering',
               'source_bindings': [record(plan_path), record(result_path), record(Path(__file__))],
               'network_calls': 0, 'model_calls': 0, 'CT_arrays_read': 0,
               'retained_reference_labels_and_masks_read': True}
    target = HERE / 'selection-verification.json'
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'receipt': record(target), 'selected': chosen}))


if __name__ == '__main__':
    main()
