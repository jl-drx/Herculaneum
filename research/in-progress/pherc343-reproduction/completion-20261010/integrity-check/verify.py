"""Read-only post-compute checks; no inference or scientific success declaration."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import cv2
import numpy as np
import tifffile

P = Path(__file__).resolve().parent.parent
OLD = P.parent / 'fortyfifth-experiment'
STEM = 'concat_w047-w048_R5B2_z9500-11000'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


state = json.loads((P / 'PIPELINE_STATUS.json').read_text())
assert state['status'] == 'COMPUTE_COMPLETE_AWAITING_REVIEW'
assert len(state['stages']) == 2 and all(x['exit_code'] == 0 for x in state['stages'])
pins = json.loads((Path(state['run_directory']) / 'INPUTS.json').read_text())
for record in list(pins.values()) + state['outputs']:
    path = Path(record['path'])
    assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
maps = []; final = None
for receipt in sorted((P / 'work').glob('maps_*/*.receipt.json')):
    record = json.loads(receipt.read_text()); path = Path(record['output']['path'])
    assert record['status'] == 'PASS'
    assert path.stat().st_size == record['output']['bytes'] and sha(path) == record['output']['sha256']
    array = tifffile.imread(path) if path.suffix == '.tif' else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert array.dtype == np.uint8 and list(array.shape) == record['shape']
    assert hashlib.sha256(array.tobytes()).hexdigest() == record['decoded_sha256']
    maps.append({'model': record['model'], 'phase': record['phase'], 'status': 'PASS',
                 'encoded_sha256': record['output']['sha256'], 'decoded_sha256': record['decoded_sha256']})
    if (record['model'], record['phase']) == ('hecate', 'padded'):
        final = array
assert len(maps) == 8 and final is not None
root = P / 'work/hecate-checkpoints'
identity = json.loads((root / 'BINDING.json').read_text())
binding_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
checkpoints = []
for directory in sorted(root.glob('rows-*')):
    record = json.loads((directory / 'MANIFEST.json').read_text())
    path = directory / 'prefix.npy'
    assert record['binding_sha256'] == binding_hash and sha(path) == record['prefix_sha256']
    array = np.load(path, allow_pickle=False)
    assert array.dtype == np.uint8 and list(array.shape) == record['shape']
    assert np.array_equal(array, final[:record['rows']])
    checkpoints.append({'rows': record['rows'], 'sha256': record['prefix_sha256'], 'matches_final_map': True})
assert checkpoints[-1]['rows'] == 2340
ours = json.loads((P / 'work/image' / f'{STEM}.json').read_text())
reference = json.loads((OLD / 'source-evidence/pherc343/outputs' / f'{STEM}.json').read_text())
for record in ours['outputs'].values():
    assert sha(P / 'work/image' / record['file']) == record['sha256']
reference_path = OLD / 'source-evidence' / f'{STEM}.png'
assert sha(reference_path) == reference['outputs']['image']['sha256']


def panel(path, metadata):
    image = cv2.imread(str(path)); box = metadata['outputs']['image']['data_panel']
    return image[box['y']:box['y'] + box['height'], box['x']:box['x'] + box['width']]


a = panel(reference_path, reference)
b = panel(P / 'work/image' / f'{STEM}.png', ours)
assert a.shape == b.shape == (2455, 5641, 3)
valid = np.load(P / 'work/valid.npy')[105:2560, 380:6021][::-1, ::-1]
delta = np.abs(a.astype(np.int16) - b.astype(np.int16))


def yellow(image):
    blue, green, red = np.moveaxis(image.astype(np.int16), -1, 0)
    return valid & (red - blue > 12) & (green - blue > 8) & (np.abs(red - green) < 50)


x, y = yellow(a), yellow(b)
metrics = {'valid_rgb_mae': int(delta[valid].sum()) / (int(valid.sum()) * 3),
           'valid_rgb_max_difference': int(delta[valid].max()),
           'reference_display_visible_pixels': int(x.sum()),
           'reproduction_display_visible_pixels': int(y.sum()),
           'visible_display_intersection': int((x & y).sum()),
           'visible_display_union': int((x | y).sum()),
           'reference_visible_rgb_mae': int(delta[x].sum()) / (int(x.sum()) * 3)}
expected = json.loads((P / 'IMAGE_COMPARISON.json').read_text())
assert all(abs(value - expected[name]) < 1e-12 for name, value in metrics.items())
metrics['display_mask_iou'] = metrics['visible_display_intersection'] / metrics['visible_display_union']
assert ours['ct']['shape'] == reference['ct']['shape'] == [66, 2560, 6720]
assert ours['ct']['stretch'] == reference['ct']['stretch']
assert ours['warnings'] == [f'--ink-stats was computed for {STEM}, the CT is {STEM}_66.zarr']
report = {'status': 'PASS', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
          'pipeline_finished_at_utc': state['finished_at_utc'], 'frozen_files_unchanged': len(pins),
          'maps': maps, 'saved_checkpoint_generations': len(checkpoints), 'checkpoint_checks': checkpoints,
          'composition_outputs_verified': len(ours['outputs']), 'comparison_recomputed': metrics,
          'warning': 'Source emits warning solely for differing surface/CT basenames. Frozen statistics are unchanged; CT shape and stretch match the reference.',
          'scope': 'Completed-output integrity and fixed display comparison; detailed all-region scientific review and phase46 sealing remain pending. No new inference.',
          'verifier_sha256': sha(Path(__file__))}
out = Path(__file__).parent / 'VERIFICATION.json'
with out.open('x') as stream:
    json.dump(report, stream, indent=2); stream.write('\n')
print(json.dumps({k: report[k] for k in ['status', 'pipeline_finished_at_utc', 'frozen_files_unchanged',
      'saved_checkpoint_generations', 'composition_outputs_verified', 'comparison_recomputed', 'scope']}, indent=2))
