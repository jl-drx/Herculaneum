"""Reference-only instance identity diagnostic, not an ink/model evaluation."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def identity_scores(truth, prediction):
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError('Expected equal 3D volumes')
    if truth.dtype.kind not in 'ui' or prediction.dtype.kind not in 'ui':
        raise ValueError('Integer instance labels required')
    if np.any(truth < 0) or np.any(prediction < 0):
        raise ValueError('Negative instance labels')
    support = truth > 0
    total = int(support.sum())
    if not total:
        raise ValueError('No labelled reference support')
    gids, gi = np.unique(truth[support], return_inverse=True)
    pids, pi = np.unique(prediction[support], return_inverse=True)
    codes, counts = np.unique(gi.astype(np.int64) * len(pids) + pi, return_counts=True)
    row, col = codes // len(pids), codes % len(pids)
    keep = pids[col] != 0
    rr, cc, nn = row[keep], col[keep], counts[keep]
    row_total = np.bincount(rr, weights=nn, minlength=len(gids))
    col_total = np.bincount(cc, weights=nn, minlength=len(pids))
    row_max = np.zeros(len(gids), dtype=np.int64)
    col_max = np.zeros(len(pids), dtype=np.int64)
    np.maximum.at(row_max, rr, nn)
    np.maximum.at(col_max, cc, nn)
    per_reference_fragments = np.bincount(rr, minlength=len(gids))
    per_prediction_references = np.bincount(cc, minlength=len(pids))
    retained = int(nn.sum())
    return {
        'reference_voxels': total, 'reference_instances': len(gids),
        'retained_reference_fraction': retained / total,
        'abstained_reference_fraction': (total - retained) / total,
        'merge_minority_fraction_of_reference': float((col_total - col_max).sum() / total),
        'fragmented_survivor_fraction_of_reference': float((row_total - row_max).sum() / total),
        'mixed_prediction_instances': int((per_prediction_references > 1).sum()),
        'split_reference_instances': int((per_reference_fragments > 1).sum()),
        'prediction_instances_on_reference': int((col_total > 0).sum()),
        'positive_voxels_outside_reference': int(((prediction > 0) & ~support).sum()),
        'outside_reference_interpretation': 'unannotated/unsupported here, not automatically a false positive',
        'reference_ids': [int(v) for v in gids],
    }


def qualify():
    a = np.zeros((6, 8, 8), np.uint16)
    a[:, 1:3, :] = 7
    a[:, 5:7, :] = 23
    n = 0
    r = identity_scores(a, a)
    assert r['retained_reference_fraction'] == 1 and r['merge_minority_fraction_of_reference'] == 0
    n += 1
    p = np.where(a == 7, 99, np.where(a == 23, 3, 0)).astype(np.uint16)
    assert identity_scores(a, p)['fragmented_survivor_fraction_of_reference'] == 0
    assert identity_scores(a, p)['merge_minority_fraction_of_reference'] == 0
    n += 1
    p = (a > 0).astype(np.uint16)
    r = identity_scores(a, p)
    assert np.array_equal(a > 0, p > 0) and r['merge_minority_fraction_of_reference'] == .5
    n += 1
    p = a.copy(); p[3:][p[3:] == 7] = 77
    assert identity_scores(a, p)['fragmented_survivor_fraction_of_reference'] == .25
    n += 1
    p = a.copy(); p[p == 7] = 0
    r = identity_scores(a, p)
    assert r['retained_reference_fraction'] == .5 and r['abstained_reference_fraction'] == .5
    n += 1
    p = a.copy(); p[:, 3, :] = 55
    r = identity_scores(a, p)
    assert r['positive_voxels_outside_reference'] == 48 and r['merge_minority_fraction_of_reference'] == 0
    n += 1
    try:
        identity_scores(np.zeros_like(a), np.zeros_like(a))
    except ValueError:
        n += 1
    else:
        raise AssertionError('Empty reference accepted')
    return {'status': 'PASS', 'checks': n,
            'cases': ['identity', 'ID permutation', 'same binary foreground with merged sheets',
                      'split sheet', 'missing sheet', 'unknown exterior', 'empty-reference rejection']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.self_test:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(qualify(), indent=2) + '\n')
        return
    import nrrd
    from scipy import ndimage
    import colorsys
    from PIL import Image, ImageDraw
    start = time.monotonic()
    names = ['00768-02152-03536', '00768-02408-03536']
    masks, images, headers, receipts = [], [], [], []
    for name in names:
        for kind, container in [('mask', masks), ('volume', images)]:
            path = args.inputs / (name + '-' + kind + '.raw')
            receipt = json.loads(path.with_suffix('.receipt.json').read_text())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256']
            data, header = nrrd.read(str(path), index_order='F')
            assert data.shape == (256, 256, 256) and data.dtype == np.uint16
            container.append(data)
            headers.append({'file': path.name, 'nrrd_order': 'F',
                            **{k: np.asarray(header[k]).tolist() if not isinstance(header[k], str) else header[k]
                               for k in ['type', 'sizes', 'space directions', 'space origin'] if k in header}})
            receipts.append({'file': path.name, 'sha256': receipt['sha256'], 'bytes': receipt['bytes']})
    # Use NRRD array axes. The shared boundary is axis 1 independently of a
    # possible external XYZ/ZYX naming convention; no scanner transform is claimed.
    assert np.array_equal(np.asarray(headers[2]['space origin']) - headers[0]['space origin'], [0, 256, 0])
    for i in [0, 2]:
        assert np.array_equal(headers[i]['space directions'], np.eye(3))
    truth = np.concatenate(masks, axis=1)
    binary = truth > 0
    scores = {'identity_oracle': identity_scores(truth, truth)}
    for connectivity, rank in [('6-connected', 1), ('26-connected', 3)]:
        components, count = ndimage.label(binary, structure=ndimage.generate_binary_structure(3, rank))
        scores[connectivity] = dict(identity_scores(truth, components), connected_components=int(count))
        del components
    left, right = masks[0][:, -1, :], masks[1][:, 0, :]
    both = (left > 0) & (right > 0)
    face = {
        'axis': 1, 'both_endpoints_labelled': int(both.sum()),
        'same_id_contacts': int(((left == right) & both).sum()),
        'different_id_contacts': int(((left != right) & both).sum()),
        'ids_shared_between_cubes': sorted(set(map(int, np.unique(masks[0]))) & set(map(int, np.unique(masks[1]))) - {0}),
        'interpretation': 'Label-interface contacts, not independently adjudicated physical sheet switches.'
    }
    args.output.mkdir(parents=True)
    ct = np.concatenate(images, axis=1)
    ids = np.unique(truth)
    canvas = Image.new('RGB', (1064, 610), 'white')
    draw = ImageDraw.Draw(canvas)
    palette = np.asarray([[0,0,0]] + [
        [round(v*255) for v in colorsys.hsv_to_rgb((i*.61803398875)%1, .72, .95)]
        for i in range(1, len(ids))], dtype=np.uint8)
    for row, index in enumerate([64, 192]):
        gray = (ct[index].T.astype(np.uint32) * 255 // 65535).astype(np.uint8)
        label_image = palette[np.searchsorted(ids, truth[index].T)]
        for col in range(2):
            left, top = 12 + 532*col, 34 + 294*row
            panel = Image.fromarray(gray if col == 0 else label_image).convert('RGB')
            canvas.paste(panel, (left, top))
            draw.text((left, top-20), ('CT fixed range 0..65535' if col == 0 else 'Reference sheet IDs; zero shown black') + f'; axis0={index}', fill='black')
            draw.line((left+256,top,left+256,top+255),fill='red',width=1)
    draw.text((12,595), 'Public development pair; NRRD array axes; red = file boundary. Reference labels are not independent ink truth.', fill='black')
    canvas.save(args.output/'REFERENCE_PANELS.png')
    result = {
        'status': 'COMPLETE', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'inputs': receipts, 'headers': headers, 'joined_shape': list(truth.shape),
        'scores': scores, 'shared_face': face, 'seconds': time.monotonic()-start,
        'claim': 'Reference-only oracle foreground-to-instance diagnostic; no released surface predictor, new ink model, physical area accuracy or fresh test evaluation.',
        'limits': 'Two selected adjacent public development cubes from one parent. Upstream labels omit small components. Zero is not independently established empty space. NRRD coordinates are not a qualified physical scanner transform.'
    }
    (args.output/'RESULTS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'scores': scores, 'shared_face': face, 'seconds': result['seconds']}, indent=2))


if __name__ == '__main__':
    main()
