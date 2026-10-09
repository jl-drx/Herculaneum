"""Independently reconstruct pooled inputs from retained CT bands; no acquisition."""
import argparse
from pathlib import Path
import hashlib
import json
import struct

import numpy as np

HERE = Path(__file__).resolve().parent


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def verify(bound):
    assert record(bound['path']) == bound
    return Path(bound['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.provenance.read_text())
    assert receipt['status'] == 'PASS'
    plan = json.loads(verify(receipt['plan']).read_text())
    verify(receipt['reader'])
    ledger = json.loads(verify(receipt['request_ledger']).read_text())
    requests = ledger['requests']
    assert len(requests) == 255 and all(r['success'] for r in requests)
    assert all(r['http_status'] == 206 for r in requests)
    assert sum(r['charged_response_bytes'] for r in requests) == ledger['charged_response_bytes']
    assert sum(r['retained_body_bytes'] for r in requests) == ledger['retained_body_bytes']
    assert ledger['charged_response_bytes'] == ledger['retained_body_bytes']
    for entry in requests:
        bound = {'path': entry['retained_path'], 'bytes': entry['retained_body_bytes'],
                 'sha256': entry['body_sha256']}
        verify(bound)
        start, end = entry['range_inclusive']
        assert entry['request_cap_bytes'] == entry['retained_body_bytes'] == end-start+1
    assert [layer['layer_index'] for layer in receipt['source_layers']] == list(range(7, 58))
    assert receipt['source_groups'] == [list(range(k, k+3)) for k in range(7, 58, 3)]
    assert receipt['pooled_source_center_indices'] == list(range(8, 57, 3))
    selection = json.loads((HERE.parents[1] / 'reference-selection/selection-result.json').read_text())
    by_case = {case['id']: case for case in selection['selected']}
    reconstructed = {}
    for output in receipt['outputs']:
        y, x, height, width = by_case[output['case_id']]['native_bbox_yxhw']
        assert output['bbox_source_xywh'] == [x, y, width, height]
        assert (height, width) == (1152, 1152)
        assert output['shape_zyx'] == [17, 384, 384]
        pooled = np.empty((17, 384, 384), dtype=np.float32)
        total = np.zeros((384, 384), dtype=np.uint32)
        for ordinal, layer in enumerate(receipt['source_layers']):
            assert (layer['width'], layer['height'], layer['dtype']) == (6064, 9791, '<u2')
            matches = [band for band in layer['row_bands']
                       if band['y_start'] <= y and y+height <= band['y_stop_exclusive']]
            assert len(matches) == 1
            band = matches[0]
            bound = band['file']
            # Full-file hashes were already verified through the complete ledger.
            assert any(r['retained_path'] == bound['path'] and r['body_sha256'] == bound['sha256']
                       and r['retained_body_bytes'] == bound['bytes'] for r in requests)
            start, end = band['range']['range_inclusive']
            assert start == layer['strip_offset'] + band['y_start']*6064*2
            assert end+1 == layer['strip_offset'] + band['y_stop_exclusive']*6064*2
            assert bound['bytes'] == (band['y_stop_exclusive']-band['y_start'])*6064*2
            layer_requests = [r for r in requests if r['url'] == layer['url']]
            assert len(layer_requests) == 5
            assert all(r['etag'] == layer['etag'] and r['object_total_bytes'] == layer['object_total_bytes']
                       for r in layer_requests)
            assert layer_requests[0]['if_match'] is None
            assert all(r['if_match'] == layer['etag'] for r in layer_requests[1:])
            header = Path(layer_requests[0]['retained_path']).read_bytes()
            assert len(header) == 8 and header[:4] == b'II*\x00'
            assert struct.unpack_from('<I', header, 4)[0] == layer['ifd_offset']
            raw = np.memmap(bound['path'], mode='r', dtype='<u2',
                shape=(band['y_stop_exclusive']-band['y_start'], 6064))
            native = raw[y-band['y_start']:y-band['y_start']+height, x:x+width]
            # Explicit 27 residue-class additions, independent of reshape+sum.
            for dy in range(3):
                for dx in range(3):
                    total += native[dy::3, dx::3]
            del native, raw
            if ordinal % 3 == 2:
                pooled[ordinal//3] = (total.astype(np.float64)/6939).astype(np.float32)
                total.fill(0)
        retained = np.load(verify(output['file']), allow_pickle=False)
        assert retained.dtype == np.float32 and retained.shape == pooled.shape
        assert np.array_equal(retained.view(np.uint32), pooled.view(np.uint32))
        reconstructed[output['case_id']] = {'output': output['file'], 'bitwise_float32_equal': True,
            'min': float(pooled.min()), 'max': float(pooled.max())}
    assert set(reconstructed) == {'frag4-context1', 'frag4-context2'}
    result = {'status': 'PASS', 'provenance': record(args.provenance),
        'checker': record(Path(__file__)), 'outputs': reconstructed,
        'method': 'Explicit residue-class uint32 sums over retained little-endian row bands; one float64 division by6939 and float32 cast',
        'retained_requests_verified': len(requests), 'network_calls': 0, 'model_calls': 0,
        'new_acquisitions': 0}
    target = HERE / 'retained-ct-verification.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'receipt': record(target)}))


if __name__ == '__main__':
    main()
