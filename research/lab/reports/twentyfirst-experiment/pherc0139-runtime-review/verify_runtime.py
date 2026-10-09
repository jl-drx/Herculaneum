"""Read-only scientific verifier; only writes a new review attempt directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
import numpy as np
from independent_geometry import decode, ensure, reconstruct
from scalar_oracle import point

HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / 'pherc0139-geometry'
PRODUCER = GEOMETRY / 'public-geometry-implementation-v2'
PLAN = GEOMETRY / 'public-geometry-plan/COORDINATE_QUERY_PLAN.json'
PLAN_SHA = '4f180f4507e77c8460c5b3a57c0683e818c8b29957c1a6d75c70dc9d53f6623d'
PRODUCER_FREEZE_SHA = 'b86b5faff357757e06f6fc9bc16ae29f6d60535885e9393419d6cb5a28c8d1aa'

def prohibited(*args, **kwargs):
    raise RuntimeError('reviewer network forbidden')

socket.create_connection = prohibited
socket.socket.connect = prohibited
socket.socket.connect_ex = prohibited

def sha(data):
    return hashlib.sha256(data).hexdigest()

def pin(path):
    data = Path(path).read_bytes()
    return {'path': str(Path(path)), 'sha256': sha(data), 'size_bytes': len(data)}

def verify_pin(record):
    ensure(pin(record['path']) == record, 'file pin mismatch: ' + record['path'])
    return Path(record['path'])

def read_json(path):
    return json.loads(Path(path).read_text())

def load_array(record, shape, dtype):
    path = verify_pin(record)
    array = np.load(path, allow_pickle=False)
    ensure(array.shape == shape and array.dtype == np.dtype(dtype), 'array schema')
    return array

def check_plan():
    ensure(pin(PLAN)['sha256'] == PLAN_SHA, 'literal plan pin')
    freeze_path = PRODUCER / 'CODE_FREEZE.json'
    ensure(pin(freeze_path)['sha256'] == PRODUCER_FREEZE_SHA, 'producer freeze pin')
    plan, freeze = read_json(PLAN), read_json(freeze_path)
    for item in plan['input_pins'] + freeze['sources'] + freeze['inputs']:
        verify_pin(item)
    verify_pin(plan['native_frame']['source'])
    ensure(plan['native_frame']['volume_id'] == '20260102150214', 'native volume identity')
    ensure(plan['native_frame']['shape_xyz'] == [26511, 26511, 76953], 'native bounds')
    ensure(len(plan['requests']) == 6 and len(plan['contexts']) == 2, 'plan cardinality')
    for i, request in enumerate(plan['requests']):
        x, y, width, height = request['window_xywh']
        source_contract = read_json(GEOMETRY / 'structural/run' / (request['source']['name'] + '.contract.json'))
        ensure(source_contract['storage_offsets'] == [8] and source_contract['width'] == 1444
               and source_contract['height'] == 1404 and source_contract['compression'] == 1
               and source_contract['byte_order'] == '<' and source_contract['sample_format'] == 3
               and source_contract['bits_per_sample'] == 32, 'independent structural preconditions')
        first = 8 + 4 * (y * 1444 + x)
        last = 8 + 4 * ((y + height - 1) * 1444 + x + width) - 1
        ensure(request['range_inclusive'] == [first, last], 'derived byte extent')
        ensure(request['source'] == source_contract['source'] and width == height == 80, 'source/window contract')
        ensure(request['context'] == plan['contexts'][i // 3]['id']
               and request['id'] == request['context'] + '-' + 'xyz'[i % 3] + '.tif', 'request identity/order')
        ensure(last - first + 1 == request['expected_body_bytes'] == 456624
               and request['reserved_body_bytes'] == 456625 and request['row_stride_bytes'] == 5776,
               'body arithmetic')
        ensure(request['useful_coordinate_bytes'] == 25600 and request['uninterpreted_gap_bytes'] == 431024,
               'decoded/opaque accounting')
    return plan, freeze

def check_transport(plan, result, run):
    ledger = read_json(verify_pin(result['transport_ledger']))
    ensure(ledger['status'] == 'PASS' and len(ledger['requests']) == 6, 'ledger completion')
    ensure(ledger['reserved_bytes_all_attempts'] == 2739750 and ledger['charged_bytes'] == 2739744, 'ledger totals')
    bodies = []
    receipts = []
    for request, actual in zip(plan['requests'], ledger['requests']):
        ensure(actual['status'] == 'PASS' and actual['http_status'] == 206, 'range status')
        ensure(actual['id'] == request['id'] and actual['url'] == request['source']['url']
               and actual['range_inclusive'] == request['range_inclusive']
               and actual['if_match'] == request['source']['strong_etag'], 'request binding')
        headers = actual['response_headers']
        lo, hi = request['range_inclusive']
        ensure(headers['Content-Range'] == f'bytes {lo}-{hi}/8109638'
               and headers['ETag'] == request['source']['strong_etag']
               and headers.get('Content-Encoding', 'identity').lower() == 'identity', 'response identity')
        ensure(headers.get('Content-Length', '456624') == '456624', 'response length')
        ensure(actual['reserved_body_bytes'] == 456625 and actual['retained_body_bytes'] == 456624, 'range accounting')
        path = verify_pin(actual['body'])
        ensure(path == run / (request['id'] + '.body'), 'literal body path')
        body = path.read_bytes()
        ensure(len(body) == 456624, 'actual body bytes')
        bodies.append(body)
        receipts.append(actual['body'])
    ensure(sum(map(len, bodies)) == 2739744, 'body total')
    return bodies, receipts

def verify(result_path, result_sha, go_sha, output):
    started = time.monotonic()
    ensure(output.parent == HERE and not output.exists(), 'new review directory only')
    freeze_path = HERE / 'CHECKER_FREEZE.json'
    checker_freeze = read_json(freeze_path)
    for item in checker_freeze['sources'] + checker_freeze['contracts']:
        verify_pin(item)
    output.mkdir()
    receipt = {'status': 'RUNNING', 'checker_freeze': pin(freeze_path), 'scientific_model_calls': 0,
               'network_requests': 0, 'native_CT_reads': 0, 'label_or_SV_reads': 0, 'contexts': []}
    try:
        ensure(sys.version.split()[0] == '3.12.14' and np.__version__ == '2.5.3', 'frozen runtime')
        ensure(result_path == PRODUCER / 'run/result.json', 'literal scientific result path')
        ensure(pin(result_path)['sha256'] == result_sha, 'root provided result pin')
        receipt['producer_result'] = pin(result_path)
        result = read_json(result_path)
        plan, _ = check_plan()
        ensure(result['status'] == 'PASS' and result['plan'] == pin(PLAN)
               and result['code_freeze'] == pin(PRODUCER / 'CODE_FREEZE.json'), 'producer result binding')
        go_path = verify_pin(result['root_go'])
        ensure(result['root_go']['sha256'] == go_sha, 'root provided GO pin')
        go = read_json(go_path)
        ensure(go['status'] == 'GO' and go['authorized_action'] == 'acquire_and_reconstruct_public_geometry'
               and go['plan'] == pin(PLAN) and go['code_freeze'] == result['code_freeze'], 'root authorization')
        for key, expected in {'request_count': 6, 'decoded_window_channel_count': 6,
                              'numerically_decoded_coordinate_bytes': 153600,
                              'reserved_bytes_all_attempts': 2739750, 'charged_bytes': 2739744,
                              'retained_range_body_bytes': 2739744, 'native_CT_requests': 0,
                              'mask_label_requests': 0, 'model_calls': 0,
                              'historical_private_validity_authenticated': False}.items():
            ensure(result[key] == expected, 'producer scope field: ' + key)
        ensure(result['output_claim'] == plan['output_claim'], 'producer claim contract')
        run = result_path.parent
        bodies, receipt['range_pins'] = check_transport(plan, result, run)
        ensure(len(result['contexts']) == 2, 'result contexts')
        scale = read_json(GEOMETRY / 'text-run/transformed_mesh_meta.body')['scale']
        for ci, (context, actual) in enumerate(zip(plan['contexts'], result['contexts'])):
            ensure(actual['id'] == context['id'] and actual['mesh_window_xywh'] == context['mesh_window_xywh'], 'context identity')
            decoded = np.stack([decode(bodies[3 * ci + channel]) for channel in range(3)], axis=-1)
            saved = load_array(actual['outputs']['xyz_window'], (80, 80, 3), 'float32')
            ensure(np.array_equal(decoded.view(np.uint32), saved.view(np.uint32)), 'window exact selected bits')
            with np.errstate(all='ignore'):
                mesh, bounded, counts, coverage = reconstruct(decoded, context, scale, plan['native_frame']['shape_xyz'], point)
            ensure(read_json(verify_pin(actual['outputs']['counts'])) == counts == actual['counts'], 'complete disjoint count reconstruction')
            array_pins = {}
            for name, array in [('mesh-stencil-support', mesh), ('mesh-native-bounds-support', bounded)]:
                producer_array = load_array(actual['outputs'][name], (384, 384), 'bool')
                ensure(np.array_equal(array, producer_array), 'complete output mask: ' + name)
                array_pins[name] = {'producer_file': actual['outputs'][name], 'independent_raw_sha256': sha(array.tobytes())}
            ensure(coverage['scalar_points_checked'] == 169, 'fixed scalar coverage')
            receipt['contexts'].append({'id': context['id'], 'counts': counts, 'masks': array_pins,
                                        'xyz_bits_sha256': sha(decoded.tobytes()), 'scalar_points_checked': 169,
                                        'all68_depth_points_checked': 169, 'scalar_sample_sha256': sha(coverage['sample_bytes']),
                                        'fine_samples_reconstructed': 2359296, 'coarse_outputs_per_mask': 147456})
        ensure(result['peak_rss_bytes'] <= 536870912 and result['elapsed_seconds'] < 240,
               'reported producer resource ceilings')
        actual_size = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
        ensure(actual_size <= 33554432, 'actual retained producer output cap')
        ensure(result['final_resource_check']['peak_rss_bytes'] <= 536870912
               and result['final_resource_check']['elapsed_seconds'] < 230, 'internal producer resource check')
        # Inputs and source identities are rechecked after reading the outputs.
        check_plan()
        for item in checker_freeze['sources'] + checker_freeze['contracts']:
            verify_pin(item)
        verify_pin(receipt['producer_result'])
        receipt.update(status='PASS', decoded_channel_windows=6, decoded_coordinate_bytes=153600,
                       opaque_gap_bytes=2586144, independent_fine_samples=4718592,
                       all68_depth_sample_points=338, scalar_point_comparisons=338,
                       source_mask_or_bounds_changes=0, producer_retained_output_bytes=actual_size)
    except BaseException as exc:
        receipt.update(status='STOP', error_type=type(exc).__name__, reason=str(exc))
    receipt['elapsed_seconds'] = time.monotonic() - started
    receipt['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    target = output / 'verification.json'
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': receipt['status'], 'verification': pin(target), 'reason': receipt.get('reason')}))
    return 0 if receipt['status'] == 'PASS' else 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--result-sha', required=True)
    parser.add_argument('--go-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(verify(args.result, args.result_sha, args.go_sha, args.output))
