"""Check completed audit metadata and seal; never open scientific arrays."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def pin(path):
    path = Path(path)
    assert path.is_relative_to(ROOT)
    return {
        'path': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'size_bytes': path.stat().st_size,
    }


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    assert not (ROOT / 'RETENTION_MANIFEST.json').exists()
    assert not (ROOT / 'FINAL_METADATA_REVIEW.json').exists()
    assert not (ROOT / 'STOP.json').exists()
    freeze = read('CODE_FREEZE.json')
    receipt = read('verification.json')
    derived = read('MODEL_INPUT_DERIVATION.json')
    diagnostic = read('tile-normalization-diagnostics.json')
    contract = read('MODEL_INPUT_CONTRACT.json')
    checks = []

    def check(ok, name):
        checks.append({'check': name, 'pass': bool(ok)})
        assert ok, name

    check(freeze['source'] == pin(freeze['source']['path']) == receipt['source'],
          'original frozen verifier and completed receipt agree')
    check(receipt['check_count'] == len(receipt['checks']) == 871
          and all(x['pass'] for x in receipt['checks']), 'all 871 audit checks pass')
    for key in ('contract', 'diagnostics'):
        check(pin(receipt[key]['path']) == receipt[key], 'audit receipt pin ' + key)
    for key in ('canonical_contract', 'diagnostic_contract', 'diagnostic_receipt', 'diagnostics', 'source'):
        check(pin(derived[key]['path']) == derived[key], 'derivation pin ' + key)
    check(derived['status'] == 'PASS' and derived['pairs_bound'] == 150,
          'derivation status and pair count')
    check(all(derived[k] == 0 for k in ('array_reads', 'normalizer_calls', 'model_calls', 'network_requests')),
          'JSON-only derivation declares no scientific execution')
    check(contract['authoritative_model_input_contract'] is True
          and contract['status'] == 'PASS_NORMALIZED_TENSOR_DEPTH_REVERSAL_CONTRACT',
          'canonical contract explicitly authoritative')
    policy = contract['input_policy']
    check(policy['adapter_read_direction'] == 'forward'
          and policy['model_depth_indices'] == list(range(16, -1, -1))
          and policy['tensor_axis'] == 2
          and policy['no_raw_input_depth_reversal_in_execution'] is True
          and policy['no_repeated_normalization'] is True,
          'canonical operational boundary normalizes original once then flips normalized depth')
    lookup = {(x['case_id'], x['tile_index']): x for x in diagnostic['tiles']}
    check(len(lookup) == 150, 'diagnostic metadata has 150 distinct tile records')
    pair_count = 0
    for case in contract['cases']:
        for index, pair in enumerate(case['paired_model_input_hashes']):
            row = lookup[(case['id'], index)]
            original = case['forward_inputs'][index]
            reversed_input = case['reversed_inputs'][index]
            check(pair['bbox_roi_xywh'] == row['bbox_roi_xywh']
                  == original['bbox_roi_xywh'] == reversed_input['bbox_roi_xywh']
                  and pair['forward_input_sha256'] == row['forward_input_sha256'] == original['input_sha256']
                  and pair['reversed_input_sha256'] == row['depth_permuted_forward_sha256']
                  == reversed_input['input_sha256'],
                  f"canonical normalized permutation pair {case['id']} tile {index}")
            pair_count += 1
    check(pair_count == 150, 'all 150 canonical pairs bound')
    result = {
        'status': 'PASS', 'checks': checks, 'check_count': len(checks),
        'source': pin(Path(__file__)), 'canonical_contract': pin(ROOT / 'MODEL_INPUT_CONTRACT.json'),
        'scope': 'local completed JSON and source integrity only; no scientific array reads or calculations',
        'array_reads': 0, 'normalizer_calls': 0, 'model_calls': 0, 'weight_decodes': 0, 'network_requests': 0,
    }
    (ROOT / 'FINAL_METADATA_REVIEW.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    files = [pin(p) for p in sorted(ROOT.iterdir()) if p.is_file()]
    manifest = {
        'status': 'SEALED', 'scope': 'Phase24 canonical depth reversal input contract; preserves diagnostic v1',
        'file_count': len(files), 'total_bytes': sum(x['size_bytes'] for x in files), 'files': files,
    }
    (ROOT / 'RETENTION_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'PASS', 'metadata_checks': len(checks), 'manifest': pin(ROOT / 'RETENTION_MANIFEST.json'),
                      'file_count': len(files), 'total_bytes': manifest['total_bytes']}))


if __name__ == '__main__':
    main()
