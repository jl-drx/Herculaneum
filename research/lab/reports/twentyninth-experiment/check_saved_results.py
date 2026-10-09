"""Read-only result/contract reconciliation; not a rerun or independent-agent review."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def record(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    for name in ('SOURCE_FREEZE.json', 'SOURCE_FREEZE_V2.json'):
        for expected in json.loads((HERE/name).read_text())['files']:
            assert record(Path(expected['path'])) == expected
    first = json.loads((HERE/'run-v1/RUN_RECEIPT.json').read_text())
    last = json.loads((HERE/'run-v2/RUN_RECEIPT.json').read_text())
    for receipt in (first, last):
        for expected in receipt['outputs'] + [receipt['source_freeze']]:
            assert record(Path(expected['path'])) == expected
        assert receipt['tests'] == 26 and receipt['failures'] == 0 and receipt['skips'] == 0
        assert receipt['actual_candidate_pixels'] == receipt['scientific_model_forwards'] == receipt['GPU_spend_GBP'] == 0
    assert first['status'] == 'FAIL' and first['errors'] == 1
    assert first['network_or_process_attempts_denied'] == ['socket.__new__']
    assert last['status'] == 'PASS' and last['errors'] == 0 and last['network_or_process_attempts_denied'] == []
    obs = json.loads((HERE/'run-v2/OBSERVATIONS.json').read_text())
    # Scalar combinatorics: 2^4 binary labels minus the2one-class vectors;3^4score assignments.
    assert obs['exhaustive_rank_cases'] == (2**4 - 2) * 3**4 == 1134
    assert obs['upstream_uint8_oracle_cases'] == 100
    assert obs['coverage_rows'] == dict(declared=[1, 3], upstream_returns_all_rows=4, explicit_covered_rows=2)
    assert obs['small_sample_mislabels']['clear_mislabeled_ink'] == 5 - 2 == 3
    assert obs['partial_band']['available_layers'] == 6 < obs['partial_band']['required_layers'] == 10
    assert obs['float_quantization'] == dict(upstream_auc=.5, upstream_ap=.5, exact_auc=1., exact_ap=1.)
    assert obs['upstream_opposite_endian_uint16_max'] == 65535.
    assert obs['tile_index_gap'] == dict(max_id_plus_one=5, occupied_tiles=4)
    groups = dict(Frag1='Paris2', Frag2='Paris2', Frag3='Paris1', Frag4='Paris1', Frag5='PHerc1667', Frag6='PHerc51')
    folds = obs['conservative_group_folds']
    assert len(folds) == len(set(groups.values())) == 4
    for fold in folds:
        assert set(fold['test']) == {k for k,v in groups.items() if v == fold['group']}
        assert set(fold['train']) == {k for k,v in groups.items() if v != fold['group']}
    acq = json.loads((HERE/'acquisition-v1/SOURCE_RECEIPT.json').read_text())
    assert acq['status'] == 'PASS' and acq['requests'] == 10 and acq['returned_body_bytes'] == 75837
    for item in acq['sources']:
        assert record(Path(item['path'])) == item
    assert (HERE/'UPSTREAM_LICENSE.txt').read_bytes() == (HERE/'acquisition-v1/LICENSE.source').read_bytes()
    return dict(status='PASS', scope='Read-only frozen-source/result integrity and scalar count reconciliation',
                qualification_attempts=2, completed_tests=26, initial_harness_errors=1,
                final_failures=0, final_errors=0, independent_agent_review=False,
                source_acquisition_requests=10, scientific_model_forwards=0, candidate_pixels=0)


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, sort_keys=True))
