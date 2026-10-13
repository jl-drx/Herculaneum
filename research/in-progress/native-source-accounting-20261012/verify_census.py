"""Independent scalar census recomputation; standard library only, no tensors.

Does not verify image provenance, loss autograd, CUDA, or physical source identity.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0., abs_tol=1e-12), (actual, expected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = json.loads(a.result.read_text())
    freeze = json.loads(a.result.with_name('FREEZE.json').read_text())
    assert result['freeze_sha256'] == sha(a.result.with_name('FREEZE.json'))
    original = a.repo/'research/in-progress/native-coverage-completion-20261012/ORIGINAL_SUPPORT.csv'
    added = a.repo/'research/in-progress/native-added-qualification-20261012/pherc0814-20250925161630-support.csv'
    assert sha(original) == freeze['pins']['original_support']
    assert sha(added) == freeze['pins']['added_support']
    rows = list(csv.DictReader(original.open()))
    groups = {key: [int(r['supervised_after_ct']) for r in rows if r['winding'] == key]
        for key in sorted(set(r['winding'] for r in rows))}
    means = {key: math.fsum(values)/len(values) for key, values in groups.items()}
    for record in result['original_equal_source_draws']:
        key = record['winding']
        assert len(groups[key]) == record['candidates']
        close(means[key]/math.fsum(means.values()), record['large_balanced_batch_bce_coefficient'])
    original_pairs = 0
    for left, right in itertools.combinations(groups, 2):
        lm, rm = groups[left], groups[right]
        n = len(lm)*len(rm)
        direct = math.fsum(x/(x+y) for x in lm for y in rm)/n
        record = next(r for r in result['original_two_source_enumeration'] if r['left'] == left and r['right'] == right)
        assert n == record['enumerated_pairs']
        close(direct, record['expected_left_bce_coefficient'])
        original_pairs += n
    rows = list(csv.DictReader(added.open()))
    added_pairs = 0
    for record in result['complete_added_case']['arms']:
        selected = [r for r in rows if record['arm'] == 'broad_B' or int(r['in_restricted_A'])]
        values = [(int(r['positive']), int(r['supervised'])) for r in selected]
        n = len(values)
        direct = math.fsum((p+q)/(m+k) for p, m in values for q, k in values)/(n*n)
        close(direct, record['raw_positive_fraction_two_iid_patches'])
        close(math.fsum(p/m for p, m in values)/n, record['raw_positive_fraction_one_patch'])
        close(sum(p for p, m in values)/sum(m for p, m in values), record['raw_positive_fraction_large_batch'])
        close(sum(p == 0 for p, m in values)/n, record['zero_positive_draw_probability'])
        added_pairs += n*n
    receipt = dict(status='PASS_INDEPENDENT_SCALAR_CENSUS', results_sha256=sha(a.result),
        verifier_sha256=sha(Path(__file__)), original_cross_source_pairs=original_pairs,
        added_ordered_pairs=added_pairs, tolerance=1e-12,
        scope='Separate scalar arithmetic over existing verified tables; no autograd or image-provenance claim',
        model_forwards=0, paid_compute=False)
    with a.output.open('x') as f:
        f.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
