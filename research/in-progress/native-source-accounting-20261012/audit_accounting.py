"""Offline source-weight census and conditional loss probe; no model forward."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from source_objective import partial_objective, validate_window


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pair_average(first, second, operation):
    """Enumerate every ordered pair in float64, bounded to 64 rows at a time."""
    total = 0.
    for start in range(0, len(first), 64):
        total += float(operation(first[start:start+64, None], second[None, :]).sum())
    return total / (len(first) * len(second))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    here = Path(__file__).resolve().parent
    original = a.repo/'research/in-progress/native-coverage-completion-20261012'
    prior = a.repo/'research/in-progress/native-added-qualification-20261012'
    retained = a.repo/'research/in-progress/native-coverage-preparation-20261012/matched_training.py'
    case_id = 'pherc0814-20250925161630'
    paths = dict(original_support=original/'ORIGINAL_SUPPORT.csv',
        original_verification=original/'ORIGINAL_VERIFICATION.json',
        added_results=prior/'RESULTS.json', added_verification=prior/'INDEPENDENT_VERIFICATION.json',
        added_support=prior/(case_id+'-support.csv'),
        added_arrays=a.qualification/(case_id+'-patches.npz'), retained_loss=retained,
        audit_implementation=Path(__file__), proposed_objective=here/'source_objective.py', plan=here/'PLAN.md')
    pins = {k: sha(v) for k, v in paths.items()}
    verified = json.loads(paths['original_verification'].read_text())
    assert verified['status'] == 'PASS' and verified['result_csv_sha256'] == pins['original_support']
    added_ver = json.loads(paths['added_verification'].read_text())
    assert added_ver['status'] == 'PASS' and added_ver['qualification_results_sha256'] == pins['added_results']
    results = json.loads(paths['added_results'].read_text())
    case = next(r for r in results['cases'] if r['id'] == case_id)
    assert case['supervision_qualified']
    assert case['outputs'][paths['added_support'].name] == pins['added_support']
    assert case['outputs'][paths['added_arrays'].name] == pins['added_arrays']
    freeze = dict(utc=datetime.now(timezone.utc).isoformat(), pins=pins,
        probe_patch_indices=[0, 1, 2, 3], probe_groups=['fixed', 'fixed', 'changed', 'changed'],
        altered_second_group='First 1024 previously supervised pixels in raster order per patch',
        probe_coefficients=dict(fixed=.5, changed=.5), probe_coefficients_are_study_hyperparameters=False,
        relative_gradient_tolerance=1e-6, model_forwards=0, optimizer_updates=0, paid_compute=False)
    write(a.output/'FREEZE.json', freeze)
    rows = list(csv.DictReader(paths['original_support'].open()))
    assert len(rows) == 876 and all(r['technically_eligible'] == 'True' for r in rows)
    groups = {}
    original_census = []
    for winding in ['w035', 'w039', 'w040', 'w041', 'w044']:
        subset = [r for r in rows if r['winding'] == winding]
        counts = np.array([int(r['supervised_after_ct']) for r in subset], dtype=np.float64)
        groups[winding] = counts
        original_census.append(dict(winding=winding, candidates=len(subset),
            mean_supervised_per_draw=float(counts.mean()), equal_draw_quota=.2,
            equal_draw_dice_coefficient=.2))
    denominator = sum(r['mean_supervised_per_draw'] for r in original_census)
    for row in original_census:
        row['large_balanced_batch_bce_coefficient'] = row['mean_supervised_per_draw']/denominator
    cross_pairs = []
    for left, right in itertools.combinations(groups, 2):
        lm, rm = groups[left], groups[right]
        mean = pair_average(lm, rm, lambda x, y: x/(x+y))
        reverse = pair_average(rm, lm, lambda x, y: x/(x+y))
        assert abs(mean + reverse - 1) < 1e-12
        cross_pairs.append(dict(left=left, right=right, enumerated_pairs=len(lm)*len(rm),
            expected_left_bce_coefficient=mean, right_bce_coefficient=reverse,
            each_dice_coefficient=.5,
            minimum_left_bce_coefficient=float(lm.min()/(lm.min()+rm.max())),
            maximum_left_bce_coefficient=float(lm.max()/(lm.max()+rm.min()))))
    added_rows = list(csv.DictReader(paths['added_support'].open()))
    assert len(added_rows) == 1290 and all(r['technically_eligible'] == 'True' for r in added_rows)
    added_census = []
    for arm in ['restricted_A', 'broad_B']:
        selected = [r for r in added_rows if arm == 'broad_B' or int(r['in_restricted_A'])]
        m = np.array([int(r['supervised']) for r in selected], dtype=np.float64)
        positives = np.array([int(r['positive']) for r in selected], dtype=np.float64)
        expected = next(r for r in case['counts'] if r['arm'] == arm)
        assert int(m.sum()) == expected['supervised_occurrences']
        assert int(positives.sum()) == expected['positive_occurrences']
        # Each term E[p_i/(m_i+m_j)] is enumerated; symmetry gives the second term.
        numerator = 0.
        for start in range(0, len(m), 64):
            numerator += float((positives[start:start+64, None]/(m[start:start+64, None]+m[None, :])).sum())
        finite_two = 2*numerator/(len(m)**2)
        added_census.append(dict(arm=arm, candidates=len(selected),
            mean_supervised_per_draw=float(m.mean()),
            supervised_fraction_minimum=float(m.min()/16384),
            zero_positive_draw_probability=float((positives == 0).mean()),
            raw_positive_fraction_one_patch=float((positives/m).mean()),
            raw_positive_fraction_two_iid_patches=finite_two,
            raw_positive_fraction_large_batch=float(positives.sum()/m.sum()),
            two_patch_ordered_pairs=len(m)**2))
    spec = importlib.util.spec_from_file_location('qualified_matched_training', retained)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with np.load(paths['added_arrays'], allow_pickle=False) as data:
        teacher = data['teacher'][:4].copy()
        valid = data['valid'][:4].copy()
    y = torch.from_numpy((teacher >= 128).astype(np.float32))[:, None]
    masks = torch.from_numpy(valid & ((teacher >= 128) | (teacher <= 64)))[:, None]
    for index in range(4):
        assert int(masks[index].sum()) == int(added_rows[index]['supervised'])
        assert int((y[index]*masks[index]).sum()) == int(added_rows[index]['positive'])
    altered = masks.clone()
    for index in [2, 3]:
        indices = torch.nonzero(masks[index].flatten(), as_tuple=True)[0]
        assert len(indices) > 1024
        altered[index].zero_()
        altered[index].view(-1)[indices[:1024]] = True
    ids = freeze['probe_groups']
    coeff = freeze['probe_coefficients']
    base = torch.linspace(-.7, .8, 4*128*128).reshape(4, 1, 128, 128)
    def measure(mask, mode, *, dice_weight=.25, pieces=None):
        logits = base.clone().requires_grad_(True)
        totals = {name: runtime.stream_totals(y[[i for i, s in enumerate(ids) if s == name]],
            mask[[i for i, s in enumerate(ids) if s == name]]) for name in coeff}
        if mode == 'pooled':
            loss = runtime.partial_stream_loss(logits, y, mask, runtime.stream_totals(y, mask), dice_weight=dice_weight)
            loss.backward()
        elif mode == 'naive_microbatch':
            loss = sum(.25*runtime.partial_stream_loss(logits[i:i+1], y[i:i+1], mask[i:i+1],
                runtime.stream_totals(y[i:i+1], mask[i:i+1]), dice_weight=dice_weight) for i in range(4))
            loss.backward()
        else:
            value = 0.
            for indices in pieces or [[0, 1, 2, 3]]:
                term = partial_objective(logits[indices], y[indices], mask[indices], [ids[i] for i in indices],
                    totals, coeff, runtime.partial_stream_loss, dice_weight=dice_weight)
                value += float(term.detach()); term.backward()
            loss = logits.new_tensor(value)
        return float(loss.detach()), logits.grad.detach(), totals
    _, pa, _ = measure(masks, 'pooled', dice_weight=0.)
    _, pb, _ = measure(altered, 'pooled', dice_weight=0.)
    expected_ratio = float(masks.sum()/altered.sum())
    ratio_error = float((pb[:2]-expected_ratio*pa[:2]).abs().max())
    assert ratio_error < 1e-10 and not torch.equal(pa[:2], pb[:2])
    fa, ga, totals = measure(masks, 'grouped')
    fb, gb, _ = measure(altered, 'grouped')
    assert torch.equal(ga[:2], gb[:2]) and not torch.equal(ga[2:], gb[2:])
    accumulation = []
    for pieces in [[[0], [1], [2], [3]], [[0, 2], [1, 3]]]:
        value, grad, _ = measure(masks, 'grouped', pieces=pieces)
        relative = float(torch.linalg.vector_norm(ga-grad)/torch.linalg.vector_norm(ga))
        assert relative < 1e-6
        accumulation.append(dict(microbatches=pieces, relative_gradient_l2=relative,
            loss_absolute_difference=abs(fa-value)))
    _, naive, _ = measure(masks, 'naive_microbatch')
    naive_relative = float(torch.linalg.vector_norm(ga-naive)/torch.linalg.vector_norm(ga))
    assert naive_relative > 1e-6
    rejected = []
    for name, fn in [
        ('missing_source', lambda: validate_window({'fixed': totals['fixed']}, coeff)),
        ('invalid_coefficient_sum', lambda: validate_window(totals, dict(fixed=.5, changed=.6))),
        ('zero_support', lambda: validate_window(dict(totals, changed=dict(pixels=0, samples=2)), coeff)),
        ('unknown_source', lambda: partial_objective(base, y, masks, ['unknown']*4, totals, coeff, runtime.partial_stream_loss))]:
        try: fn()
        except ValueError: rejected.append(name)
        else: raise AssertionError('Expected rejection: '+name)
    output = dict(status='PASS_OFFLINE_SOURCE_ACCOUNTING', freeze_sha256=sha(a.output/'FREEZE.json'),
        original_equal_source_draws=original_census, original_two_source_enumeration=cross_pairs,
        complete_added_case=dict(id=case_id, arms=added_census),
        conditional_probe=dict(patch_ids=[r['id'] for r in added_rows[:4]],
            groups_are_arithmetic_probes_not_physical_sources=True,
            supervised_before=[int(v) for v in masks.flatten(1).sum(1)],
            supervised_after=[int(v) for v in altered.flatten(1).sum(1)],
            fixed_group_pooled_bce_gradient_ratio=expected_ratio, ratio_max_absolute_error=ratio_error,
            fixed_group_source_normalized_bce_plus_dice_gradients_bitwise_equal=True,
            changed_group_gradients_differ=True, source_normalized_losses=[fa, fb],
            accumulation=accumulation, naive_microbatch_relative_gradient_l2=naive_relative,
            rejected_invalid_contracts=rejected),
        interpretation='Source loss coefficients are not source draw quotas. Proposal only; the actual eight-source training objective and CUDA loader remain unqualified.',
        limitations=['No full-population source/class effect estimate from one added segment',
            'BCE coefficient and class-mass bookkeeping is not model gradient magnitude or ink accuracy',
            'Class fractions precede BCE target smoothing; Dice is nonlinear in predictions',
            'Arithmetic census is unaugmented; final denominators must follow actual augmentation',
            'Source coefficients do not fix within-source class, support, annotation quality or physical overlap',
            'Complete accumulation consumption and actual loader require later end-to-end qualification'],
        numpy_version=np.__version__, torch_version=torch.__version__,
        model_forwards=0, optimizer_updates=0, full_cohort_admitted=False, paid_compute=False)
    write(a.output/'RESULTS.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
