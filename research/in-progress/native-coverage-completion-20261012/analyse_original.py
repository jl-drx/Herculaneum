"""Account for native annotation exposure and UV overlap; no training or scoring."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--support', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    verified = json.loads(a.verification.read_text())
    assert verified['status'] == 'PASS' and verified['complete_ct_rows'] == 876
    assert verified['result_csv_sha256'] == sha(a.support)
    rows = list(csv.DictReader(a.support.open()))
    assert len(rows) == 876 and all(r['status'] == 'CT_CHECKED' and r['technically_eligible'] == 'True' for r in rows)
    request_plan = json.loads((a.original / 'REQUEST_PLAN.json').read_text())
    summaries = []
    for winding in ['w035', 'w039', 'w040', 'w041', 'w044']:
        annotation = a.original / 'decoded' / (winding + '.npz')
        with np.load(annotation, allow_pickle=False) as f:
            mask = f['supervision']
            target = f['target']
        assert mask.dtype == target.dtype == np.bool_ and not np.any(target & ~mask)
        group = [r for r in rows if r['winding'] == winding]
        multiplicity = np.zeros(mask.shape, np.uint8)
        for r in group:
            x, y = int(r['x']), int(r['y'])
            assert 0 <= x <= mask.shape[1]-128 and 0 <= y <= mask.shape[0]-128
            multiplicity[y:y+128, x:x+128] += 1
        assert multiplicity.max() <= 4  # 128-square patches on the frozen stride-96 grid.
        known = np.zeros_like(mask)
        centre = np.zeros_like(mask)
        for r in request_plan['requests']:
            if r['source_id'] != winding:
                continue
            receipt = json.loads((a.original / 'receipts' / (r['id'] + '.json')).read_text())
            source = a.original / r['path']
            assert receipt['status'] == 'PRESENT' and receipt['request'] == r
            assert source.stat().st_size == 28*128*128 and sha(source) == receipt['sha256']
            raw = np.fromfile(source, np.uint8).reshape(28, 128, 128)
            y, x = r['cy']*128, r['cx']*128
            h, w = min(128, mask.shape[0]-y), min(128, mask.shape[1]-x)
            known[y:y+h, x:x+w] = True
            centre[y:y+h, x:x+w] = raw[14, :h, :w] > 0
        selected = multiplicity > 0
        assert np.all(known[selected])
        supported = selected & mask & centre
        positive = supported & target
        support_occurrences = sum(int(r['supervised_after_ct']) for r in group)
        positive_occurrences = sum(int(r['positive_after_ct']) for r in group)
        assert int(multiplicity[supported].sum()) == support_occurrences
        assert int(multiplicity[positive].sum()) == positive_occurrences
        support_values = np.array([int(r['supervised_after_ct']) for r in group])
        positive_values = np.array([int(r['positive_after_ct']) for r in group])
        summaries.append(dict(winding=winding, candidates=len(group),
            source_supervised_uv_pixels=int(mask.sum()),
            selected_core_uv_pixels=int(selected.sum()),
            selected_supervised_uv_pixels_before_ct=int((selected & mask).sum()),
            unique_supported_uv_pixels=int(supported.sum()),
            unique_positive_uv_pixels=int(positive.sum()),
            supervised_source_uv_pixels_outside_selected_cores=int((mask & ~selected).sum()),
            selected_supervised_uv_pixels_rejected_by_ct=int((selected & mask & ~centre).sum()),
            supervised_occurrences=support_occurrences, positive_occurrences=positive_occurrences,
            negative_occurrences=support_occurrences-positive_occurrences,
            supervised_multiplicity={str(i): int((multiplicity[supported] == i).sum()) for i in range(1,5)},
            mean_supervised_reuse=support_occurrences/int(supported.sum()),
            zero_positive_patches=int((positive_values == 0).sum()),
            zero_negative_patches=int((positive_values == support_values).sum()),
            supervised_fraction_quantiles={str(q):float(np.quantile(support_values/16384,q)) for q in [0,.25,.5,.75,1]},
            positive_fraction_of_supported_occurrences=positive_occurrences/support_occurrences,
            annotation_sha256=sha(annotation)))
    total_candidates = len(rows)
    sampler_examples = []
    for name in ['uniform_candidate', 'equal_winding_then_uniform_candidate']:
        quotas = {s['winding']:(s['candidates']/total_candidates if name == 'uniform_candidate' else .2) for s in summaries}
        expected_sup = sum(quotas[s['winding']]*s['supervised_occurrences']/s['candidates'] for s in summaries)
        expected_pos = sum(quotas[s['winding']]*s['positive_occurrences']/s['candidates'] for s in summaries)
        sampler_examples.append(dict(policy=name, winding_draw_probability=quotas,
            expected_supervised_occurrences_per_unaugmented_draw=expected_sup,
            expected_positive_occurrences_per_unaugmented_draw=expected_pos,
            positive_fraction_of_expected_occurrences=expected_pos/expected_sup,
            zero_positive_patch_probability=sum(quotas[s['winding']]*s['zero_positive_patches']/s['candidates'] for s in summaries)))
    totals = {name:sum(s[name] for s in summaries) for name in ['candidates', 'source_supervised_uv_pixels',
        'selected_core_uv_pixels', 'selected_supervised_uv_pixels_before_ct', 'unique_supported_uv_pixels',
        'unique_positive_uv_pixels', 'supervised_source_uv_pixels_outside_selected_cores',
        'selected_supervised_uv_pixels_rejected_by_ct', 'supervised_occurrences', 'positive_occurrences',
        'negative_occurrences', 'zero_positive_patches', 'zero_negative_patches']}
    assert totals['supervised_occurrences'] == verified['supervised_pixels_with_patch_overlap']
    assert totals['positive_occurrences'] == verified['positive_pixels_with_patch_overlap']
    result = dict(status='COMPLETE_ORIGINAL_STREAM_ACCOUNTING', cases=summaries, totals=totals,
        sampler_examples=sampler_examples, training_sampler_selected=False,
        sampler_scope='Unaugmented source exposure arithmetic; ratios of expected counts are not expected finite-batch BCE gradients.',
        uv_scope='Deduplicated only within each stored winding UV grid. Sums across windings do not establish unique physical tissue or independent subjects.',
        label_scope='Values under declared native supervision masks, not authenticated ink or verified physical blank controls.',
        role='Fixed original training stream only. Completed native fits/evaluation are not reopened.',
        pins=dict(support_sha256=sha(a.support), verification_sha256=sha(a.verification),
            request_plan_sha256=sha(a.original/'REQUEST_PLAN.json'), implementation_sha256=sha(Path(__file__))),
        full_coverage_pair_admitted=False, new_model_forwards=0, paid_compute=False)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], totals=totals, sampler_examples=sampler_examples)), flush=True)


if __name__ == '__main__':
    main()
