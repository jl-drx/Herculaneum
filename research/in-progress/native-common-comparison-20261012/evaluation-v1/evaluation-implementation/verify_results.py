"""Separate read-only file, map and metric recomputation; never runs a model.

This is a second implementation in this workspace, not an external human review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


SCALARS = ['ap', 'auc', 'prevalence', 'brier', 'bce', 'ece', 'mean_positive', 'mean_negative', 'negative_ge_half']
MODELS = ['reader_v2', 'dnative', 'seed42', 'seed43', 'hecate']


def load(p):
    return json.loads(p.read_text())


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verified_file(root, row):
    p = root/row['path']
    assert p.resolve().is_relative_to(root.resolve()) and p.is_file() and not p.is_symlink()
    assert p.stat().st_size == row['bytes'] and digest(p) == row['sha256'], str(p)
    return p


def independent_metrics(y, p):
    y = np.asarray(y, bool).reshape(-1); p = np.asarray(p, np.float64).reshape(-1)
    assert y.size == p.size and y.size and np.isfinite(p).all() and np.logical_and(p >= 0, p <= 1).all()
    count = y.size; pos = int(np.count_nonzero(y)); neg = count-pos
    # Ascending score groups, pairwise positive-vs-negative wins for ROC AUC.
    values, group, sizes = np.unique(p, return_inverse=True, return_counts=True)
    gp = np.bincount(group, weights=y.astype(int), minlength=values.size)
    gn = sizes-gp
    if pos and neg:
        neg_below = np.cumsum(gn)-gn
        auc = float(np.dot(gp, neg_below+.5*gn)/(pos*neg))
        positives_at_or_above = np.cumsum(gp[::-1])[::-1]
        selected_at_or_above = np.cumsum(sizes[::-1])[::-1]
        ap = float(np.dot(gp/pos, positives_at_or_above/selected_at_or_above))
    else:
        ap = auc = None
    # Class-separated sums avoid using the producer's elementwise BCE expression.
    q = np.maximum(1e-6, np.minimum(1-1e-6, p))
    bce = float((-np.log(q[y]).sum()-np.log(1-q[~y]).sum())/count)
    brier = float((np.square(1-p[y]).sum()+np.square(p[~y]).sum())/count)
    ids = np.clip(np.floor(p*10).astype(np.int64), 0, 9)
    n = np.bincount(ids, minlength=10)
    score_sum = np.bincount(ids, weights=p, minlength=10)
    positive_sum = np.bincount(ids, weights=y.astype(int), minlength=10)
    calibration = [dict(bin=i, lower=i/10, upper=(i+1)/10, count=int(n[i]),
        mean_score=float(score_sum[i]/n[i]) if n[i] else None,
        positive_fraction=float(positive_sum[i]/n[i]) if n[i] else None) for i in range(10)]
    return dict(pixels=count, positives=pos, negatives=neg, ap=ap, auc=auc,
        prevalence=pos/count, bce=bce, brier=brier, ece=float(np.abs(score_sum-positive_sum).sum()/count),
        mean_positive=float(p[y].sum()/pos) if pos else None,
        mean_negative=float(p[~y].sum()/neg) if neg else None,
        negative_ge_half=float(np.count_nonzero(p[~y] >= .5)/neg) if neg else None, calibration=calibration)


def same(expected, actual, where='root'):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), where
        for k in expected:
            same(expected[k], actual[k], where+'/'+k)
    elif isinstance(expected, list):
        assert len(expected) == len(actual), where
        for i, (l, rr) in enumerate(zip(expected, actual)):
            same(l, rr, where+'/'+str(i))
    elif isinstance(expected, (float, np.floating)):
        assert actual is not None and abs(expected-actual) <= 2e-12, (where, expected, actual)
    else:
        assert expected == actual, (where, expected, actual)


def verify(inputs, run):
    go = load(run/'GO.json'); go_sha = digest(run/'GO.json')
    done = load(run/'INFERENCE_COMPLETE.json')
    assert done['status'] == 'ALL_MAPS_COMPLETE' and done['go_sha256'] == go_sha
    assert (done['completed_jobs'], done['core_maps'], done['unique_model_forwards']) == (20,120,1698)
    for row in go['input_pins']:
        verified_file(inputs, row)
    for row in done['receipts']:
        verified_file(run, row)
    selection = load(inputs/'SELECTION.json'); patches = load(inputs/'PATCHES_COMPLETE.json')
    by_case = {x['id']:x for x in patches['cases']}
    records = []; jobs = tiles = output_files = 0
    for case in selection['cases']:
        maps = {}
        for model in MODELS:
            folder = run/'jobs'/(model+'--'+case['id']); receipt = load(folder/'COMPLETE.json')
            model_row = next(x for x in go['models'] if x['name'] == model)
            assert receipt['binding'] == dict(go_sha256=go_sha, case=case['id'], model=model,
                                             checkpoint_sha256=model_row['checkpoint']['sha256'])
            family = 'hecate' if model == 'hecate' else 'hybrid'
            rows = sorted([x for x in by_case[case['id']]['patches'] if x['family'] == family and x['raw_nonempty']],
                          key=lambda x:x['origin_yx'])
            assert receipt['counts']['model_calls'] == len(rows)
            assert receipt['read_only_replay'] == dict(model_calls=len(rows),new_forwards=0,reused_forwards=len(rows))
            files = sorted(folder.glob('tile-*.json')); assert len(files) == len(rows)
            for index, (file, source) in enumerate(zip(files, rows)):
                tile = load(file)
                assert tile['binding'] == receipt['binding'] and tile['index'] == index
                assert tile['origin_yx'] == source['origin_yx'] and tile['source_sha256'] == source['sha256']
                path = verified_file(folder, tile['logits']); arr = np.load(path, allow_pickle=False)
                assert list(arr.shape) == tile['decoded_logits']['shape'] and str(arr.dtype) == tile['decoded_logits']['dtype']
                assert hashlib.sha256(arr.tobytes()).hexdigest() == tile['decoded_logits']['sha256']
                assert arr.ndim == 4 and arr.shape[:2] == (1,1) and arr.dtype == np.float32 and np.isfinite(arr).all()
                assert len(tile['inputs']) == (2 if family == 'hecate' else 1)
                assert tile['inputs'][0]['shape'] == ([1,1,16,64,64] if family == 'hecate' else [1,1,17,128,128])
                assert tile['inputs'][0]['dtype'] == 'float32'; tiles += 1
            outputs = {x['path']:verified_file(folder, x) for x in receipt['outputs']}; output_files += len(outputs)
            native = None
            if model == 'hecate':
                with np.load(outputs['_model_grid.npz']) as z:
                    grid = z['raw_grid']; assert grid.dtype == np.uint8
                    assert list(grid.shape) == by_case[case['id']]['hecate_grid_yx']
                    h,w = case['native']['shape_zyx'][1:]
                    native = cv2.resize(grid.astype(np.float32)/255, (w,h), interpolation=cv2.INTER_LINEAR)
            for core in case['cores']:
                with np.load(outputs[core['id']+'.npz']) as z:
                    aligned = z['aligned'].copy()
                    assert aligned.shape == (128,128) and aligned.dtype == np.float32
                    assert np.isfinite(aligned).all() and (aligned >= 0).all() and (aligned <= 1).all()
                    if model != 'hecate':
                        assert (z['denominator'] > 1e-6).all()
                        prob = np.clip(z['numerator']/z['denominator'],0,1)
                        assert np.array_equal(prob,z['unquantized'])
                        raw = (prob*255).astype(np.uint8)
                        assert np.array_equal(raw,z['raw_grid']) and np.array_equal(raw,z['display'])
                        assert np.array_equal(raw.astype(np.float32)/255,aligned)
                    else:
                        x,y,cw,ch = core['core_xywh']
                        assert np.array_equal(aligned,native[y:y+ch,x:x+cw])
                        assert np.array_equal(np.rint(aligned*255).astype(np.uint8),z['display'])
                maps[(model,core['id'])] = aligned
            jobs += 1
        for core in case['cores']:
            with np.load(inputs/'cores'/(core['id']+'.npz')) as z:
                mask=z['score_mask']; ink=z['ink']; assert mask.dtype == ink.dtype == bool
            scores = {m:maps[(m,core['id'])][mask] for m in MODELS}
            scores['mean5'] = (np.stack([maps[(m,core['id'])] for m in MODELS]).sum(axis=0,dtype=np.float32)/np.float32(5))[mask]
            scores['constant_0.5'] = np.full(int(mask.sum()),.5,np.float32)
            records.append(dict(id=core['id'], case=case['id'], parent=case['sample'], y=ink[mask], p=scores))
    assert (jobs,tiles,output_files) == (20,1698,124)
    result = load(run/'evaluation/METRICS.json'); assert result['go_sha256'] == go_sha
    assert len(result['summary']['core']) == 168 and len(result['summary']['segment']) == 28
    metrics_count = 0
    for level in ['core','segment','parent_pooled','panel_pooled']:
        for row in result['summary'][level]:
            chosen = [x for x in records if level == 'panel_pooled' or
                      x[{'core':'id','segment':'case','parent_pooled':'parent'}[level]] == row['id']]
            y = np.concatenate([x['y'] for x in chosen]); score = np.concatenate([x['p'][row['model']] for x in chosen])
            same(independent_metrics(y,score), row['metrics'], level+'/'+row['id']+'/'+row['model']); metrics_count += 1
    for row in result['summary']['parent_macro']:
        chosen = [x['metrics'] for x in result['summary']['segment'] if x['parent'] == row['id'] and x['model'] == row['model']]
        value = {k:sum(x[k] for x in chosen)/len(chosen) for k in SCALARS}
        same(value,row['metrics'],'parent_macro'); metrics_count += 1
    for row in result['summary']['equal_parent_macro']:
        chosen = [x['metrics'] for x in result['summary']['parent_macro'] if x['model'] == row['model']]
        same({k:sum(x[k] for x in chosen)/2 for k in SCALARS},row['metrics'],'equal_parent_macro'); metrics_count += 1
    gap = result['comparisons']['released_checkpoint_variation']
    parent = {(x['id'],x['model']):x['metrics']['ap'] for x in result['summary']['parent_macro']}
    macro = {x['model']:x['metrics']['ap'] for x in result['summary']['equal_parent_macro']}
    same(abs(macro['seed42']-macro['seed43']),gap['equal_parent_ap'])
    for p,d in gap['parent_ap'].items():
        same(abs(parent[(p,'seed42')]-parent[(p,'seed43')]),d)
    for pair in result['comparisons']['pairs']:
        l,rr = pair['left'],pair['right']; delta = {p:parent[(p,l)]-parent[(p,rr)] for p in gap['parent_ap']}
        same(delta,pair['parent_ap_delta']); same(macro[l]-macro[rr],pair['equal_parent_ap_delta'])
        assert pair['opposite_parent_order'] == (any(d < 0 for d in delta.values()) and any(d > 0 for d in delta.values()))
        assert pair['macro_difference_smaller_than_released_seed_gap'] == (abs(macro[l]-macro[rr]) < gap['equal_parent_ap'])
        assert pair['parents_difference_smaller_than_seed_gap'] == [p for p,d in delta.items() if abs(d) < gap['parent_ap'][p]]
    assert len(records) == 24 and sum(len(x['y']) for x in records) == 232250
    assert sum(int(x['y'].sum()) for x in records) == 20137
    assert sum(not x['y'].any() or x['y'].all() for x in records) == 14
    return dict(status='PASS',go_sha256=go_sha, jobs=jobs, committed_logits=tiles, output_files=output_files,
        core_maps=120, scored_pixels=232250, positives=20137, single_class_cores_retained=14,
        recomputed_metric_records=metrics_count, derived_output_checks='PASS', comparison_flags='PASS',
        trained_model_forwards=0, scope='Separate implementation, not external independent review; original runner performed all exact logit replays')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a = p.parse_args(); assert not a.output.exists()
    result = verify(a.inputs,a.run)
    result['implementation_sha256'] = digest(Path(__file__))
    a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
