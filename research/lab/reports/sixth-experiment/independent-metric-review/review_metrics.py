"""Independent raw-array review; no benchmark metric imports or model execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import numpy as np

ROLES = ['russo_ex01675k','russo_control75k','seed42','seed43']


def require(value,message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def file_records(value,records):
    if isinstance(value,dict):
        if {'path','size_bytes','sha256'} <= value.keys():
            path = str(Path(value['path']).resolve())
            item = dict(path=path,size_bytes=value['size_bytes'],sha256=value['sha256'])
            require(path not in records or records[path] == item,'Conflicting file records')
            records[path] = item
        for item in value.values():
            file_records(item,records)
    elif isinstance(value,list):
        for item in value:
            file_records(item,records)


def verify_records(records):
    for path,item in records.items():
        require(Path(path).stat().st_size == item['size_bytes'] and sha(path) == item['sha256'],
                'Hash/size changed: '+Path(path).name)


def snapshot(path,records):
    path = Path(path).resolve()
    item = dict(path=str(path),size_bytes=path.stat().st_size,sha256=sha(path))
    require(str(path) not in records or records[str(path)] == item,'Receipt differs from current file')
    records[str(path)] = item


def ranked_metrics(labels,scores):
    """AUC via concordant pairs by ascending groups; AP via ascending suffix counts."""
    y = np.asarray(labels,dtype=bool).reshape(-1)
    s = np.asarray(scores).reshape(-1)
    require(y.shape == s.shape and np.isfinite(s).all(),'Invalid ranking inputs')
    n = int(y.size)
    p = int(y.sum())
    q = n-p
    levels,index = np.unique(s,return_inverse=True)
    group_total = np.bincount(index,minlength=levels.size).astype(np.int64)
    group_pos = np.bincount(index,weights=y.astype(np.int64),minlength=levels.size).astype(np.int64)
    group_neg = group_total-group_pos
    # Integer twice-concordance counts assign exactly half credit to each tied pos/neg pair.
    lower_neg = 0
    twice_concordance = 0
    for positive,negative in zip(group_pos,group_neg):
        twice_concordance += int(positive)*(2*lower_neg+int(negative))
        lower_neg += int(negative)
    auc = twice_concordance/(2*p*q) if p and q else None
    suffix_pos = np.cumsum(group_pos[::-1])[::-1]
    suffix_total = np.cumsum(group_total[::-1])[::-1]
    ap = math.fsum(int(a)/p * int(b)/int(c) for a,b,c in zip(group_pos,suffix_pos,suffix_total)
                  if a) if p else None
    predicted = s >= .5
    tp = int(np.count_nonzero(y & predicted))
    fp = int(np.count_nonzero(~y & predicted))
    fn = p-tp
    tn = q-fp
    balanced = .5*(tp/p+tn/q) if p and q else None
    return dict(pixel_count=n,supplied_positive_pixels=p,supplied_negative_pixels=q,
        roc_auc=auc,average_precision=ap,balanced_accuracy=balanced,
        sensitivity=tp/p if p else None,specificity=tn/q if q else None,
        precision=tp/(tp+fp) if tp+fp else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        supplied_positive_fraction=p/n if n else None,
        unique_score_count=int(levels.size),tied_score_group_count=int(np.count_nonzero(group_total>1)),
        tied_score_pixel_count=int(group_total[group_total>1].sum()),
        twice_concordant_positive_negative_pairs=twice_concordance,
        fixed_threshold=dict(operator='>=',threshold=.5,tp=tp,fp=fp,fn=fn,tn=tn))


def fixtures():
    cases = [([1,0,1,0],[.8,.8,.4,.1],.625,7/12),
             ([1,1,0,0],[.9,.8,.7,.6],1.,1.),
             ([0,0,1,1],[.9,.8,.7,.6],0.,5/12),
             ([1,0,1,0,0],[.5]*5,.5,.4)]
    for labels,scores,auc,ap in cases:
        result = ranked_metrics(labels,scores)
        require(abs(result['roc_auc']-auc)<1e-14 and abs(result['average_precision']-ap)<1e-14,
                'Independent synthetic rank/tie fixture failed')
    # Direct pair enumeration cross-check uses neither group formula nor benchmark code.
    generator = np.random.default_rng(20261008)
    for _ in range(30):
        y = generator.integers(0,2,size=23).astype(bool)
        s = generator.integers(0,5,size=23)
        if not y.any() or y.all():
            continue
        expected = math.fsum(float(a>b)+.5*float(a==b) for a in s[y] for b in s[~y])/(int(y.sum())*int((~y).sum()))
        require(abs(ranked_metrics(y,s)['roc_auc']-expected)<1e-14,'Pair enumeration AUC fixture failed')
    return dict(fixed_auc_ap_cases=4,random_tied_pair_enumeration_cases=30,seed=20261008)


def review(plan_path,plan_sha,run_dir,output):
    require(sha(plan_path) == plan_sha,'Unexpected frozen plan SHA')
    plan = json.loads(plan_path.read_text())
    require(plan['ready_for_execution'] is True and plan['model_order'] == ROLES,'Final frozen four-arm plan required')
    require((run_dir/'frozen-plan.json').read_bytes() == plan_path.read_bytes(),'Run used another plan')
    execution = json.loads((run_dir/'execution.json').read_text())
    require(execution['status'] == 'complete' and all(r['exit_code'] == 0 for r in execution['runs']),
            'Original run/analysis did not complete')
    report = json.loads((run_dir/'analysis/metrics.json').read_text())
    support = json.loads((run_dir/'analysis/common-support.json').read_text())
    records = {}
    for item in (plan,execution,report,support):
        file_records(item,records)
    for path in [plan_path,run_dir/'frozen-plan.json',run_dir/'execution.json',
                 run_dir/'analysis/metrics.json',run_dir/'analysis/common-support.json',
                 run_dir/'analysis/final-integrity.json',run_dir/'preflight.json']:
        snapshot(path,records)
    arrays = {name:np.load(record['path'],allow_pickle=False) for name,record in plan['inputs'].items()}
    surface = arrays['surface']
    require(surface.dtype == np.uint8 and list(surface.shape) == plan['data']['shape'],'Surface shape/dtype')
    shape = surface.shape[1:]
    binary = {}
    for name in ('inklabels','supervision_mask','validation_mask'):
        raw = arrays[name]
        foreground = plan['data']['binary_foreground'][name]
        require(raw.dtype == np.uint8 and raw.shape == shape and set(np.unique(raw)) <= {0,foreground},
                'Binary annotation guard')
        binary[name] = raw == foreground
    labels = binary['inklabels']
    observed = arrays['acquisition_observed_mask']
    occupancy = arrays['source_observed_nonzero_mask']
    require(observed.dtype == bool and observed.shape == shape and observed.all(),'Observed source guard')
    require(occupancy.dtype == bool and np.array_equal(occupancy,np.any(surface[2:19] != 0,axis=0)),
            'Occupancy contract')
    common = np.ones(shape,dtype=bool)
    raw_scores = {}
    for role in ROLES:
        metadata = json.loads((run_dir/role/'metadata.json').read_text())
        file_records(metadata,records)
        snapshot(run_dir/role/'metadata.json',records)
        require(metadata['plan']['sha256'] == plan_sha and metadata['inputs'] == plan['inputs'] and
                metadata['execution_code'] == plan['code'],'Output model/source/code provenance mismatch')
        artifact = next(x for x in plan['models'] if x['label'] == role)
        require(metadata['model']['sha256'] == artifact['sha256'] and metadata['model']['step'] == 74999 and
                metadata['model']['strict_state_match'],'Output checkpoint contract mismatch')
        score = np.load(metadata['raw_scores']['path'],allow_pickle=False)
        evaluated = np.load(metadata['evaluated_mask']['path'],allow_pickle=False)
        require(score.dtype == np.float32 and score.shape == shape and evaluated.dtype == bool and evaluated.shape == shape,
                'Raw result array contract')
        require(np.all((score[np.isfinite(score)]>=0)&(score[np.isfinite(score)]<=1)),'Raw score range')
        raw_scores[role] = score
        common &= evaluated & np.isfinite(score)
    verify_records(records)
    margin = np.zeros(shape,dtype=bool)
    margin[64:-64,64:-64] = True
    before = dict(primary=binary['validation_mask'] & observed & margin,
                  secondary=binary['validation_mask'] & observed)
    domains = {name:mask & common for name,mask in before.items()}
    checks = {}
    metrics = {}
    for name,mask in domains.items():
        snapshot(run_dir/(name+'-before-predictions-mask.npy'),records)
        snapshot(run_dir/'analysis'/(name+'-common-mask.npy'),records)
        require(np.array_equal(before[name],np.load(run_dir/(name+'-before-predictions-mask.npy'),allow_pickle=False)),
                'Pre-inference domain mismatch')
        require(np.array_equal(mask,np.load(run_dir/'analysis'/(name+'-common-mask.npy'),allow_pickle=False)),
                'Final common mask differs from independent formula')
        expected_counts = dict(pixel_count=int(mask.sum()),supplied_positive_pixels=int((mask & labels).sum()),
                               supplied_negative_pixels=int((mask & ~labels).sum()))
        require(expected_counts == support['final_domains'][name],'Domain class counts differ')
        metrics[name] = {}
        for role in ROLES:
            independent = ranked_metrics(labels[mask],raw_scores[role][mask])
            expected = report['metrics'][name]['models'][role]
            require(all(independent[k] == expected[k] for k in
                        ('pixel_count','supplied_positive_pixels','supplied_negative_pixels')),
                    'Per-model class counts differ')
            errors = {}
            for field in ('roc_auc','average_precision','balanced_accuracy','sensitivity','specificity','precision','f1',
                          'supplied_positive_fraction'):
                actual,value = independent[field],expected[field]
                require((actual is None)==(value is None),'Undefined metric handling differs')
                errors[field] = abs(actual-value) if actual is not None else None
                require(actual is None or errors[field] <= 1e-12,'Independent metric differs: '+name+'/'+role+'/'+field)
            require({k:independent['fixed_threshold'][k] for k in ('tp','fp','tn','fn')} == expected['confusion'],
                    'Fixed >=0.5 confusion counts differ')
            require(independent['unique_score_count'] == expected['distinct_score_groups'] and
                    independent['tied_score_pixel_count'] == expected['tied_score_pixel_count'],'Raw tie counts differ')
            independent['absolute_differences_from_report'] = errors
            independent['occupancy_strata'] = {}
            for stratum,submask in [('nonzero_profiles',mask & occupancy),('observed_zero_profiles',mask & ~occupancy)]:
                value = ranked_metrics(labels[submask],raw_scores[role][submask])
                expected_stratum = expected['occupancy_strata'][stratum]
                require(all(value[k] == expected_stratum[k] for k in
                            ('pixel_count','supplied_positive_pixels','supplied_negative_pixels')),
                        'Occupancy stratum class counts differ')
                require({k:value['fixed_threshold'][k] for k in ('tp','fp','tn','fn')} == expected_stratum['confusion'],
                        'Occupancy stratum confusion counts differ')
                for field in ('roc_auc','average_precision','balanced_accuracy'):
                    require(value[field] == expected_stratum[field] if value[field] is None else
                            abs(value[field]-expected_stratum[field]) <= 1e-12,'Occupancy stratum metric differs')
                independent['occupancy_strata'][stratum] = value
            metrics[name][role] = independent
        paired = {}
        for field in ('roc_auc','average_precision','balanced_accuracy'):
            first,second = metrics[name]['russo_ex01675k'][field],metrics[name]['russo_control75k'][field]
            delta = first-second if first is not None and second is not None else None
            expected_delta = report['metrics'][name]['ex016_vs_matched_control']['ex016_minus_control_'+field]
            require((delta is None and expected_delta is None) or
                    (delta is not None and expected_delta is not None and abs(delta-expected_delta)<=1e-12),
                    'Matched paired difference differs')
            paired['ex016_minus_control_'+field] = delta
        a = raw_scores['russo_ex01675k'][mask].astype(np.float64)
        b = raw_scores['russo_control75k'][mask].astype(np.float64)
        ac = (a>=.5) == labels[mask]
        bc = (b>=.5) == labels[mask]
        pair_counts = dict(both_correct=int((ac & bc).sum()),both_wrong=int((~ac & ~bc).sum()),
            first_correct_second_wrong=int((ac & ~bc).sum()),second_correct_first_wrong=int((~ac & bc).sum()),
            fixed_midpoint_prediction_disagreements=int(((a>=.5)!=(b>=.5)).sum()))
        expected_pair = report['metrics'][name]['ex016_vs_matched_control']
        for key,value in pair_counts.items():
            require(value == expected_pair[key],'Paired threshold/correctness count differs')
        differences = a-b
        centered_a,centered_b = a-a.mean(),b-b.mean()
        denominator = math.sqrt(float(np.dot(centered_a,centered_a))*float(np.dot(centered_b,centered_b)))
        correlation = float(np.dot(centered_a,centered_b))/denominator if denominator else None
        diagnostics = dict(first_minus_second_score_mean=float(differences.mean()),
            first_minus_second_score_median=float(np.median(differences)),
            mean_absolute_score_difference=float(np.abs(differences).mean()),pearson_r=correlation)
        for key,value in diagnostics.items():
            require(value is None if expected_pair[key] is None else abs(value-expected_pair[key])<=1e-12,
                    'Paired raw score diagnostic differs')
        seed_a,seed_b = raw_scores['seed42'][mask]>=.5,raw_scores['seed43'][mask]>=.5
        union,intersection = int((seed_a | seed_b).sum()),int((seed_a & seed_b).sum())
        checks[name] = dict(mask_formula_matches=True,counts=expected_counts,paired_differences=paired,
            paired_counts=pair_counts,paired_raw_score_diagnostics=diagnostics,
            seed_fixed_midpoint_stability=dict(intersection=intersection,union=union,jaccard=intersection/union if union else None))
    verify_records(records)
    require(sha(plan_path) == plan_sha,'Frozen plan changed during independent review')
    output.mkdir(exist_ok=False)
    result = dict(status='independently_verified',plan_path=str(plan_path),plan_sha256=plan_sha,
        reviewer_code_sha256=sha(__file__),fixture_checks=fixtures(),verified_input_output_records=list(records.values()),
        independent_metrics=metrics,common_domain_checks=checks,tolerance=1e-12,
        benchmark_metric_code_imported=False,inference_calls=0,network_requests=0,
        interpretation='Descriptive supplied-label agreement in one developmental correlated crop; no ink authenticity or physical/teacher holdout established.')
    (output/'independent-verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    def blocked(*args,**kwargs):
        raise RuntimeError('Independent metric review attempted network access')
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    fixtures()
    result = review(args.plan.resolve(),args.plan_sha256,args.run_dir.resolve(),args.output_dir.resolve())
    print(json.dumps(dict(status=result['status'],verified_records=len(result['verified_input_output_records']),
                         common_domain_checks=result['common_domain_checks'])))


if __name__ == '__main__':
    main()
