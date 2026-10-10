"""Descriptive paired fresh-region endpoints and the fixed three-comparison screen."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiment_common import ARMS, load_plan, record, verify, write_json, module, require, evaluation


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True); parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); plan,path=load_plan(args.plan,execute=True); before=record(path)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    metric=module(plan['metrics'],'eighth_metrics'); cutoff=module(plan['cutoff'],'eighth_operating')
    workers={a:json.loads((args.run_dir/a/'result.json').read_text()) for a in ARMS}
    worker_records={a:record(args.run_dir/a/'result.json') for a in ARMS}
    require(all(w['status']=='PASS' and w['plan']==before for w in workers.values()),'Incomplete/mismatched worker')
    require(len({w['initial_state_sha256'] for w in workers.values()})==1,'Initializer equality failed')
    locks={a:json.loads(verify(w['threshold_lock']).read_text()) for a,w in workers.items()}
    results={}; records=[]
    for case in plan['fresh']:
        data,labels,primary=evaluation(case); maps={}; common=primary.copy()
        for arm,w in workers.items():
            item=w['fresh'][case['id']]; s=np.load(verify(item['scores']),allow_pickle=False); e=np.load(verify(item['evaluated']),allow_pickle=False)
            require(s.dtype==np.float32 and s.shape==primary.shape and e.dtype==np.bool_ and e.shape==primary.shape,'Raw map contract changed')
            require(np.all((s[np.isfinite(s)]>=0)&(s[np.isfinite(s)]<=1)),'Scores outside raw sigmoid range')
            maps[arm]=s; common &= e & np.isfinite(s); records.extend([item['scores'],item['evaluated']])
        require(int((common&labels).sum())>=256 and int((common&~labels).sum())>=256,'Common primary class gate failed; no relaxation')
        case_dir=args.output_dir/case['id']; case_dir.mkdir(); np.save(case_dir/'primary-common-mask.npy',common)
        results[case['id']]=dict(physical_group=case['physical_group'],bbox_source_xywh=case['data']['bbox_source_xywh'],
            evaluation_participation_role=case['evaluation_participation_role'],fixed_primary_count=int(primary.sum()),
            common_primary_count=int(common.sum()),positive=int((common&labels).sum()),negative=int((common&~labels).sum()),models={})
        for arm,scores in maps.items():
            values=scores[common]; yy=labels[common]; ranking,curves=metric.binary_metrics(yy,values)
            np.savez_compressed(case_dir/(arm+'-curves.npz'),**curves)
            operating={r:cutoff.operating_metrics(yy,values,locks[arm][r]) for r in ['primary','secondary']}
            for op in operating.values(): op['specificity_target_met_exact']=op['confusion']['fp']<=int((~yy).sum())//20
            strata={}
            for name,mask in [('observed_nonzero_profile',data['source_observed_nonzero_mask']),('observed_zero_profile',~data['source_observed_nonzero_mask'])]:
                domain=common&mask; sub,_=metric.binary_metrics(labels[domain],scores[domain]); strata[name]=sub
            results[case['id']]['models'][arm]=dict(auc=ranking['roc_auc'],ap=ranking['average_precision'],midpoint=ranking,
                operating=operating,score_quantiles=dict(zip(['min','q05','median','q95','max'],map(float,np.quantile(values,[0,.05,.5,.95,1])))),strata=strata)
    calibration={}
    for arm,w in workers.items():
        calibration[arm]={}
        for case in plan['development']:
            _,labels,primary=evaluation(case); item=w['development'][case['id']]
            scores=np.load(verify(item['scores']),allow_pickle=False); evaluated=np.load(verify(item['evaluated']),allow_pickle=False)
            require(evaluated[primary].all() and np.isfinite(scores[primary]).all(),'Development domain changed')
            ranking,_=metric.binary_metrics(labels[primary],scores[primary]); calibration[arm][case['id']]=dict(ranking=ranking,
                operating={r:cutoff.operating_metrics(labels[primary],scores[primary],locks[arm][r]) for r in ['primary','secondary']})
            records.extend([item['scores'],item['evaluated']])
    gates={}; g=plan['gate']
    for comparator in g['comparators']:
        deltas=[]; constraints=[]
        for case in results.values():
            c=case['models']['C']; other=case['models'][comparator]
            sensitivity_delta=c['operating']['primary']['sensitivity']-other['operating']['primary']['sensitivity']
            row=dict(sensitivity=sensitivity_delta,auc=c['auc']-other['auc'],ap=c['ap']-other['ap']); deltas.append(row)
            constraints.append(c['operating']['primary']['specificity_target_met_exact'] and sensitivity_delta>=g['per_context_sensitivity_delta_min']
                and row['auc']>=g['per_context_auc_delta_min'] and row['ap']>=g['per_context_ap_delta_min'])
        mean=float(np.mean([d['sensitivity'] for d in deltas])); gates[comparator]=dict(per_context_deltas=deltas,
            mean_sensitivity_delta=mean,pass_screen=all(constraints) and mean>=g['mean_sensitivity_gain_min'])
    macro={arm:{key:float(np.mean([c['models'][arm][key] for c in results.values()])) for key in ['auc','ap']} for arm in ARMS}
    for arm in ARMS:
        macro[arm]['primary_sensitivity']=float(np.mean([c['models'][arm]['operating']['primary']['sensitivity'] for c in results.values()]))
        macro[arm]['primary_specificity']=float(np.mean([c['models'][arm]['operating']['primary']['specificity'] for c in results.values()]))
    for item in records: verify(item)
    for worker in workers.values(): verify(worker['threshold_lock'])
    for arm,worker in workers.items():
        require(record(args.run_dir/arm/'result.json')==worker_records[arm],'Worker receipt changed')
        if worker['training'] is not None:
            verify(worker['training']['checkpoint']); verify(worker['training']['steps_log'])
    load_plan(path,execute=True); require(record(path)==before,'Plan changed after analysis')
    output=dict(status='PASS',plan=before,fresh=results,calibration=calibration,equal_context_macro=macro,
        gates=gates,promising_candidate_beyond_cheap_controls=all(v['pass_screen'] for v in gates.values()),
        interpretation='One-seed continuation mechanism pilot on supplied annotations, not historical model-heldout accuracy, teacher-independent truth, statistical significance or a reading.',
        model_results=worker_records,thresholds={a:w['threshold_lock'] for a,w in workers.items()})
    write_json(args.output_dir/'metrics.json',output)
    size=sum(p.stat().st_size for p in args.run_dir.rglob('*') if p.is_file())+sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())
    require(size<=plan['limits']['output_bytes'],'Aggregate output cap failed')
    print(json.dumps(dict(status='PASS',metrics=record(args.output_dir/'metrics.json'),gates=gates)))


if __name__=='__main__': main()
