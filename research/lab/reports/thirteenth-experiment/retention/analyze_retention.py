"""Fixed-mask ranking and common-baseline-cutoff retention, without fitting."""
import argparse
from pathlib import Path
import socket
import resource
import signal


def offline(*args, **kwargs): raise RuntimeError('Network disabled in retention analysis')
socket.create_connection=offline; socket.socket.connect=offline; socket.socket.connect_ex=offline

from retention_common import ARMS, ANALYSIS_LIMITS, load_plan, load_case, module, verify, record, read_json, write_json, collect, require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','run-dir','output-dir']: parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    signal.alarm(ANALYSIS_LIMITS['wall_seconds'])
    resource.setrlimit(resource.RLIMIT_CPU,(ANALYSIS_LIMITS['cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_AS,(ANALYSIS_LIMITS['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,)*2)
    plan,path=load_plan(args.plan,execute=True); before=record(path)
    index_record=record(args.run_dir/'results.json'); index=read_json(index_record)
    require(index['plan']==before and index['eligible_arms']==plan['eligible_arms'] and set(index['models'])==set(plan['eligible_arms']), 'Conditional run index differs')
    require(args.output_dir.resolve()!=args.run_dir.resolve() and args.run_dir.resolve() not in args.output_dir.resolve().parents,'Analysis needs fresh separate directory')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    if not plan['eligible_arms']:
        require(index['status']=='SKIPPED' and index['tile_forward_count']==0,'Zero-eligible run was not skipped')
        write_json(args.output_dir/'metrics.json',dict(status='SKIPPED',plan=before,run_index=index_record,
            eligible_arms=[],per_context={},screen=None,reason=index['skip_reason'],model_calls=0,threshold_fitting_calls=0)); signal.alarm(0); return
    require(index['status']=='PASS','Eligible run incomplete')
    import numpy as np
    metric=module(plan['metrics'],'thirteenth_binary_metrics'); cutoff=module(plan['cutoff'],'thirteenth_fixed_operating')
    workers={arm:read_json(item) for arm,item in index['models'].items()}; receipts={}; collect(index,receipts); collect(workers,receipts)
    for item in receipts.values(): verify(item)
    for arm,worker in workers.items():
        require(worker['status']=='PASS' and worker['arm']==arm and worker['plan']==before
            and worker['state_sha256']==plan['models'][arm]['state_sha256']
            and set(worker['runs'])=={c['id'] for c in plan['cases']},'Worker state/context provenance differs')
    results={}; roles=['frozen_seed42','old_A_preserve_1e4','old_B_preserve_1e4']+plan['eligible_arms']
    for case in plan['cases']:
        surface,labels,mask=load_case(plan,case); del surface
        require(int(mask.sum())==case['canonical_counts']['fixed_primary_count']
            and int((mask&labels).sum())==case['canonical_counts']['positive']
            and int((mask&~labels).sum())==case['canonical_counts']['negative'],'Canonical class counts changed')
        maps={'frozen_seed42':case['historical_maps']['frozen_seed42'],
            'old_A_preserve_1e4':case['historical_maps']['A_preserve'],'old_B_preserve_1e4':case['historical_maps']['B_preserve']}
        maps.update({arm:worker['runs'][case['id']] for arm,worker in workers.items()}); models={}
        target=args.output_dir/case['id']; target.mkdir(); np.save(target/'fixed-primary-mask.npy',mask,allow_pickle=False)
        for role,item in maps.items():
            scores=np.load(verify(item['scores']),allow_pickle=False); evaluated=np.load(verify(item['evaluated']),allow_pickle=False)
            require(scores.dtype==np.float32 and scores.shape==(384,384) and evaluated.dtype==np.bool_ and evaluated.shape==(384,384), 'Saved score shape/dtype differs')
            require(evaluated[mask].all() and np.isfinite(scores[mask]).all() and np.all((scores[mask]>=0)&(scores[mask]<=1)),
                'Fixed support has unscored/nonfinite values; no mask shrinkage')
            values=scores[mask]; yy=labels[mask]; ranking,curves=metric.binary_metrics(yy,values)
            curves_path=target/(role+'-curves.npz'); np.savez_compressed(curves_path,**curves)
            models[role]=dict(auc=ranking['roc_auc'],ap=ranking['average_precision'],
                distinct_score_groups=ranking['distinct_score_groups'],tied_score_pixel_count=ranking['tied_score_pixel_count'],
                fixed_baseline_operating=cutoff.operating_metrics(yy,values,case['fixed_baseline_cutoff']),
                curves=record(curves_path),raw_scores=item['scores'],evaluated=item['evaluated'])
        deltas={arm:{comparison:{key:models[arm][key]-models[comparison][key] for key in ['auc','ap']}
            for comparison in ['frozen_seed42','old_'+arm+'_1e4']} for arm in plan['eligible_arms']}
        results[case['id']]=dict(physical_group=case['physical_group'],input_family=case['family'],layer_start=case['layer_start'],
            fixed_mask=record(target/'fixed-primary-mask.npy'),pixel_count=int(mask.sum()),positive=int((mask&labels).sum()),
            negative=int((mask&~labels).sum()),models=models,deltas=deltas)
    groups={name:[row for row in results.values() if row['physical_group']==name] for name in ['PHerc0814','PHercParis1Fr39']}
    endpoints=lambda row,role:dict(auc=row['models'][role]['auc'],ap=row['models'][role]['ap'],
        sensitivity=row['models'][role]['fixed_baseline_operating']['sensitivity'],specificity=row['models'][role]['fixed_baseline_operating']['specificity'])
    average=lambda rows:{role:{key:float(np.mean([endpoints(row,role)[key] for row in rows]))
        for key in ['auc','ap','sensitivity','specificity']} for role in roles}
    group_macro={name:average(rows) for name,rows in groups.items()}; context_macro=average(list(results.values()))
    specimen_macro={role:{key:float(np.mean([group_macro[name][role][key] for name in groups]))
        for key in ['auc','ap','sensitivity','specificity']} for role in roles}
    screen={}
    for arm in plan['eligible_arms']:
        context_checks={name:{key:row['models'][arm][key]-row['models']['frozen_seed42'][key]>=plan['screen_contract']['per_context_'+key+'_delta_min']
            for key in ['auc','ap']} for name,row in results.items()}
        group_checks={name:{key:value[arm][key]-value['frozen_seed42'][key]>=plan['screen_contract']['domain_macro_'+key+'_delta_min'] for key in ['auc','ap']}
            for name,value in group_macro.items()}
        screen[arm]=dict(context_checks=context_checks,domain_macro_checks=group_checks,
            passed=all(all(v.values()) for v in list(context_checks.values())+list(group_checks.values())),
            automatic_promotion=False,scope='Descriptive retention screen on reused contexts only')
    load_plan(path,execute=True); require(record(path)==before,'Plan changed during analysis')
    for item in receipts.values(): verify(item)
    output=dict(status='PASS',plan=before,run_index=index_record,eligible_arms=plan['eligible_arms'],roles=roles,per_context=results,
        domain_macro=group_macro,equal_context_macro=context_macro,equal_specimen_macro=specimen_macro,screen=screen,
        baseline_cutoff=plan['baseline_cutoff'],retained_models=plan['historical'],model_results=index['models'],
        tile_forward_count=index['tile_forward_count'],model_calls=0,threshold_fitting_calls=0,
        interpretation=plan['interpretation'])
    require(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024<=plan['analysis_limits']['rss_bytes'],'Analysis RSS exceeds acceptance')
    write_json(args.output_dir/'metrics.json',output)
    require(sum(p.stat().st_size for folder in [args.run_dir,args.output_dir] for p in folder.rglob('*') if p.is_file())<=plan['limits']['output_bytes'], 'Run plus analysis output cap exceeded')
    print(__import__('json').dumps(dict(status='PASS',metrics=record(args.output_dir/'metrics.json'))))
    signal.alarm(0)


if __name__=='__main__': main()
