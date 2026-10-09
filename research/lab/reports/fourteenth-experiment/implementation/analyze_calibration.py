"""Apply already locked development cutoffs to literal retained diagnostic maps."""
import argparse
import json
from fractions import Fraction
from pathlib import Path
import resource
import signal
import socket


def offline(*args,**kwargs):raise RuntimeError('Network disabled in calibration analysis')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline

from calibration_common import ARMS,ROLES,ANALYSIS_LIMITS,BASELINE_CUTOFF,load_plan,module,verify,record,read_json,write_json,collect,require,recovery_screen,sensitivity,fraction_record


def operating_delta(first,second):
    return dict(confusion={key:first['confusion'][key]-second['confusion'][key] for key in ['tp','fp','tn','fn']},
        **{key:first[key]-second[key] if first[key] is not None and second[key] is not None else None
            for key in ['sensitivity','specificity','precision']})


def mean_endpoints(rows,role,rule):
    import math
    values=[row['models'][role]['operating'][rule] for row in rows]
    return {key:math.fsum(v[key] for v in values)/len(values) if all(v[key] is not None for v in values) else None
        for key in ['sensitivity','specificity','precision']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','run-dir','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();signal.alarm(ANALYSIS_LIMITS['wall_seconds'])
    resource.setrlimit(resource.RLIMIT_CPU,(ANALYSIS_LIMITS['cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_AS,(ANALYSIS_LIMITS['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(ANALYSIS_LIMITS['output_bytes'],)*2)
    plan,path=load_plan(args.plan,execute=True);before=record(path)
    # Read and bind both new locks before importing NumPy or loading diagnostic arrays.
    index_record=record(args.run_dir/'locks-ready.json');index=read_json(index_record)
    require(index['status']=='PASS' and index['plan']==before and set(index['models'])==set(ARMS)
        and set(index['threshold_locks'])==set(ARMS) and index['tile_forward_count']==118
        and index['both_new_locks_written_before_diagnostic_analysis'] is True,'Locks-ready index differs')
    locks={arm:read_json(item) for arm,item in index['threshold_locks'].items()};baseline=read_json(plan['baseline_lock'])
    reconstruction=read_json(index['baseline_reconstruction'])
    require(reconstruction['status']=='PASS' and reconstruction['original_lock']==plan['baseline_lock']
        and reconstruction['primary']==reconstruction['secondary']==BASELINE_CUTOFF,'Baseline lock reconstruction failed')
    workers={arm:read_json(item) for arm,item in index['models'].items()};inputs={};collect(index,inputs);collect(workers,inputs)
    for item in inputs.values():verify(item)
    for arm in ARMS:
        lock,worker=locks[arm],workers[arm]
        require(lock['arm']==arm and lock['plan']==worker['plan']==before and lock['state_sha256']==worker['state_sha256']==plan['models'][arm]['state_sha256']
            and worker['threshold_lock']==index['threshold_locks'][arm] and worker['tile_forward_count']==59,'Lock/worker state lineage differs')
        require(set(lock['development'])==set(plan['negative_counts']) and set(worker['development'])==set(plan['negative_counts']), 'Three fixed calibration cases not retained')
        require(lock['primary']==max(row['selection']['threshold'] for row in lock['development'].values())
            and lock['secondary']==lock['development']['pherc1667-w029']['selection']['threshold'],'Primary/secondary lock aggregation differs')
        for case,row in lock['development'].items():
            require(row['selection']['negative_count']==plan['negative_counts'][case]
                and row['maps']==worker['development'][case],'Regional lock map/count binding differs')
    require(args.output_dir.resolve()!=args.run_dir.resolve() and args.run_dir.resolve() not in args.output_dir.resolve().parents,'Analysis requires separate fresh output')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    import numpy as np
    metric=module(plan['metrics'],'fourteenth_ranking');cutoff=module(plan['cutoff'],'fourteenth_fixed_operating')
    retained=module(plan['retention_common'],'fourteenth_diagnostic_adapter');old=read_json(plan['retention_metrics'])
    thresholds={role:dict(primary=BASELINE_CUTOFF,secondary=BASELINE_CUTOFF,common_baseline=BASELINE_CUTOFF) if role=='frozen_seed42'
        else dict(primary=locks[role]['primary'],secondary=locks[role]['secondary'],common_baseline=BASELINE_CUTOFF) for role in ROLES}
    results={};ranking_error=0.
    for case in plan['diagnostic_cases']:
        surface,labels,mask=retained.load_case(plan,case);del surface
        original=old['per_context'][case['id']]
        require(int(mask.sum())==original['pixel_count'] and int((mask&labels).sum())==original['positive']
            and int((mask&~labels).sum())==original['negative'],'Canonical diagnostic mask/class counts changed')
        models={}
        for role in ROLES:
            item=plan['diagnostic_maps'][role][case['id']]
            scores=np.load(verify(item['scores']),allow_pickle=False);evaluated=np.load(verify(item['evaluated']),allow_pickle=False)
            require(scores.dtype==np.float32 and scores.shape==(384,384) and evaluated.dtype==np.bool_ and evaluated.shape==(384,384)
                and evaluated[mask].all() and np.isfinite(scores[mask]).all(),'Fixed diagnostic support invalid; no shrinking')
            require(item['scores']==original['models'][role]['raw_scores'] and item['evaluated']==original['models'][role]['evaluated'],'Literal reused score/coverage record differs')
            yy=labels[mask];values=scores[mask];rank,_=metric.binary_metrics(yy,values)
            errors=[abs(rank['roc_auc']-original['models'][role]['auc']),abs(rank['average_precision']-original['models'][role]['ap'])]
            require(max(errors)<=1e-12,'Ranking changed for reused scores');ranking_error=max(ranking_error,*errors)
            operating={rule:cutoff.operating_metrics(yy,values,value) for rule,value in thresholds[role].items()}
            require(operating['common_baseline']==original['models'][role]['fixed_baseline_operating'],'Common-cutoff operating results changed')
            models[role]=dict(auc=rank['roc_auc'],ap=rank['average_precision'],operating=operating,
                raw_scores=item['scores'],evaluated=item['evaluated'],ranking_integrity_max_abs_error=max(errors))
        for role in ROLES:
            models[role]['deltas']=dict(primary_vs_common=operating_delta(models[role]['operating']['primary'],models[role]['operating']['common_baseline']),
                primary_vs_baseline=operating_delta(models[role]['operating']['primary'],models['frozen_seed42']['operating']['primary']))
        results[case['id']]=dict(physical_group=case['physical_group'],fixed_mask=original['fixed_mask'],pixel_count=int(mask.sum()),
            positive=int((mask&labels).sum()),negative=int((mask&~labels).sum()),models=models)
    groups={name:{key:row for key,row in results.items() if row['physical_group']==name} for name in ['PHerc0814','PHercParis1Fr39']}
    domain_macro={};screens={arm:{} for arm in ARMS};increases={arm:{} for arm in ARMS};baseline_floor={}
    for group,rows in groups.items():
        domain_macro[group]={role:dict(auc=float(np.mean([row['models'][role]['auc'] for row in rows.values()])),
            ap=float(np.mean([row['models'][role]['ap'] for row in rows.values()])),
            operating={rule:mean_endpoints(list(rows.values()),role,rule) for rule in ['primary','secondary','common_baseline']}) for role in ROLES}
        original_conf={name:row['models']['frozen_seed42']['operating']['primary']['confusion'] for name,row in rows.items()}
        baseline_floor[group]={name:20*cc['fp']<=cc['fp']+cc['tn'] for name,cc in original_conf.items()}
        for arm in ARMS:
            candidate={name:row['models'][arm]['operating']['primary']['confusion'] for name,row in rows.items()}
            screens[arm][group]=recovery_screen(candidate,original_conf)
            differences={name:sensitivity(row['models'][arm]['operating']['primary']['confusion'])
                -sensitivity(row['models'][arm]['operating']['common_baseline']['confusion']) for name,row in rows.items()}
            mean=sum(differences.values(),Fraction(0))/2
            increases[arm][group]=dict(equal_context_mean_delta=fraction_record(mean),increased=mean>0,
                per_context_delta={name:fraction_record(value) for name,value in differences.items()},primary_screen_not_redefined=True)
    load_plan(path,execute=True);require(record(path)==before,'Frozen plan changed during analysis')
    for item in inputs.values():verify(item)
    require(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024<=ANALYSIS_LIMITS['rss_bytes'],'Analysis RSS exceeds acceptance')
    output=dict(status='PASS',plan=before,locks_ready=index_record,threshold_locks=index['threshold_locks'],thresholds=thresholds,
        baseline_lock=plan['baseline_lock'],baseline_reconstruction=index['baseline_reconstruction'],roles=ROLES,per_context=results,
        domain_macro=domain_macro,screens=screens,sensitivity_increased_vs_common=increases,baseline_specificity_floor=baseline_floor,
        both_specimen_screens_passed={arm:all(row['passed'] for row in screens[arm].values()) for arm in ARMS},automatic_promotion=False,
        unchanged_ranking_max_abs_error=ranking_error,retention_metrics=plan['retention_metrics'],model_calls=0,diagnostic_fitting_calls=0,
        interpretation=plan['interpretation'])
    write_json(args.output_dir/'metrics.json',output)
    require(sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())<=ANALYSIS_LIMITS['output_bytes'],'Analysis output cap exceeded')
    print(json.dumps(dict(status='PASS',metrics=record(args.output_dir/'metrics.json'))));signal.alarm(0)


if __name__=='__main__':main()
