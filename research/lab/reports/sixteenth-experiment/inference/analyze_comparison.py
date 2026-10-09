"""Saved-output scoring at retained and Frag3-only cutoffs; no fitting or model imports."""
import argparse
import json
from pathlib import Path
import resource
import signal
import socket

import numpy as np

from comparison_common import (ROLES,EVALUATION_IDS,load_plan,load_arrays,verify_locks_ready,
    module,record,verify,read_json,write_json,require,equal_context_mean,require_primary_coverage)


def offline(*args,**kwargs):raise RuntimeError('Network disabled in saved-output analysis')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline


def derived(labels,scores,threshold,selector):
    value=selector.operating_metrics(labels,scores,threshold);c=value['confusion'];count=int(labels.size)
    value['f1']=2*c['tp']/(2*c['tp']+c['fp']+c['fn']) if 2*c['tp']+c['fp']+c['fn'] else None
    value['predicted_positive_fraction']=(c['tp']+c['fp'])/count
    value['zero_predicted_positives']=c['tp']+c['fp']==0
    value['specificity_at_least_95']=20*c['fp']<=c['fp']+c['tn']
    return value


def macro(rows):
    result={key:equal_context_mean([row[key] for row in rows]) for key in ['auc','ap','supplied_positive_prevalence']}
    result['operating']={rule:{key:equal_context_mean([row['operating'][rule][key] for row in rows])
        for key in ['sensitivity','specificity','precision','f1','predicted_positive_fraction','balanced_accuracy']}
        for rule in ['retained_fourteenth','frag3_only']}
    return result


def delta(candidate,baseline):
    result={key:None if candidate[key] is None or baseline[key] is None else candidate[key]-baseline[key]
        for key in ['auc','ap']}
    result['operating']={rule:{key:None if candidate['operating'][rule][key] is None or baseline['operating'][rule][key] is None
        else candidate['operating'][rule][key]-baseline['operating'][rule][key]
        for key in ['sensitivity','specificity','precision','f1','predicted_positive_fraction','balanced_accuracy']}
        for rule in ['retained_fourteenth','frag3_only']}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();signal.alarm(120);resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
    plan,path=load_plan(args.plan,execute=True);plan['_plan_record']=record(path)
    source_record=record(args.results);results=read_json(source_record)
    require(results['status']=='PASS' and results['plan']==plan['_plan_record'] and results['tile_forward_count']==450,'Incomplete run')
    ready,barrier=verify_locks_ready(plan,results['locks_ready']['path'],results['locks_ready']['sha256'])
    scorer=module(plan['metrics'],'sixteenth_retained_metrics');selector=module(plan['cutoff'],'sixteenth_operating_metrics')
    receipts={phase:{role:read_json(results['models'][phase][role]) for role in ROLES} for phase in ['calibration','evaluation']}
    per_context={};thresholds={role:dict(retained_fourteenth=plan['models'][role]['original_cutoff'],
        frag3_only=read_json(ready['threshold_locks'][role])['primary']) for role in ROLES}
    pins=[source_record,barrier,*[item for phase in results['models'].values() for item in phase.values()]]
    for case in plan['cases']:
        arrays=load_arrays(plan,case);mask=arrays['primary_mask'];labels=arrays['labels'][mask].astype(bool)
        phase=case['role'];phase='calibration' if phase=='calibration' else 'evaluation';models={};reference_capture=None
        for role in ROLES:
            item=receipts[phase][role]['runs'][case['id']];pins.extend([item['scores'],item['evaluated'],item['logits']])
            scores=np.load(verify(item['scores']),allow_pickle=False);evaluated=np.load(verify(item['evaluated']),allow_pickle=False)
            require_primary_coverage(scores,evaluated,mask)
            capture=item['captured_tiles'];require(len(capture)==25,'Capture count differs')
            if reference_capture is None:reference_capture=capture
            else:require(capture==reference_capture,'Normalized model inputs differ across states')
            metric,_=scorer.binary_metrics(labels,scores[mask]);models[role]=dict(auc=metric['roc_auc'],ap=metric['average_precision'],
                supplied_positive_prevalence=float(labels.mean()),operating={rule:derived(labels,scores[mask],cutoff,selector)
                    for rule,cutoff in thresholds[role].items()},scores=item['scores'],evaluated=item['evaluated'],logits=item['logits'])
        per_context[case['id']]=dict(fragment=case['fragment'],parent_papyrus=case['parent_papyrus'],role=case['role'],
            fixed_primary_mask=case['arrays']['primary_mask'],pixel_count=int(mask.sum()),positive=int(labels.sum()),negative=int((~labels).sum()),models=models,
            candidate_minus_baseline={role:delta(models[role],models['frozen_seed42']) for role in ROLES[1:]})
    fragments={}
    for fragment in dict.fromkeys(c['fragment'] for c in plan['cases']):
        ids=[c['id'] for c in plan['cases'] if c['fragment']==fragment]
        fragments[fragment]=dict(context_ids=ids,models={role:macro([per_context[k]['models'][role] for k in ids]) for role in ROLES})
    paris2_fragments=['PHercParis2Fr47','PHercParis2Fr143']
    paris2={role:macro([fragments[g]['models'][role] for g in paris2_fragments]) for role in ROLES}
    checks={role:{rule:dict(per_context={case:per_context[case]['models'][role]['operating'][rule]['specificity_at_least_95']
        for case in EVALUATION_IDS},all_four=all(per_context[c]['models'][role]['operating'][rule]['specificity_at_least_95'] for c in EVALUATION_IDS))
        for rule in ['retained_fourteenth','frag3_only']} for role in ROLES}
    differences={scope:{role:delta(data[role],data['frozen_seed42']) for role in ROLES[1:]}
        for scope,data in [('Paris2_equal_fragment',paris2),*[(g,v['models']) for g,v in fragments.items()]]}
    for pin in pins:verify(pin)
    load_plan(path,execute=True);verify_locks_ready(plan,barrier['path'],barrier['sha256'])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    output=dict(status='PASS',plan=plan['_plan_record'],results=source_record,locks_ready=barrier,roles=ROLES,thresholds=thresholds,
        per_context=per_context,per_fragment=fragments,paris2_equal_fragment=paris2,candidate_minus_baseline=differences,
        specificity_transfer_checks=checks,primary_specificity_transfer_rule='frag3_only',
        primary_specificity_transfer_all_four={role:checks[role]['frag3_only']['all_four'] for role in ROLES},
        automatic_promotion=False,model_calls=0,threshold_fitting_calls=0,
        interpretation='Conditional supplied-reference agreement; Paris2 is one parent, labels gate-selected, upstream exposure unknown; no significance/CI/promotion.',
        all_bound_inputs_unchanged_before_after=True)
    write_json(args.output_dir/'metrics.json',output)
    require(sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())<=16*1024**2,'Analysis byte cap exceeded')
    print(json.dumps(dict(status='PASS',metrics=record(args.output_dir/'metrics.json'))))


if __name__=='__main__':main()
