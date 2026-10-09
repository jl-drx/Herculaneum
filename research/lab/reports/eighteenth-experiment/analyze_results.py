"""Publish fixed development metrics using the retained benchmark implementation."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ARMS=['raw_control','explicit_depth']
def record(path):
    p=Path(path).resolve(strict=True)
    with p.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha)
def verify(pin):
    assert record(pin['path'])==pin,pin['path']
    return Path(pin['path'])
def read(pin):return json.loads(verify(pin).read_text())
def array(pin):return np.load(verify(pin),allow_pickle=False)
def module(path,sha,name):
    pin=record(path);assert pin['sha256']==sha
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m,pin
def mean(values):return None if any(v is None for v in values) else math.fsum(values)/len(values)
def macro(rows):
    return {k:mean([r[k] for r in rows]) for k in ['auc','ap','sensitivity','specificity','precision','balanced_accuracy','f1']}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','training-results','development-results','output-dir']:p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args();assert not args.output_dir.exists()
    planpin=record(args.plan);plan=read(planpin)
    trainpin=record(args.training_results);devpin=record(args.development_results)
    train=read(trainpin);dev=read(devpin)
    assert train['status']==dev['status']=='PASS' and train['plan']==dev['plan']==planpin
    assert dev['training_results']==trainpin and dev['all_six_contexts_both_final_arms_retained']
    review=read(dev['training_verification']);assert review['status']=='PASS' and review['training_results']==trainpin
    scorer,metricpin=module(HERE.parent/'sixth-experiment/benchmark/binary_metrics.py','8fe2848e0b66aa0c9dcd6a05e33db0c1380e4b9781a115262585ca551891923f','retained_rank_metrics')
    selector,oppin=module(HERE.parent/'seventh-experiment/benchmark/operating_point.py','38a0ebbab0835845ee2213c2be9505cf9b41a129ab13dd6668ff797bfacacb73','retained_operating_metrics')
    input_pins=[planpin,trainpin,devpin,metricpin,oppin,dev['training_verification']]
    training={};dev_results={}
    for arm in ARMS:
        r=read(train['results']['training'][arm]);input_pins.append(train['results']['training'][arm])
        d=read(dev['results']['development'][arm]);input_pins.append(dev['results']['development'][arm]);dev_results[arm]=d
        initial=r['baseline']['equal_batch_means'];final=r['fixed_evaluations']['32']['equal_batch_means']
        for key in initial:
            assert initial[key]==math.fsum(x[key] for x in r['baseline']['per_batch'])/32
            assert final[key]==math.fsum(x[key] for x in r['fixed_evaluations']['32']['per_batch'])/32
        training[arm]=dict(initial=initial,final=final,
            improvement={key:initial[key]-final[key] for key in ['base','full']},
            fixed_updates={k:v['equal_batch_means'] for k,v in r['fixed_evaluations'].items()},
            final_adapter_state_sha256=r['final_adapter_state_sha256'])
    per_case={};threshold=plan['gates']['fixed_threshold'];assert threshold==0.7558947205543518
    for case in plan['cases']:
        mask=array(case['arrays']['primary_mask']);labels=array(case['arrays']['labels'])
        assert mask.dtype==np.bool_ and mask.shape==labels.shape==(384,384)
        assert set(np.unique(labels)).issubset({0,1})
        target=labels[mask].astype(bool);baseline=read(case['baseline_metadata'])
        input_pins += [case['arrays']['primary_mask'],case['arrays']['labels'],case['baseline_metadata']]
        models={}
        for arm,item in [('frozen_seed42',baseline)]+[(a,dev_results[a]['runs'][case['id']]) for a in ARMS]:
            scores=array(item['scores']);evaluated=array(item['evaluated'])
            input_pins += [item['scores'],item['evaluated']]
            assert scores.dtype==np.float32 and scores.shape==mask.shape and evaluated.dtype==np.bool_ and evaluated.shape==mask.shape
            assert evaluated[mask].all() and np.isfinite(scores[mask]).all()
            rank,_=scorer.binary_metrics(target,scores[mask]);op=selector.operating_metrics(target,scores[mask],threshold)
            counts=op['confusion'];denom=2*counts['tp']+counts['fp']+counts['fn']
            models[arm]=dict(auc=rank['roc_auc'],ap=rank['average_precision'],
                sensitivity=op['sensitivity'],specificity=op['specificity'],precision=op['precision'],
                balanced_accuracy=op['balanced_accuracy'],f1=2*counts['tp']/denom if denom else None,
                confusion=counts,distinct_score_groups=rank['distinct_score_groups'],scores=item['scores'])
        per_case[case['id']]=dict(fragment=case['fragment'],parent_papyrus=case['parent_papyrus'],
            role='reused development diagnostic',positive=int(target.sum()),negative=int((~target).sum()),models=models)
    roles=['frozen_seed42']+ARMS
    fragments={f:{arm:macro([c['models'][arm] for c in per_case.values() if c['fragment']==f]) for arm in roles}
               for f in dict.fromkeys(c['fragment'] for c in per_case.values())}
    p2={arm:macro([fragments[f][arm] for f in ['PHercParis2Fr47','PHercParis2Fr143']]) for arm in roles}
    p1=fragments['PHercParis1Fr34'];g=plan['gates'];checks={}
    def check(name,value,bound):
        checks[name]=dict(value=value,required_minimum=bound,margin=value-bound,passed=value>=bound)
    for key in ['base','full']:check('training_'+key+'_improvement',training['explicit_depth']['improvement'][key],g['training_improvement_min'])
    check('full_risk_advantage_vs_raw_control',training['raw_control']['final']['full']-training['explicit_depth']['final']['full'],g['candidate_vs_control_full_loss_improvement_min'])
    for key in ['auc','ap']:
        check('Paris2_development_'+key+'_advantage',p2['explicit_depth'][key]-p2['raw_control'][key],g['paris2_'+key+'_improvement_min'])
    for ident,c in per_case.items():
        for key in ['auc','ap']:check(ident+'_'+key+'_retention',c['models']['explicit_depth'][key]-c['models']['frozen_seed42'][key],-g['maximum_per_case_rank_loss_vs_baseline'])
    for pin in input_pins:verify(pin)
    output=dict(status='PASS',scope='Saved-output DEVELOPMENT analysis; no fitting or new model execution',
        plan=planpin,training_results=trainpin,development_results=devpin,source=record(__file__),input_pins=input_pins,
        training=training,per_case=per_case,per_fragment=fragments,Paris2_equal_fragment=p2,Paris1_equal_context=p1,
        threshold=threshold,feature_screen=checks,feature_screen_all=all(v['passed'] for v in checks.values()),
        automatic_promotion=False,new_model_forwards=0,threshold_fits=0,network_calls=0)
    args.output_dir.mkdir();dest=args.output_dir/'metrics.json'
    with dest.open('x') as f:json.dump(output,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PASS',metrics=record(dest),feature_screen_all=output['feature_screen_all'])))
if __name__=='__main__':main()
