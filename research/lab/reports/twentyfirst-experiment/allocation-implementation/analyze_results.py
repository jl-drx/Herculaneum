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
ARMS=['manual_double','manual_retired']
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
    scorer,metricpin=module(HERE.parents[1]/'sixth-experiment/benchmark/binary_metrics.py','8fe2848e0b66aa0c9dcd6a05e33db0c1380e4b9781a115262585ca551891923f','retained_rank_metrics')
    selector,oppin=module(HERE.parents[1]/'seventh-experiment/benchmark/operating_point.py','38a0ebbab0835845ee2213c2be9505cf9b41a129ab13dd6668ff797bfacacb73','retained_operating_metrics')
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
    p1=fragments['PHercParis1Fr34'];g=plan['gates'];mechanism={};practical={}
    def check(collection,name,value,bound):
        collection[name]=dict(value=value,required_minimum=bound,margin=value-bound,passed=value>=bound)
    for arm in ARMS:
        check(mechanism,arm+'_own_full_improvement',training[arm]['improvement']['full'],g['mechanism']['own_full_improvement_min'])
    check(mechanism,'candidate_manual_BCE_improvement',training['manual_retired']['improvement']['base'],g['mechanism']['candidate_manual_BCE_improvement_min'])
    check(mechanism,'candidate_retired_KL_reduction_vs_control',training['manual_double']['final']['kl_retired']-training['manual_retired']['final']['kl_retired'],g['mechanism']['candidate_retired_KL_reduction_vs_control_min'])
    check(mechanism,'candidate_manual_BCE_excess_margin',training['manual_double']['final']['base']+g['mechanism']['candidate_manual_BCE_excess_vs_control_max']-training['manual_retired']['final']['base'],0.)
    for key in ['auc','ap']:
        check(practical,'Paris2_candidate_'+key+'_gain_vs_seed42',p2['manual_retired'][key]-p2['frozen_seed42'][key],g['practical']['candidate_paris2_'+key+'_gain_vs_seed42_min'])
    for ident,c in per_case.items():
        for key in ['auc','ap']:
            check(practical,ident+'_'+key+'_retention',c['models']['manual_retired'][key]-c['models']['frozen_seed42'][key],-g['practical']['candidate_maximum_per_case_rank_loss_vs_seed42'])
    assert len(mechanism)==5 and len(practical)==14
    historic=read(plan['historical_phase20_metrics']);input_pins.append(plan['historical_phase20_metrics'])
    historical_reference=dict(source=plan['historical_phase20_metrics'],scope='Historical descriptive reference only; different own objective B_M+K_M; no new predictions or reoptimization',
        training=historic['training']['explicit_depth'],Paris2=historic['Paris2_equal_fragment']['explicit_depth'],
        Paris1=historic['Paris1_equal_context']['explicit_depth'])
    for pin in input_pins:verify(pin)
    output=dict(status='PASS',scope='Saved-output DEVELOPMENT analysis; no fitting or new model execution',
        plan=planpin,training_results=trainpin,development_results=devpin,source=record(__file__),input_pins=input_pins,
        training=training,per_case=per_case,per_fragment=fragments,Paris2_equal_fragment=p2,Paris1_equal_context=p1,
        threshold=threshold,mechanism_screen=mechanism,mechanism_screen_all=all(v['passed'] for v in mechanism.values()),
        practical_screen=practical,practical_screen_all=all(v['passed'] for v in practical.values()),
        historical_phase20_explicit_depth=historical_reference,
        interpretation='Group preservation reallocation on supplied M/V roles; own objectives differ, shared objective_A/objective_B retained; development is reused',
        automatic_promotion=False,new_model_forwards=0,threshold_fits=0,network_calls=0)
    args.output_dir.mkdir();dest=args.output_dir/'metrics.json'
    with dest.open('x') as f:json.dump(output,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PASS',metrics=record(dest),mechanism_screen_all=output['mechanism_screen_all'],practical_screen_all=output['practical_screen_all'])))
if __name__=='__main__':main()
