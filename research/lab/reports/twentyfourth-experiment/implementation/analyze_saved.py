"""Four fixed model/direction observations; no direction choice or model execution."""
from __future__ import annotations
import argparse
from pathlib import Path
import resource
import signal
import time
import numpy as np
from common import (PHASE,require,pin,verify,read_json,write_json,load_plan,load_array,check_coverage)
from metrics import rank_metrics,paired_difference,parent_summary,raw_delta,difference_of_changes


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--release',required=True)
    p.add_argument('--results',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    started=time.monotonic();signal.alarm(120);resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
    plan,contract=load_plan(args.plan,args.release);controller_pin=pin(args.results);controller=read_json(controller_pin)
    require(controller['status']=='PASS' and controller['plan']==pin(args.plan)
        and controller['full_model_calls']==300 and controller['model_constructions']==controller['strict_loads']==2,
        'Incomplete paired inference')
    results={role:read_json(controller['results'][role]) for role in ('seed42','native')}
    for role,result in results.items():
        require(result['status']=='PASS' and result['model']==role and result['direction']=='reverse'
            and result['plan']==pin(args.plan) and result['full_model_calls']==150,'Per-model inference identity differs')
    out=Path(args.output);out.mkdir(exist_ok=False);contexts={};retained=[]
    for case in contract['cases']:
        labels=load_array(case['arrays']['labels']);mask=load_array(case['arrays']['primary_mask'])
        require(labels.dtype==np.uint8 and mask.dtype==np.bool_ and labels.shape==mask.shape==(384,384),'Fixed labels/support differ')
        require(np.isin(labels[mask],[0,1]).all(),'Nonbinary supplied references')
        y=labels[mask].astype(bool);models={};score_inputs={}
        require(int(y.sum())==case['counts']['primary_positive'] and int((~y).sum())==case['counts']['primary_negative'],'Class counts differ')
        for role in ('seed42','native'):
            item=results[role]['runs'][case['id']]
            require(item['id']==case['id'] and item['model']==role and item['status']=='PASS'
                and item['captured_tiles']==case['reversed_inputs'] and item['original_captured_tiles']==case['forward_inputs']
                and item['adapter_read_direction']=='forward' and item['model_depth_permutation']==list(range(16,-1,-1)),
                'Exact normalized-depth intervention differs')
            forward_record=case['forward_outputs'][role]
            forward=load_array(forward_record['scores']);fe=load_array(forward_record['evaluated'])
            reverse=load_array(item['arrays']['raw_scores']);re=load_array(item['arrays']['evaluated_mask'])
            for scores,evaluated in ((forward,fe),(reverse,re)):check_coverage(scores,evaluated,mask)
            require(np.array_equal(fe,re),'Evaluated canvases differ')
            models[role]={direction:rank_metrics(y,scores[mask]) for direction,scores in [('forward',forward),('reverse',reverse)]}
            models[role]['reverse_minus_forward']=raw_delta(models[role]['reverse'],models[role]['forward'])
            models[role]['probability_drift']=paired_difference(reverse[mask],forward[mask])
            score_inputs[role]=dict(forward=forward_record['scores'],reverse=item['arrays']['raw_scores'])
            retained.extend([forward_record['scores'],forward_record['evaluated'],*item['arrays'].values()])
        contexts[case['id']]=dict(parent_papyrus=case['parent_papyrus'],fragment=case['fragment'],models=models,
            counts=dict(primary=int(mask.sum()),primary_positive=int(y.sum()),primary_negative=int((~y).sum())),
            primary_mask=case['arrays']['primary_mask'],labels=case['arrays']['labels'],
            score_inputs=score_inputs,difference_of_changes=difference_of_changes(models),
            full_original_primary_support_preserved=True,role='reused_development_diagnostic')
        retained.extend([case['arrays']['labels'],case['arrays']['primary_mask']])
    parents={name:parent_summary(contexts,name) for name in ('Paris1','Paris2')}
    for record in retained:verify(record)
    verify(controller_pin);load_plan(args.plan,args.release)
    require(sum(c['counts']['primary_positive'] for c in contexts.values())==86535
        and sum(c['counts']['primary_negative'] for c in contexts.values())==434259,'Aggregate primary support differs')
    output=dict(status='PASS',plan=pin(args.plan),controller_result=controller_pin,input_contract=plan['sources']['input_contract'],
        per_context=contexts,per_parent=parents,model_calls=0,training_steps=0,threshold_fits=0,threshold_classifications=0,
        automatic_promotion=False,direction_selection=False,performance_pass_gate=None,
        interpretation='Descriptive response to exact normalized depth permutation on reused development inputs; no physical orientation inference.',
        elapsed_seconds=time.monotonic()-started,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    require(output['max_rss_bytes']<=4*1024**3,'Analysis RSS exceeded')
    write_json(out/'metrics.json',output)
    require(sum(p.stat().st_size for p in PHASE.rglob('*') if p.is_file())<=plan['limits']['phase_output_bytes'],'Phase output cap exceeded')
    print({'status':'PASS','metrics':pin(out/'metrics.json')})


if __name__=='__main__':main()
