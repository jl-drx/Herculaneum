"""Frozen descriptive metrics from saved arrays only; no models or thresholds."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import resource
import signal
import time
import numpy as np
from common import (PHASE,require,pin,verify,read_json,write_json,load_plan,load_array,check_coverage,encode)
from metrics import model_metrics,paired_difference,parent_summary,raw_delta


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--release',required=True)
    p.add_argument('--results',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    started=time.monotonic();signal.alarm(120);resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
    plan,contract=load_plan(args.plan,args.release)
    result_pin=pin(args.results);result=read_json(result_pin)
    require(result['status']=='PASS' and result['plan']==pin(args.plan) and result['full_model_calls']==150
        and result['model_constructions']==result['strict_loads']==1 and result['training_steps']==result['gradients']==0,
        'Incomplete native inference')
    out=Path(args.output);out.mkdir(exist_ok=False);contexts={};retained=[]
    for case in contract['cases']:
        item=result['runs'][case['id']]
        require(item['id']==case['id'] and item['status']=='PASS'
            and item['captured_tiles']==case['tile_schedule_and_captured_inputs'],'Native capture differs')
        labels=load_array(case['arrays']['labels']);mask=load_array(case['arrays']['primary_mask'])
        require(labels.dtype==np.uint8 and mask.dtype==np.bool_ and labels.shape==mask.shape==(384,384),'Fixed labels/support type differs')
        require(np.isin(labels[mask],[0,1]).all(),'Nonbinary supplied references')
        y=labels[mask].astype(bool)
        require(int(y.sum())==case['counts']['primary_positive'] and int((~y).sum())==case['counts']['primary_negative'],'Class counts differ')
        native=load_array(item['arrays']['raw_scores']);ne=load_array(item['arrays']['evaluated_mask'])
        baseline=load_array(case['seed42']['scores']);be=load_array(case['seed42']['evaluated'])
        for scores,evaluated in ((native,ne),(baseline,be)): check_coverage(scores,evaluated,mask)
        require(np.array_equal(ne,be),'Evaluated canvas differs')
        encoded={}
        for role,scores in [('native',native),('seed42',baseline)]:
            byte,decoded=encode(scores)
            if role=='native':
                require(np.array_equal(byte,load_array(item['arrays']['uint8']))
                    and np.array_equal(decoded,load_array(item['arrays']['decoded'])),'Native saved quantization differs')
                encoded[role]={key:item['arrays'][key] for key in ('uint8','decoded')}
            else:
                target=out/case['id'];target.mkdir()
                encoded[role]={}
                for key,array in [('uint8',byte),('decoded',decoded)]:
                    path=target/(key+'.npy');np.save(path,array,allow_pickle=False)
                    encoded[role][key]={**pin(path),'shape':list(array.shape),'dtype':array.dtype.str,
                        'raw_c_order_sha256':hashlib.sha256(array.tobytes()).hexdigest()}
        models={role:model_metrics(y,scores[mask]) for role,scores in [('native',native),('seed42',baseline)]}
        contexts[case['id']]=dict(parent_papyrus=case['parent_papyrus'],fragment=case['fragment'],
            counts=dict(primary=int(mask.sum()),primary_positive=int(y.sum()),primary_negative=int((~y).sum())),
            primary_mask=case['arrays']['primary_mask'],labels=case['arrays']['labels'],models=models,
            native_minus_seed42=raw_delta(models['native']['raw'],models['seed42']['raw']),
            score_differences=paired_difference(native[mask],baseline[mask]),
            score_inputs=dict(native=item['arrays']['raw_scores'],seed42=case['seed42']['scores']),
            diagnostic_arrays=encoded,full_original_primary_support_preserved=True,role='reused_development_diagnostic')
        retained.extend([case['arrays']['labels'],case['arrays']['primary_mask'],*item['arrays'].values(),
            case['seed42']['scores'],case['seed42']['evaluated']])
    parents={name:parent_summary(contexts,name) for name in ('Paris1','Paris2')}
    for record in retained: verify(record)
    verify(result_pin);load_plan(args.plan,args.release)
    require(sum(c['counts']['primary_positive'] for c in contexts.values())==86535
        and sum(c['counts']['primary_negative'] for c in contexts.values())==434259,'Aggregate primary counts differ')
    output=dict(status='PASS',plan=pin(args.plan),native_result=result_pin,input_contract=plan['sources']['input_contract'],
        per_context=contexts,per_parent=parents,model_calls=0,training_steps=0,threshold_fits=0,
        threshold_classifications=0,automatic_promotion=False,performance_pass_gate=None,
        interpretation='Descriptive supplied-reference agreement on six reused development contexts and two physical parents; no fresh holdout or causal attribution.',
        elapsed_seconds=time.monotonic()-started,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    require(output['max_rss_bytes']<=4*1024**3,'Analysis RSS exceeded')
    write_json(out/'metrics.json',output)
    require(sum(p.stat().st_size for p in PHASE.rglob('*') if p.is_file())<=plan['limits']['phase_output_bytes'],'Phase output cap exceeded')
    print({'status':'PASS','metrics':pin(out/'metrics.json')})


if __name__=='__main__':main()
