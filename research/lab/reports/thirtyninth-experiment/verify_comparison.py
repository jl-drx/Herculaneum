"""Independent ascending-rank reconstruction, exact midpoint and support checks."""
import argparse
import resource
import time
import common
from common import HERE, LAB, KEYS, ROLES, now, pin, save, verify, require, frozen_plan, load_array
import json
import math
from pathlib import Path
import numpy as np

def rank(y,s):
    order=np.argsort(s,kind='stable');values=s[order];labels=y[order]
    starts=np.r_[0,np.flatnonzero(values[1:]!=values[:-1])+1]
    sizes=np.diff(np.r_[starts,len(values)]);pos=np.add.reduceat(labels.astype(np.int64),starts);neg=sizes-pos
    P=int(pos.sum());N=int(neg.sum());lower=np.cumsum(neg)-neg
    auc=math.fsum(float(p)*(float(n)+.5*float(t)) for p,n,t in zip(pos,lower,neg))/(P*N)
    above_pos=np.cumsum(pos[::-1])[::-1];above_count=np.cumsum(sizes[::-1])[::-1]
    ap=math.fsum(float(a)*float(b)/float(c) for a,b,c in zip(pos,above_pos,above_count))/P
    return dict(average_precision=ap,roc_auc=auc)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab-root',type=Path,default=LAB)
    parser.add_argument('--run',type=Path,default=HERE/'run-v1')
    args=parser.parse_args();common.LAB=args.lab_root.resolve();run=args.run.resolve()
    p=frozen_plan();started=time.monotonic();receipt=json.loads((run/'COMPLETE.json').read_text())
    require(pin(run/'METRICS.json')['sha256']==receipt['metrics']['sha256'],'Result identity')
    metrics=json.loads((run/'METRICS.json').read_text());independent={r:{} for r in ROLES};checks=0;max_error=0.;pixels=0
    for c in p['cases']:
        src=c['inputs'];y=load_array(src['labels']).astype(bool)
        if c['family']=='paris17':
            observed=load_array(src['known_mask']);m=np.zeros((384,384),bool);m[32:352,32:352]=observed[32:352,32:352]
        else:
            observed=load_array(src['observed']);sup=load_array(src['supervision'])>0
            m=np.zeros((384,384),bool);m[64:320,64:320]=(observed&sup)[64:320,64:320]
        require(np.array_equal(m,load_array(src['primary_mask'])),'Independent canonical support')
        require(int(y[m].sum())==c['positive'] and int((~y[m]).sum())==c['negative'],'Independent class counts')
        source_scores={role:load_array(source['scores']) for role,source in c['sources'].items()}
        row=next(item for item in receipt['scores'] if item['id']==c['id']);midpath=run/(c['id']+'-midpoint.npy')
        require(pin(midpath)['sha256']==row['scores']['sha256'],'Midpoint bytes')
        s=np.load(midpath,allow_pickle=False)
        reconstructed=(source_scores['seed42'].astype(np.float64)+source_scores['A_control-512'].astype(np.float64))/2
        require(s.dtype==np.float64 and np.array_equal(s,reconstructed),'Independent exact FP64 midpoint')
        source_scores['midpoint']=s;pixels+=s.size
        for role,scores in source_scores.items():
            target=metrics['roles'][role]['contexts'][c['id']];actual=rank(y[m],scores[m]);independent[role][c['id']]=actual
            for key,value in actual.items():
                error=abs(value-target[key]);max_error=max(max_error,error);checks+=1
                require(error<1e-12,'Independent rank tolerance')
            for cutoff,field in [(.5,'confusion'),(p['thresholds'][1],'old_cutoff_confusion')]:
                predicted=scores[m]>=cutoff;label=y[m]
                require(dict(tp=int((predicted&label).sum()),fp=int((predicted&~label).sum()),tn=int((~predicted&~label).sum()),fn=int((~predicted&label).sum()))==target[field],'Fixed threshold diagnostic')
    for role in ROLES:
        target=metrics['roles'][role]
        parents={parent:{k:math.fsum(independent[role][c['id']][k] for c in p['cases'] if c['parent']==parent)/sum(c['parent']==parent for c in p['cases']) for k in KEYS} for parent in ('Paris1','Paris2','PHerc0814')}
        for parent in parents:
            for k in KEYS:require(abs(parents[parent][k]-target['parents'][parent][k])<1e-12,'Independent parent aggregate')
        for k in KEYS:require(abs((parents['Paris1'][k]+parents['Paris2'][k])/2-target['equal_parent_Paris'][k])<1e-12,'Independent equal-parent aggregate')
    # Reconstruct contrasts and Boolean criteria without the runner's decision helper.
    roles=metrics['roles'];base=roles['seed42'];mins={}
    for role in ROLES:
        diffs={cid:{k:row[k]-base['contexts'][cid][k] for k in KEYS} for cid,row in roles[role]['contexts'].items()}
        require(diffs==metrics['region_deltas_vs_seed42'][role],'All reported regional deltas')
        mins[role]={k:min(r[k] for r in diffs.values()) for k in KEYS}
        require(mins[role]==metrics['worst_deltas_vs_seed42'][role],'All worst deltas')
    ap=lambda role:roles[role]['equal_parent_Paris']['average_precision']
    gain={ref:ap('midpoint')-ap(ref) for ref in ('seed42','A_control-512')}
    bad=[dict(context=cid,metric=k,delta=roles['midpoint']['contexts'][cid][k]-row[k]) for cid,row in base['contexts'].items() for k in KEYS if roles['midpoint']['contexts'][cid][k]-row[k]<-.02]
    screen=metrics['advancement_screen']
    require(screen['Paris_AP_gains']==gain and screen['regional_failures']==bad and screen['passes']==(min(gain.values())>=.01 and not bad),'Independent advancement screen')
    relation=dict(Paris_AP_no_lower=ap('midpoint')>=ap('C_coverage-512'),worst_AP_delta_no_lower=mins['midpoint']['average_precision']>=mins['C_coverage-512']['average_precision'],worst_AUROC_delta_no_lower=mins['midpoint']['roc_auc']>=mins['C_coverage-512']['roc_auc'])
    relation['passes']=all(relation.values());require(relation==metrics['descriptive_relation'],'Independent descriptive relation')
    elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(elapsed<p['limits']['seconds_per_process'] and peak<p['limits']['peak_RSS_bytes'],'Verifier operational bounds')
    result=dict(status='PASS',created_at_utc=now(),rank_values_checked=checks,maximum_rank_difference=max_error,exact_midpoint_pixels=pixels,canonical_masks_checked=8,all_aggregates_and_decisions_verified=True,new_model_forwards=0,elapsed_seconds=elapsed,peak_RSS_bytes=peak,verifier=pin(__file__),metrics=pin(run/'METRICS.json'))
    save(run/'VERIFICATION.json',result);print(json.dumps(result))

if __name__=='__main__':main()
