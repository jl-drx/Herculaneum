"""Single frozen CPU comparison. --lab-root supports a verified relocated capsule."""
import argparse
import resource
import time
import common
from common import *

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab-root', type=Path, default=LAB)
    parser.add_argument('--output', type=Path, default=HERE/'run-v1')
    args=parser.parse_args();common.LAB=args.lab_root.resolve()
    p=frozen_plan();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();metric=metric_module(verify(p['metrics']))
    contexts={role:{} for role in ROLES};outputs=[]
    for c in p['cases']:
        inputs=c['inputs'];raw=load_array(inputs['labels'])
        require(np.isin(raw,[0,1,255]).all(),'Binary supplied labels')
        y=raw.astype(bool);m=load_array(inputs['primary_mask'])
        require(m.dtype==np.bool_ and y[m].sum()==c['positive'] and (~y[m]).sum()==c['negative'],'Exact fixed support/counts')
        scores={}
        for role,source in c['sources'].items():
            scores[role]=load_array(source['scores']);ev=load_array(source['evaluated'])
            require(ev.dtype==np.bool_ and ev.all(),'Full unchanged evaluated support')
            require(scores[role].dtype==np.float32 and np.isfinite(scores[role]).all(),'Retained finite FP32')
        scores['midpoint']=midpoint(scores['seed42'],scores['A_control-512'])
        target=out/(c['id']+'-midpoint.npy');np.save(target,scores['midpoint'],allow_pickle=False)
        outputs.append(dict(id=c['id'],scores=pin(target)))
        for role,s in scores.items():
            row,_=metric.binary_metrics(y[m],s[m]);pred=s[m]>=p['thresholds'][1];label=y[m]
            row['old_cutoff']=p['thresholds'][1]
            row['old_cutoff_confusion']=dict(tp=int((pred&label).sum()),fp=int((pred&~label).sum()),tn=int((~pred&~label).sum()),fn=int((~pred&label).sum()))
            contexts[role][c['id']]=row
    roles={role:summarise(contexts[role],p['cases']) for role in ROLES}
    old=json.loads(verify(p['prior_metrics']).read_text())
    for role in ROLES[:-1]:
        for cid,row in roles[role]['contexts'].items():
            for k in KEYS:
                require(abs(row[k]-old['roles'][role]['contexts'][cid][k])<1e-12,'Reproduced historical reference rank')
            require(row['confusion']==old['roles'][role]['contexts'][cid]['confusion'] and row['old_cutoff_confusion']==old['roles'][role]['contexts'][cid]['old_cutoff_confusion'],'Historical diagnostic confusion')
    result=dict(status='COMPLETED',created_at_utc=now(),roles=roles,**decisions(roles),all_regions_reused_development=True,independent_ink_accuracy=False,threshold_fitting=False,coefficient_search=False)
    save(out/'METRICS.json',result)
    elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(elapsed<p['limits']['seconds_per_process'] and peak<p['limits']['peak_RSS_bytes'],'Operational bounds')
    save(out/'COMPLETE.json',dict(status='PASS',created_at_utc=now(),GO=pin(HERE/'GO.json'),scores=outputs,metrics=pin(out/'METRICS.json'),elapsed_seconds=elapsed,peak_RSS_bytes=peak,scientific_attempts=1,new_model_forwards=0,optimizer_updates=0,GPU_spend_GBP=0))
    print(json.dumps(dict(advancement=result['advancement_screen'],descriptive=result['descriptive_relation'],elapsed_seconds=elapsed)))

if __name__=='__main__':main()
