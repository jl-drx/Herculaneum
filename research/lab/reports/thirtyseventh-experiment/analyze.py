"""Fixed paired endpoints; no threshold selection or checkpoint selection."""
import math
import numpy as np
from study_common import *

def main():
    p=plan();require(not (HERE/'METRICS.json').exists(),'Write once');metric=module(verify(p['metrics']),'phase37_binary_metrics');cases={c['id']:c for c in p['cases']};all_results={}
    for role in ['seed42','A_control-32','A_control-512','B_dice-32','B_dice-512']:
        receipt=None if role=='seed42' else json.loads((HERE/'run-v1'/('eval-'+role)/'COMPLETE.json').read_text())
        records={}
        for c in p['cases']:
            _,labels,mask=load_case(p,c)
            r=c['baseline_metadata'] if receipt is None else next(x for x in receipt['contexts'] if x['id']==c['id'])
            scores=np.load(verify(r['scores']),allow_pickle=False);evaluated=np.load(verify(r['evaluated']),allow_pickle=False)
            require(evaluated[mask].all() and np.isfinite(scores[mask]).all(),'Full canonical scoring support')
            m,_=metric.binary_metrics(labels[mask],scores[mask]);pred=scores[mask]>=p['cutoff'];y=labels[mask]
            m['old_cutoff_confusion']=dict(tp=int((pred&y).sum()),fp=int((pred&~y).sum()),tn=int((~pred&~y).sum()),fn=int((~pred&y).sum()))
            m['old_cutoff']=p['cutoff'];records[c['id']]=m
        parents={}
        for parent in ['Paris1','Paris2','PHerc0814']:
            rows=[records[c['id']] for c in p['cases'] if c['parent']==parent]
            parents[parent]={key:math.fsum(r[key] for r in rows)/len(rows) for key in ['average_precision','roc_auc']}
        paris={key:(parents['Paris1'][key]+parents['Paris2'][key])/2 for key in ['average_precision','roc_auc']}
        all_results[role]=dict(contexts=records,parents=parents,equal_parent_Paris=paris)
    baseline=all_results['seed42'];screens={}
    for role,r in all_results.items():
        if role=='seed42':continue
        diffs={cid:{k:m[k]-baseline['contexts'][cid][k] for k in ['average_precision','roc_auc']} for cid,m in r['contexts'].items()}
        gain=r['equal_parent_Paris']['average_precision']-baseline['equal_parent_Paris']['average_precision']
        failures=[dict(context=cid,metric=k,delta=v) for cid,row in diffs.items() for k,v in row.items() if v<-.02]
        screens[role]=dict(Paris_AP_gain=gain,minimum_gain=.01,region_deltas=diffs,regional_failures=failures,passes=gain>=.01 and not failures)
    contrasts={}
    for name,left,right in [('budget_A','A_control-512','A_control-32'),('budget_B','B_dice-512','B_dice-32'),('objective32','B_dice-32','A_control-32'),('objective512','B_dice-512','A_control-512')]:
        delta=all_results[left]['equal_parent_Paris']['average_precision']-all_results[right]['equal_parent_Paris']['average_precision']
        contrasts[name]=dict(left=left,right=right,Paris_AP_delta=delta,material_gain=delta>=.01)
    save(HERE/'METRICS.json',dict(status='COMPLETED',created_at_utc=now(),roles=all_results,screens=screens,contrasts=contrasts,all_regions_reused_development=True,independent_ink_accuracy=False,threshold_fitting=False))
    print(json.dumps(dict(screens={k:{a:b for a,b in v.items() if a!='region_deltas'} for k,v in screens.items()},contrasts=contrasts)),flush=True)
if __name__=='__main__':main()
