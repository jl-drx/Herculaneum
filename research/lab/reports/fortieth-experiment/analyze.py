"""Frozen endpoint and separate descriptive crop-replication criterion."""
import copy,math
import numpy as np
from common import *
KEYS=['average_precision','roc_auc']
ROLE='D_replication-512'
def main():
    p=plan();prior=json.loads(verify(p['prior_metrics']).read_text());roles=copy.deepcopy(prior['roles']);metric=module(verify(p['metrics']),'phase40_metrics')
    receipt=json.loads((HERE/'run-v1/eval-D_replication-512/COMPLETE.json').read_text());contexts={}
    for c in p['cases']:
        _,y,m=load_case(p,c);r=next(r for r in receipt['contexts'] if r['id']==c['id']);s=np.load(verify(r['scores']));ev=np.load(verify(r['evaluated']));require(ev[m].all(),'All support scored')
        row,_=metric.binary_metrics(y[m],s[m]);pred=s[m]>=p['cutoff'];yy=y[m]
        row.update(old_cutoff=p['cutoff'],old_cutoff_confusion=dict(tp=int((pred&yy).sum()),fp=int((pred&~yy).sum()),tn=int((~pred&~yy).sum()),fn=int((~pred&yy).sum())))
        contexts[c['id']]=row
    parents={parent:{k:math.fsum(contexts[c['id']][k] for c in p['cases'] if c['parent']==parent)/sum(c['parent']==parent for c in p['cases']) for k in KEYS} for parent in ['Paris1','Paris2','PHerc0814']}
    aggregate={k:(parents['Paris1'][k]+parents['Paris2'][k])/2 for k in KEYS};roles[ROLE]=dict(contexts=contexts,parents=parents,equal_parent_Paris=aggregate)
    deltas={ref:{cid:{k:row[k]-roles[ref]['contexts'][cid][k] for k in KEYS} for cid,row in contexts.items()} for ref in ['seed42','A_control-512','C_coverage-512','midpoint']}
    gains={ref:aggregate['average_precision']-roles[ref]['equal_parent_Paris']['average_precision'] for ref in ['seed42','A_control-512']}
    bad=[dict(context=cid,metric=k,delta=v) for cid,r in deltas['seed42'].items() for k,v in r.items() if v<-.02]
    worst={role:{k:min(r['contexts'][cid][k]-roles['seed42']['contexts'][cid][k] for cid in contexts) for k in KEYS} for role,r in roles.items()}
    relation=dict(Paris_AP_no_lower_than_midpoint=aggregate['average_precision']>=roles['midpoint']['equal_parent_Paris']['average_precision'],worst_AP_delta_no_lower_than_midpoint=worst[ROLE]['average_precision']>=worst['midpoint']['average_precision'],worst_AUROC_delta_no_lower_than_midpoint=worst[ROLE]['roc_auc']>=worst['midpoint']['roc_auc']);relation['passes']=all(relation.values())
    screen=dict(Paris_AP_gains=gains,minimum_gain_each=.01,regional_failures=bad,passes=all(v>=.01 for v in gains.values()) and not bad)
    save(HERE/'METRICS.json',dict(status='COMPLETED',created_at_utc=now(),roles=roles,region_deltas=deltas,worst_deltas_vs_seed42=worst,screen=screen,descriptive_replication=relation,all_regions_reused_development=True,independent_ink_accuracy=False,threshold_fitting=False))
    print(json.dumps(dict(advancement=screen,descriptive=relation)),flush=True)
if __name__=='__main__':main()
