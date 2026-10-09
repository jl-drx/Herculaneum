import math,copy
import numpy as np
from common import *
def main():
    p=plan();prior=json.loads(verify(p['prior_metrics']).read_text());metric=module(verify(p['metrics']),'phase38_metric');roles={k:copy.deepcopy(prior['roles'][k]) for k in ['seed42','A_control-512']}
    receipt=json.loads((HERE/'run-v1/eval-C_coverage-512/COMPLETE.json').read_text());records={}
    for c in p['cases']:
        _,y,m=load_case(p,c);s=next(r for r in receipt['contexts'] if r['id']==c['id']);scores=np.load(verify(s['scores']));ev=np.load(verify(s['evaluated']));require(ev[m].all(),'Full support')
        r,_=metric.binary_metrics(y[m],scores[m]);pred=scores[m]>=p['cutoff'];labels=y[m];r['old_cutoff']=p['cutoff'];r['old_cutoff_confusion']=dict(tp=int((pred&labels).sum()),fp=int((pred&~labels).sum()),tn=int((~pred&~labels).sum()),fn=int((~pred&labels).sum()));records[c['id']]=r
    parents={parent:{key:math.fsum(records[c['id']][key] for c in p['cases'] if c['parent']==parent)/sum(c['parent']==parent for c in p['cases']) for key in ['average_precision','roc_auc']} for parent in ['Paris1','Paris2','PHerc0814']}
    paris={key:(parents['Paris1'][key]+parents['Paris2'][key])/2 for key in ['average_precision','roc_auc']};roles['C_coverage-512']=dict(contexts=records,parents=parents,equal_parent_Paris=paris)
    deltas={reference:{cid:{key:row[key]-roles[reference]['contexts'][cid][key] for key in ['average_precision','roc_auc']} for cid,row in records.items()} for reference in ['seed42','A_control-512']}
    gains={reference:paris['average_precision']-roles[reference]['equal_parent_Paris']['average_precision'] for reference in ['seed42','A_control-512']}
    failures=[dict(context=cid,metric=key,delta=d) for cid,values in deltas['seed42'].items() for key,d in values.items() if d<-.02]
    screen=dict(Paris_AP_gains=gains,minimum_gain_each=.01,regional_failures=failures,passes=all(v>=.01 for v in gains.values()) and not failures)
    save(HERE/'METRICS.json',dict(status='COMPLETED',created_at_utc=now(),roles=roles,region_deltas=deltas,screen=screen,all_regions_reused_development=True,independent_ink_accuracy=False,threshold_fitting=False))
    print(json.dumps(screen),flush=True)
if __name__=='__main__':main()
