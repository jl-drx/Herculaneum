"""Frozen equal-window comparison and independent exact-ranking reconciliation."""
import numpy as np
from common import *
sys.path.insert(0,str(REPORTS/'twentyninth-experiment'))
from reference_metrics import rank_metrics

def independent_rank(p,labels):
    valid=labels<2;s=np.asarray(p,dtype=np.float64)[valid];y=labels[valid]
    pos=np.sort(s[y==1]);neg=np.sort(s[y==0]);all_scores=np.sort(s)
    require(len(pos)>0 and len(neg)>0,'Both classes required')
    less=np.searchsorted(neg,pos,side='left');leq=np.searchsorted(neg,pos,side='right')
    auc=float(np.sum(less+.5*(leq-less))/(len(pos)*len(neg)))
    thresholds,counts=np.unique(pos,return_counts=True)
    positive_at_or_above=len(pos)-np.searchsorted(pos,thresholds,side='left')
    all_at_or_above=len(s)-np.searchsorted(all_scores,thresholds,side='left')
    ap=float(np.sum((counts/len(pos))*(positive_at_or_above/all_at_or_above)))
    return dict(ap=ap,auc=auc)

def fixed_arms(p,d):
    require(p.shape==(5,128,128) and d in (-1,0,1),'Fixed window/displacement contract')
    q=p.astype(np.float64)
    return q[d+2,32:96,32:96],q[d+1:d+4,32:96,32:96].mean(axis=0)

def summarize(contexts):
    require(len(contexts)==4,'All four fixed contexts required')
    robust=float(np.mean([np.mean([r['AP_delta'] for r in c['displacements']]) for c in contexts]))
    clean=float(np.mean([next(r['AP_delta'] for r in c['displacements'] if r['displacement']==0) for c in contexts]))
    return dict(mean_robust_AP_delta=robust,mean_clean_AP_delta=clean,
                mean_clean_AP_loss=-clean,minimum_robust_gain=.01,maximum_clean_loss=.01,
                promising_for_replication=robust>=.01 and -clean<=.01,
                parent_count=1,independent_accuracy=False,model_promotion=False)

def main():
    freeze=check_freeze('MODEL_GO.json')
    inference=json.loads((HERE/'inference-v1/COMPLETE.json').read_text())
    require(inference['status']=='PASS' and inference['completed_forwards']==20,'Incomplete inference')
    plan=json.loads((HERE/'PLAN.json').read_text());out=HERE/'scores-v1';out.mkdir(exist_ok=False)
    contexts=[];maxdiff=0.
    for c in plan['contexts']:
        name=c['id'];entries=sorted([r for r in inference['records'] if r['context']==name],key=lambda r:r['start'])
        require([e['start'] for e in entries]==list(range(5)),'All five windows required')
        p=np.stack([np.load(verify(r['probabilities']),allow_pickle=False) for r in entries])
        labels=np.load(verify(c['core64']),allow_pickle=False);require(labels.shape==(64,64),'Core shape')
        mask=np.ones(labels.shape,bool);rows=[];arms=[]
        for d in (-1,0,1):
            baseline,candidate=fixed_arms(p,d);arms.append(np.stack([baseline,candidate]))
            metrics={}
            for role,values in [('baseline',baseline),('candidate',candidate)]:
                m=rank_metrics(values,labels,selected=mask,prediction_valid=mask)
                require(m['status']=='defined' and m['counts']['missing_prediction']==0,'Metric support')
                reference=independent_rank(values,labels)
                for key in ('ap','auc'):
                    difference=abs(m[key]-reference[key]);maxdiff=max(maxdiff,difference)
                    require(difference<=1e-12,'Independent ranking disagreement')
                metrics[role]=m
            require(metrics['baseline']['counts']==metrics['candidate']['counts'],'Paired support differs')
            rows.append(dict(displacement=d,baseline=metrics['baseline'],candidate=metrics['candidate'],
                             AP_delta=metrics['candidate']['ap']-metrics['baseline']['ap'],
                             AUROC_delta=metrics['candidate']['auc']-metrics['baseline']['auc']))
        np.save(out/(name+'-arms.npy'),np.stack(arms),allow_pickle=False)
        contexts.append(dict(id=name,parent='PHerc51',label_source=c['core64'],
           prevalence=float(np.count_nonzero(labels==1)/np.count_nonzero(labels<2)),
           displacements=rows,mean_robust_AP_delta=float(np.mean([r['AP_delta'] for r in rows])),
           clean_AP_delta=rows[1]['AP_delta'],scores=pin(out/(name+'-arms.npy')),
           score_array_axes=['displacement_minus1_0_plus1','baseline_candidate','row','column']))
    summary=summarize(contexts)
    result=dict(status='PASS',created_at_utc=now(),summary=summary,contexts=contexts,
                independent_metric_pairs=24,independent_scalar_metrics=48,maximum_metric_difference=maxdiff,
                source_inference=pin(HERE/'inference-v1/COMPLETE.json'),source_GO=pin(HERE/'MODEL_GO.json'),
                thresholds_fit=0,scientific_forwards_added=0)
    save(out/'RESULTS.json',result)
    print(json.dumps(summary,indent=2))
    for c in contexts:print(json.dumps(dict(id=c['id'],prevalence=c['prevalence'],robust_AP_delta=c['mean_robust_AP_delta'],clean_AP_delta=c['clean_AP_delta'],rows=[{k:r[k] for k in ('displacement','AP_delta','AUROC_delta')} for r in c['displacements']])))

if __name__=='__main__':main()
