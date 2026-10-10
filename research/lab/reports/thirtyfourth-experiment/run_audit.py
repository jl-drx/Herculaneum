"""One bounded offline execution of every frozen CT feature profile."""
import resource,signal,socket,sys,time
import numpy as np
from audit_common import *
from features import features,encode,distribution
sys.path.insert(0,str(REPORTS/'twentyninth-experiment'))
from reference_metrics import rank_metrics

def offline(*a,**k):raise RuntimeError('Audit network disabled')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline

def main():
    started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('120-second audit cap')));signal.alarm(120)
    check_freeze();plan=json.loads((HERE/'PLAN.json').read_text());out=HERE/'run-v1';out.mkdir(exist_ok=False)
    try:
        contexts=[];records=distributions=defined=0
        for c in plan['contexts']:
            raw=np.load(verify(c['raw']),allow_pickle=False);y=np.load(verify(c['core64']),allow_pickle=False)
            prior=np.load(verify(c['prior_arrays']),allow_pickle=False)
            require(np.array_equal(y,prior['shifted_labels'][4]),'Prior label identity')
            masks=dict(original=np.ones((64,64),bool),interior=prior['band_masks'][2].copy());prior.close()
            arrays=features(raw);del raw
            data={}
            for scale,item in arrays.items():
                data[scale+'_numerators']=item['numerators'];data[scale+'_denominators']=item['denominators']
            target=out/(c['id']+'-features.npz');np.savez(target,**data)
            metrics=[];stats=[];summary=[]
            for scale,item in arrays.items():
                nums=item['numerators'];den=item['denominators'];depths=range(1,64) if scale=='single' else range(2,63,3)
                for di,depth in enumerate(depths):
                    for fi,feature in enumerate(['intensity','contrast']):
                        for support,mask in masks.items():
                            stats.append(dict(scale=scale,depth=depth,feature=feature,support=support,
                                classes=distribution(nums[di,fi].astype(np.float64)/int(den[fi]),y,mask)));distributions+=1
                            for polarity in (1,-1):
                                p=encode(nums[di,fi],den[fi],feature,polarity)
                                m=rank_metrics(p,y,selected=mask,prediction_valid=np.ones(mask.shape,bool));n=m['counts']
                                m['prevalence']=n['scored_positive']/(n['scored_positive']+n['scored_negative']) if n['known'] else None
                                metrics.append(dict(scale=scale,depth=depth,feature=feature,polarity=polarity,support=support,**m))
                                records+=1;defined+=m['status']=='defined'
                for feature in ('intensity','contrast'):
                    for polarity in (1,-1):
                        for support in masks:
                            subset=[r for r in metrics if (r['scale'],r['feature'],r['polarity'],r['support'])==(scale,feature,polarity,support)]
                            values=[r for r in subset if r['status']=='defined']
                            s=dict(scale=scale,feature=feature,polarity=polarity,support=support,defined_depths=len(values),total_depths=len(subset),
                                   prevalence=subset[0]['prevalence'],AUROC_gt_half=sum(r['auc']>.5 for r in values),AUROC_gt_point6=sum(r['auc']>.6 for r in values))
                            for key in ('ap','auc'):
                                v=np.array([r[key] for r in values]);s[key+'_median']=float(np.median(v)) if v.size else None
                                s[key+'_q25']=float(np.quantile(v,.25)) if v.size else None;s[key+'_q75']=float(np.quantile(v,.75)) if v.size else None
                            summary.append(s)
            result=dict(id=c['id'],source=c,features=pin(target),metrics=metrics,distributions=stats,summary=summary)
            save(out/(c['id']+'.json'),result);contexts.append(dict(id=c['id'],result=pin(out/(c['id']+'.json'))))
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=1024**3,'Worker RSS cap')
            print(json.dumps(dict(context=c['id'],metric_records=len(metrics),distribution_records=len(stats))),flush=True)
        require(records==2688 and distributions==1344,'Complete fixed family')
        require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<64*1024**2,'Phase disk cap')
        check_freeze()
        result=dict(status='PASS',created_at_utc=now(),GO=pin(HERE/'GO.json'),contexts=contexts,metric_records=records,
            defined_metric_records=defined,undefined_metric_records=records-defined,distribution_records=distributions,
            elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            new_source_requests=0,new_model_forwards=0,optimizer_updates=0,GPU_spend_GBP=0)
        save(out/'COMPLETE.json',result);print(json.dumps({k:v for k,v in result.items() if k!='contexts'}))
    except Exception as e:
        save(out/'FAILURE.json',dict(status='STOP',error=type(e).__name__+': '+str(e)));raise
if __name__=='__main__':main()
