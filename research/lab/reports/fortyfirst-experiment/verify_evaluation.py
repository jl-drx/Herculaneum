"""Independent ranking, support, score blending and decision reconstruction."""
import math,time
import numpy as np
from scipy.special import expit
from common import *
KEYS=['average_precision','roc_auc'];ROLE='E_no_new_BCE-512'
def rank(y,s):
    order=np.argsort(s,kind='stable');v=s[order];a=y[order];starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1];n=np.diff(np.r_[starts,len(v)])
    pos=np.add.reduceat(a.astype(np.int64),starts);neg=n-pos;P=int(pos.sum());N=int(neg.sum());lower=np.cumsum(neg)-neg
    auc=math.fsum(float(x)*(float(y)+.5*float(z)) for x,y,z in zip(pos,lower,neg))/(P*N)
    sp=np.cumsum(pos[::-1])[::-1];sn=np.cumsum(n[::-1])[::-1];ap=math.fsum(float(x)/P*float(y)/float(z) for x,y,z in zip(pos,sp,sn))
    return dict(average_precision=ap,roc_auc=auc)
def main():
    start=time.monotonic();p=plan();metrics=json.loads((HERE/'METRICS.json').read_text());checks=0;max_error=0.;blendmax=0.;pixels=0
    receipts={role:json.loads(verify(item).read_text()) for role,item in p['reference_receipts'].items()}
    receipts[ROLE]=json.loads((HERE/'run-v1/eval-E_no_new_BCE-512/COMPLETE.json').read_text())
    for role,row in metrics['roles'].items():
        for c in p['cases']:
            cid=c['id']
            if c['family']=='paris17':
                y=np.load(verify(c['arrays']['labels'])).astype(bool);known=np.load(verify(c['arrays']['known_mask']));m=np.zeros((384,384),bool);m[32:352,32:352]=known[32:352,32:352];original=np.load(verify(c['arrays']['primary_mask']))
            else:
                src=c['legacy_case']['inputs'];y=np.load(verify(src['inklabels'])).astype(bool);obs=np.load(verify(src['acquisition_observed_mask']));sup=np.load(verify(src['supervision_mask']))>0;m=np.zeros((384,384),bool);m[64:320,64:320]=(obs&sup)[64:320,64:320];original=np.load(verify(c['primary_mask']))
            require(np.array_equal(m,original) and int(y[m].sum())==c['positive'] and int((~y[m]).sum())==c['negative'],'Canonical support/count reconstruction')
            if role=='seed42':source=c['baseline_metadata']
            elif role=='midpoint':source=next(r for r in receipts[role]['scores'] if r['id']==cid)
            else:source=next(r for r in receipts[role]['contexts'] if r['id']==cid)
            s=np.load(verify(source['scores']),allow_pickle=False);target=row['contexts'][cid]
            if 'evaluated' in source:require(np.load(verify(source['evaluated']))[m].all(),'All fixed support scored')
            for k,value in rank(y[m],s[m].astype(float)).items():
                error=abs(value-target[k]);max_error=max(error,max_error);require(error<1e-12,'Independent exact-tie rank');checks+=1
            for cutoff,field in [(.5,'confusion'),(p['cutoff'],'old_cutoff_confusion')]:
                pred=s[m]>=cutoff;yy=y[m];require(dict(tp=int((pred&yy).sum()),fp=int((pred&~yy).sum()),tn=int((~pred&~yy).sum()),fn=int((~pred&yy).sum()))==target[field],'Fixed diagnostic threshold')
            if role==ROLE:
                z=np.load(verify(source['logits']));require(z.shape==(25,128,128) and np.isfinite(z).all(),'Complete raw tiles')
                h=np.hanning(128);w=np.maximum(np.outer(h,h)/np.outer(h,h).max(),.001);numer=np.zeros((384,384));denom=np.zeros_like(numer)
                for i,(yy,xx) in enumerate(( (yy,xx) for yy in range(0,257,64) for xx in range(0,257,64))):numer[yy:yy+128,xx:xx+128]+=expit(z[i].astype(float))*w;denom[yy:yy+128,xx:xx+128]+=w
                error=float(abs(numer/denom-s).max());blendmax=max(blendmax,error);require(error<2e-6,'Independent full sigmoid/Hann blend');pixels+=s.size
        for parent in ['Paris1','Paris2','PHerc0814']:
            ids=[c['id'] for c in p['cases'] if c['parent']==parent]
            for k in KEYS:require(math.fsum(row['contexts'][i][k] for i in ids)/len(ids)==row['parents'][parent][k],'Parent aggregate')
        for k in KEYS:require((row['parents']['Paris1'][k]+row['parents']['Paris2'][k])/2==row['equal_parent_Paris'][k],'Equal-parent aggregate')
    roles=metrics['roles'];candidate=roles[ROLE];base=roles['seed42'];minimums={}
    for role,r in roles.items():
        minimums[role]={k:min(v[k]-base['contexts'][cid][k] for cid,v in r['contexts'].items()) for k in KEYS}
    require(minimums==metrics['worst_deltas_vs_seed42'],'Every minimum delta')
    for ref,rows in metrics['region_deltas'].items():
        for cid,row in rows.items():
            for k,v in row.items():require(v==candidate['contexts'][cid][k]-roles[ref]['contexts'][cid][k],'All reference contrasts')
    gain={ref:candidate['equal_parent_Paris']['average_precision']-roles[ref]['equal_parent_Paris']['average_precision'] for ref in ['seed42','A_control-512']}
    bad=[dict(context=cid,metric=k,delta=row[k]-base['contexts'][cid][k]) for cid,row in candidate['contexts'].items() for k in KEYS if row[k]-base['contexts'][cid][k]<-.02]
    screen=metrics['screen'];order=lambda x:(x['context'],x['metric']);require(screen['Paris_AP_gains']==gain and sorted(screen['regional_failures'],key=order)==sorted(bad,key=order) and screen['passes']==(all(v>=.01 for v in gain.values()) and not bad),'Frozen advancement decision')
    deltas=[candidate['equal_parent_Paris']['average_precision']-roles['D_replication-512']['equal_parent_Paris']['average_precision'],minimums[ROLE]['average_precision']-minimums['D_replication-512']['average_precision'],minimums[ROLE]['roc_auc']-minimums['D_replication-512']['roc_auc']]
    outcome='tradeoff'
    if min(deltas)>=0:outcome='new_crop_BCE_unnecessary_for_observed_tradeoff'
    elif max(deltas)<=0 and min(deltas)<0:outcome='supports_new_crop_BCE_contribution'
    expected=dict(control_minus_full_mixture=dict(zip(['Paris_AP','worst_AP','worst_AUROC'],deltas)),outcome=outcome,conditional_on_label_selected_support=True,statistical_significance_established=False)
    require(expected==metrics['supervision_contribution'],'Frozen supervision contribution criterion')
    require(time.monotonic()-start<120 and rss()<1024**3,'Verifier bounds')
    result=dict(status='PASS',created_at_utc=now(),rank_values_checked=checks,maximum_rank_difference=max_error,canonical_masks_verified=8,blended_score_pixels=pixels,raw_tiles=200,maximum_blend_difference=blendmax,all_aggregates_and_decisions_verified=True,new_model_forwards=0,elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=rss(),verifier=pin(__file__))
    save(HERE/'EVALUATION_VERIFICATION.json',result);print(json.dumps(result))
if __name__=='__main__':main()
