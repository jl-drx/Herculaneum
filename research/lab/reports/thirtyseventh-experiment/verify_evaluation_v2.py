"""Read-only scientific reconstruction then a write-once verification receipt."""
import math,time
import numpy as np
from scipy.special import expit
from study_common import *

def independent_rank(y,s):
    # Ascending score groups: pair-count AUC, suffix-precision AP.
    order=np.argsort(s,kind='stable');v=s[order];label=y[order]
    start=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1];n=np.diff(np.r_[start,len(v)])
    pos=np.add.reduceat(label.astype(np.int64),start);neg=n-pos;P=int(pos.sum());N=int(neg.sum())
    lower_neg=np.cumsum(neg)-neg
    auc=math.fsum(float(a)*(float(b)+.5*float(c)) for a,b,c in zip(pos,lower_neg,neg))/(P*N)
    suffix_pos=np.cumsum(pos[::-1])[::-1];suffix_n=np.cumsum(n[::-1])[::-1]
    ap=math.fsum(float(a)/P*float(b)/float(c) for a,b,c in zip(pos,suffix_pos,suffix_n))
    return dict(roc_auc=auc,average_precision=ap)

def main():
    started=time.monotonic();p=plan();metrics=json.loads((HERE/'METRICS.json').read_text());require(not (HERE/'EVALUATION_VERIFICATION.json').exists(),'Write once')
    labels={};masks={};checks=0;rankmax=blendmax=0.;blendpixels=0;tiles=0
    for c in p['cases']:
        if c['family']=='paris17':
            y=np.load(verify(c['arrays']['labels']),allow_pickle=False).astype(bool);known=np.load(verify(c['arrays']['known_mask']),allow_pickle=False)
            m=np.zeros((384,384),bool);m[32:352,32:352]=known[32:352,32:352];original=np.load(verify(c['arrays']['primary_mask']),allow_pickle=False)
        else:
            src=c['legacy_case']['inputs'];y=np.load(verify(src['inklabels']),allow_pickle=False).astype(bool)
            observed=np.load(verify(src['acquisition_observed_mask']),allow_pickle=False);sup=np.load(verify(src['supervision_mask']),allow_pickle=False)>0
            m=np.zeros((384,384),bool);m[64:320,64:320]=(sup&observed)[64:320,64:320];original=np.load(verify(c['primary_mask']),allow_pickle=False)
        require(np.array_equal(m,original),'Independent canonical support construction')
        require(int(y[m].sum())==c['positive'] and int((~y[m]).sum())==c['negative'],'Independent class counts');labels[c['id']]=y;masks[c['id']]=m
    w=np.hanning(128);weights=np.maximum(np.outer(w,w)/np.outer(w,w).max(),.001)
    for role,r in metrics['roles'].items():
        receipt=None if role=='seed42' else json.loads((HERE/'run-v1'/('eval-'+role)/'COMPLETE.json').read_text())
        for c in p['cases']:
            cid=c['id'];mask=masks[cid];y=labels[cid][mask];source=c['baseline_metadata'] if receipt is None else next(q for q in receipt['contexts'] if q['id']==cid)
            scores=np.load(verify(source['scores']),allow_pickle=False);evaluated=np.load(verify(source['evaluated']),allow_pickle=False);require(evaluated[mask].all(),'Full supplied support')
            independently=independent_rank(y,scores[mask].astype(np.float64));target=r['contexts'][cid]
            for k,val in independently.items():
                delta=abs(val-target[k]);rankmax=max(rankmax,delta);require(delta<1e-12,'Independent exact-tie ranking');checks+=1
            pred=scores[mask]>=p['cutoff'];conf=dict(tp=int((pred&y).sum()),fp=int((pred&~y).sum()),tn=int((~pred&~y).sum()),fn=int((~pred&y).sum()))
            require(conf==target['old_cutoff_confusion'],'Fixed old-cutoff confusion')
            mid=scores[mask]>=.5;require(dict(tp=int((mid&y).sum()),fp=int((mid&~y).sum()),tn=int((~mid&~y).sum()),fn=int((~mid&y).sum()))==target['confusion'],'Midpoint confusion')
            if receipt is not None:
                logits=np.load(verify(source['logits']),allow_pickle=False);require(logits.shape==(25,128,128) and np.isfinite(logits).all(),'All raw tiles')
                numer=np.zeros((384,384),np.float64);denom=np.zeros_like(numer)
                for j,(yy,xx) in enumerate(( (yy,xx) for yy in range(0,257,64) for xx in range(0,257,64))):
                    numer[yy:yy+128,xx:xx+128]+=expit(logits[j].astype(np.float64))*weights;denom[yy:yy+128,xx:xx+128]+=weights
                delta=float(abs(numer/denom-scores).max());blendmax=max(blendmax,delta);require(delta<2e-6,'Independent sigmoid/Hann blend tolerance')
                require(evaluated.all(),'Complete full crop coverage');blendpixels+=scores.size;tiles+=25
        for parent in ['Paris1','Paris2','PHerc0814']:
            ids=[c['id'] for c in p['cases'] if c['parent']==parent]
            for k in ['average_precision','roc_auc']:require(math.fsum(r['contexts'][i][k] for i in ids)/len(ids)==r['parents'][parent][k],'Parent macro')
        for k in ['average_precision','roc_auc']:require((r['parents']['Paris1'][k]+r['parents']['Paris2'][k])/2==r['equal_parent_Paris'][k],'Equal-parent macro')
    for role,s in metrics['screens'].items():
        r=metrics['roles'][role];base=metrics['roles']['seed42'];gain=r['equal_parent_Paris']['average_precision']-base['equal_parent_Paris']['average_precision']
        bad=[dict(context=cid,metric=k,delta=row[k]-base['contexts'][cid][k]) for cid,row in r['contexts'].items() for k in ['average_precision','roc_auc'] if row[k]-base['contexts'][cid][k]<-.02]
        require(s['Paris_AP_gain']==gain and sorted(s['regional_failures'],key=lambda x:(x['context'],x['metric']))==sorted(bad,key=lambda x:(x['context'],x['metric'])) and s['passes']==(gain>=.01 and not bad),'All screen decisions')
    for contrast in metrics['contrasts'].values():
        d=metrics['roles'][contrast['left']]['equal_parent_Paris']['average_precision']-metrics['roles'][contrast['right']]['equal_parent_Paris']['average_precision']
        require(d==contrast['Paris_AP_delta'] and contrast['material_gain']==(d>=.01),'Matched contrasts')
    require(time.monotonic()-started<120 and rss()<1024**3,'Verification resource bounds')
    save(HERE/'EVALUATION_VERIFICATION.json',dict(status='PASS',created_at_utc=now(),verifier=pin(__file__),metrics=pin(HERE/'METRICS.json'),rank_values_checked=checks,maximum_rank_difference=rankmax,blended_score_pixels=blendpixels,raw_tiles=tiles,maximum_blend_difference=blendmax,full_canonical_masks_verified=8,all_gates_verified=True,new_model_forwards=0,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=rss()))
    print(json.dumps(dict(status='PASS',rank_values=checks,maximum_rank_difference=rankmax,maximum_blend_difference=blendmax)))
if __name__=='__main__':main()
