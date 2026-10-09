"""Independent direct-offset energy and scalar boundary checks; no worker math imports."""
import math,resource,time
import numpy as np
from common import *
started=time.monotonic();require(not (HERE/'VERIFICATION.json').exists(),'Write once')
for p in json.loads((HERE/'RECOVERY_GO.json').read_text())['files']:verify(p)
complete=json.loads((HERE/'run-v2/COMPLETE.json').read_text());plan=json.loads((HERE/'PLAN.json').read_text())
gate=json.loads(verify(complete['synthetic_gate']).read_text())
require(len(gate['cases'])==41 and gate['passed']==41 and gate['failed']==0,'Every fixed synthetic case')
for r in gate['cases']:
    require(r['coverage']==r['accepted']/r['pixels'],'Synthetic coverage')
    expected=(r['coverage']>=.95 and r['accepted_and_within_tolerance']/r['pixels']>=.95) if r['case'] in ['single','sloped'] else r['coverage']<=.05
    require(r['passed'] is expected and expected,'Synthetic gate reconstruction')
values=scalars=histogram_pixels=0;maxdiff=0.;edge=[]
for c,result in zip(plan['contexts'],complete['contexts']):
    require(c['id']==result['id'],'Context order')
    with np.load(verify(c['arrays']),allow_pickle=False) as a:x=a['CT_mean3']
    with np.load(verify(result['arrays']),allow_pickle=False) as a:
        padded=np.pad(x,((1,1),(0,0),(0,0)),mode='edge')
        smoothed=(padded[:-2]+padded[1:-1]+padded[2:])/3
        for name,input_volume,ekey in [('baseline',x,'energy'),('smooth',smoothed,'smooth_energy')]:
            # Explicit shifted sums replace scipy's separable uniform filter.
            energy=np.zeros((63,120,120),np.float64)
            for dy in range(7):
                for dx in range(7):
                    cy=dy+1;cx=dx+1
                    gx=(input_volume[:,cy:cy+120,cx+1:cx+121]-input_volume[:,cy:cy+120,cx-1:cx+119])/2
                    gy=(input_volume[:,cy+1:cy+121,cx:cx+120]-input_volume[:,cy-1:cy+119,cx:cx+120])/2
                    energy+=gx*gx+gy*gy
            energy/=49
            delta=float(np.max(abs(energy-a[ekey])));maxdiff=max(maxdiff,delta);values+=energy.size
            require(np.allclose(energy,a[ekey],rtol=1e-9,atol=1e-6),'Direct local energy')
            for y in [0,39,79,119]:
                for xx in [0,39,79,119]:
                    v=energy[:,y,xx].tolist();windows=[math.fsum(v[i:i+5])/5 for i in range(59)];scale=max(windows)
                    scores=[(windows[k]-windows[k+5])/scale if scale else 0 for k in range(54)]
                    index=max(range(54),key=lambda k:abs(scores[k]));peak=abs(scores[index]);other=max(abs(s) for k,s in enumerate(scores) if abs(k-index)>5)
                    ratio=other/peak if peak else 1;accepted=peak>=.5 and ratio<.8 and scale>0
                    require(index+5.5==a[name+'_candidate'][y,xx] and accepted==bool(a[name+'_accepted'][y,xx]),'Scalar boundary/abstention')
                    require(abs(peak-a[name+'_contrast'][y,xx])<1e-9 and abs(ratio-a[name+'_competitor_ratio'][y,xx])<1e-9,'Scalar contrast/margin')
                    require(int(np.sign(scores[index]))==int(a[name+'_orientation'][y,xx]),'Scalar direction');scalars+=1
            accepted=a[name+'_accepted'];depth=a[name+'_depth'];r=result['variants']['baseline' if name=='baseline' else 'smooth3']
            require(np.array_equal(np.isfinite(depth),accepted),'Abstention locations')
            d=depth[accepted];hist={str(float(v)):int(n) for v,n in zip(*np.unique(d,return_counts=True))}
            require(hist==r['depth_histogram'] and sum(hist.values())==r['accepted'] and r['coverage']==r['accepted']/14400,'Every histogram pixel')
            require(np.quantile(d,[0,.25,.5,.75,1]).tolist()==r['depth_quantiles_0_25_50_75_100'],'All depth quantiles')
            require(int((a[name+'_orientation'][accepted]<0).sum())==r['rising'] and int((a[name+'_orientation'][accepted]>0).sum())==r['falling'],'Directions')
            histogram_pixels+=accepted.size
            edge.append(dict(context=c['id'],variant=name,accepted=r['accepted'],at_first_search_boundary=int((d==5.5).sum()),at_last_search_boundary=int((d==58.5).sum()),interpretation='Posthoc diagnostic: possible search-edge censoring, not an exclusion or changed map'))
        b=a['baseline_accepted'];s=a['smooth_accepted'];joint=b&s;delta=abs(a['baseline_depth'][joint]-a['smooth_depth'][joint])
        require(int(joint.sum())==result['jointly_accepted'] and int((b&~s).sum())==result['lost_acceptance'] and int((~b&s).sum())==result['gained_acceptance'],'Acceptance transitions')
        require(float(delta.mean())==result['joint_depth_MAE'] and float((delta<=2).mean())==result['joint_within_2_layers'],'Smoothing depth agreement')
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;require(rss<512*1024**2 and time.monotonic()-started<120,'Verifier resources')
save(HERE/'SEARCH_EDGE_DIAGNOSTIC.json',dict(status='DESCRIPTIVE_POSTHOC',records=edge,original_maps_unchanged=True))
receipt=dict(status='PASS',created_at_utc=now(),verifier=pin(__file__),complete=pin(HERE/'run-v2/COMPLETE.json'),verified_energy_values=values,maximum_energy_difference=maxdiff,scalar_boundaries_checked=scalars,histogram_support_pixels=histogram_pixels,synthetic_receipt_records=41,synthetic_images_independently_regenerated=False,
    search_edge_diagnostic=pin(HERE/'SEARCH_EDGE_DIAGNOSTIC.json'),elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=rss,new_source_requests=0,new_model_forwards=0)
save(HERE/'VERIFICATION.json',receipt);print(json.dumps(receipt,indent=2))
