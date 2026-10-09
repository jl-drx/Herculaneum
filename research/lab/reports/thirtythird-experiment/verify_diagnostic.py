"""Offline, separate-indexing and ranking audit of the completed diagnostic."""
import resource, signal, socket, sys, time
import numpy as np
from PIL import Image
from diag_common import *
sys.path.insert(0,str(REPORTS/'thirtysecond-experiment'))
from score import independent_rank

def offline(*a,**k): raise RuntimeError('Verification is offline')
socket.create_connection=offline; socket.socket.connect=offline; socket.socket.connect_ex=offline
started=time.monotonic()
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('Verification 120-second cap')))
signal.alarm(120)
require(not (HERE/'VERIFICATION.json').exists(),'Write-once verification')
check_freeze(); plan=json.loads((HERE/'PLAN.json').read_text())
result=json.loads((HERE/'run-v1/COMPLETE.json').read_text())
verify(result['GO']); verify(result['source_primary_result'])
require(result['status']=='PASS' and len(result['contexts'])==4,'Complete four-context diagnostic')
metrics_checked=scalar_metrics=shift_labels=optical_groups=band_pixels=0; max_diff=0.

def check_metric(m, scores, labels, selected):
    global metrics_checked,scalar_metrics,max_diff
    known=selected&(labels!=2); pos=int(np.count_nonzero(known&(labels==1))); neg=int(np.count_nonzero(known&(labels==0)))
    counts=dict(selected=int(selected.sum()),unknown=int(np.count_nonzero(selected&(labels==2))),known=pos+neg,
                missing_prediction=0,scored_positive=pos,scored_negative=neg)
    require(m['counts']==counts,'Count reconciliation')
    require(m['prevalence']==(pos/(pos+neg) if pos+neg else None),'Prevalence reconciliation')
    require(np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all(),'Finite probability range')
    if pos and neg:
        require(m['status']=='defined','Unexpected undefined metric')
        y=labels.copy();y[~selected]=2
        ref=independent_rank(scores,y)
        for key in ('ap','auc'):
            delta=abs(ref[key]-m[key]);max_diff=max(max_diff,delta);scalar_metrics+=1
            require(delta<1e-12,'Independent ranking mismatch')
    else:
        require(m['status']=='undefined_requires_both_classes' and m['ap'] is None and m['auc'] is None,'Undefined policy')
    metrics_checked+=1

with Image.open(verify(plan['optical'])) as ir:
    for src,c in zip(plan['contexts'],result['contexts'],strict=True):
        require(src==c['source_context'],'Context contract differs')
        require(json.loads((HERE/'run-v1'/(c['id']+'.json')).read_text())==c,'Summary differs from context record')
        labels=np.load(verify(src['labels128']),allow_pickle=False); core=labels[32:96,32:96]; known=core<2
        require(np.array_equal(core,np.load(verify(src['core64']),allow_pickle=False)),'Core extraction')
        a=np.load(verify(c['arrays']),allow_pickle=False)
        shifts=[(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)]
        sl=np.empty((9,64,64),np.uint8)
        for i,(dy,dx) in enumerate(shifts):
            for y in range(64):
                for x in range(64):
                    sl[i,y,x]=labels[32+y-dy,32+x-dx];shift_labels+=1
        require(np.array_equal(sl,a['shifted_labels']),'Scalar label translations')
        common=np.logical_and.reduce([s!=2 for s in sl]); stable=common&np.logical_and.reduce([s==core for s in sl])
        require(np.array_equal(common,a['common']) and np.array_equal(stable,a['stable']),'Support reconstruction')
        expected_support=dict(original_known=int(known.sum()),common_known=int(common.sum()),removed_for_common=int((known&~common).sum()),
            class_stable_common=int(stable.sum()),potentially_changeable_common=int((common&~stable).sum()),optical_opaque_on_any_scored_pixel=True)
        require(c['support']==expected_support,'Support counts')
        # Direct squared lattice distances <= 16; no distance transform used.
        nearest=np.full((64,64),17,np.int32)
        for dy in range(-4,5):
            for dx in range(-4,5):
                d2=dy*dy+dx*dx
                if d2<=16:
                    different=labels[32+dy:96+dy,32+dx:96+dx]!=core
                    nearest[different]=np.minimum(nearest[different],d2)
        bands=np.stack([known&(nearest<=4),known&(nearest>4)&(nearest<=16),known&(nearest>16)])
        require(np.array_equal(bands,a['band_masks']),'Direct lattice band membership')
        require(np.array_equal(bands.sum(0),known.astype(int)),'Band partition')
        require(np.all(a['distance'][~known]==0),'Unknown distance')
        local=known&(nearest<=16)
        require(np.allclose(a['distance'][local]**2,nearest[local],rtol=0,atol=1e-12),'Near/middle exact squared distance')
        band_pixels+=core.size
        y0,x0,h,w=src['source_bbox_yxhw'];rgba=np.asarray(ir.crop((x0,y0,x0+w,y0+h)))
        require(np.array_equal(rgba[...,0],rgba[...,1]) and np.array_equal(rgba[...,0],rgba[...,2]),'Optical grayscale')
        optical=np.zeros((64,64),np.float64)
        for y in range(64):
            for x in range(64):
                total=sum(int(rgba[3*(32+y)+dy,3*(32+x)+dx,0]) for dy in range(3) for dx in range(3))
                optical[y,x]=1-total/2295.;optical_groups+=1
                if np.any(sl[:,y,x]!=2):
                    require(np.all(rgba[3*(32+y):3*(33+y),3*(32+x):3*(33+x),3]==255),'Opaque scoring footprint')
        require(np.array_equal(optical,a['optical128'][32:96,32:96]),'Scalar optical pooling')
        old=np.load(verify(src['prior_scores']),allow_pickle=False)
        pred=np.stack([old[1,0],old[1,1],optical])
        require(np.array_equal(pred,a['predictions']),'Predictor identity')
        roles=['baseline','candidate','optical_darkness'];require(c['predictor_order']==roles,'Predictor order')
        all_selected=np.ones((64,64),bool)
        for k,role in enumerate(roles):
            check_metric(c['original'][role],pred[k],core,all_selected)
            check_metric(c['stable'][role],pred[k],core,stable)
            if role!='optical_darkness':
                for key in ('counts','ap','auc','status'):
                    require(c['original'][role][key]==src['prior_clean'][role][key],'Original CT score preserved')
        for i,shift in enumerate(shifts):
            for name,mask in [('common_shifts',common),('moving_support_shifts',all_selected)]:
                row=c[name][i];require(row['shift']==list(shift),'Shift order')
                for k,role in enumerate(roles):check_metric(row['metrics'][role],pred[k],sl[i],mask)
            require(c['common_shifts'][i]['changed_labels_on_common']==int((common&(sl[i]!=core)).sum()),'Common label changes')
            moving=sl[i]<2; row=c['moving_support_shifts'][i]
            for key,value in dict(support_added_vs_original=int((moving&~known).sum()),support_removed_vs_original=int((known&~moving).sum()),
                changed_labels_on_shared_known=int((known&moving&(sl[i]!=core)).sum())).items():require(row[key]==value,'Moving support changes')
        for i,row in enumerate(c['bands']):
            require(row['band']==['near_le2','middle_gt2_le4','interior_gt4'][i],'Band order')
            for k,role in enumerate(roles):check_metric(row['metrics'][role],pred[k],core,bands[i])
        for family in ('common','moving_support'):
            rows=c[family+'_shifts']
            for role in roles:
                vals=[row['metrics'][role]['ap'] for row in rows];summary=c[family+'_shift_ranges'][role]
                for key,value in dict(defined_shifts=9,zero_AP=vals[4],min_AP=min(vals),max_AP=max(vals),AP_span=max(vals)-min(vals),
                    min_delta_vs_same_support_rule_zero=min(vals)-vals[4],max_delta_vs_same_support_rule_zero=max(vals)-vals[4]).items():
                    require(summary[key]==value,'Range summary')
        a.close()
require(metrics_checked==276 and scalar_metrics==552,'Full metric family verified')
check_freeze();verify(result['source_primary_result'])
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;require(rss<=1024**3,'Verification RSS cap')
receipt=dict(status='PASS',created_at_utc=now(),verifier=pin(__file__),diagnostic=pin(HERE/'run-v1/COMPLETE.json'),
    scalar_label_indices=shift_labels,scalar_optical_core_groups=optical_groups,direct_lattice_band_pixels=band_pixels,
    metric_records=metrics_checked,scalar_rank_metrics=scalar_metrics,maximum_metric_difference=max_diff,
    original_CT_metric_records_preserved=8,old_primary_unchanged=True,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=rss,
    independent_agent_review=False,independent_ink_truth=False,new_model_forwards=0,new_source_requests=0)
save(HERE/'VERIFICATION.json',receipt);print(json.dumps(receipt,indent=2))
