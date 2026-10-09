"""Independent slice/offset arithmetic and signed-integer rank verification."""
import resource,signal,socket,sys,time
import numpy as np
from audit_common import *
sys.path.insert(0,str(REPORTS/'thirtysecond-experiment'))
from score import independent_rank

def offline(*a,**k):raise RuntimeError('Verification network disabled')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline
started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('120-second verification cap')));signal.alarm(120)
require(not (HERE/'VERIFICATION.json').exists(),'New verification receipt required')
check_freeze();plan=json.loads((HERE/'PLAN.json').read_text());run=json.loads((HERE/'run-v1/COMPLETE.json').read_text());verify(run['GO'])
require(run['status']=='PASS' and len(run['contexts'])==4,'Complete run')
counts=dict(feature_numerators=0,scalar_spatial_depth_groups=0,pooled_FP32_values=0,metric_records=0,scalar_rank_metrics=0,distribution_records=0,distribution_values=0,profile_summaries=0)
max_rank=max_stat=0.

def quantile_sorted(v,q):
    x=(len(v)-1)*q;i=int(np.floor(x));j=int(np.ceil(x));return float(v[i]+(v[j]-v[i])*(x-i))

for src,ref in zip(plan['contexts'],run['contexts'],strict=True):
    c=json.loads(verify(ref['result']).read_text());require(c['source']==src,'Context source identity')
    raw=np.load(verify(src['raw']),allow_pickle=False);y=np.load(verify(src['core64']),allow_pickle=False)
    old=np.load(verify(src['prior_arrays']),allow_pickle=False);masks=dict(original=np.ones((64,64),bool),interior=old['band_masks'][2]);old.close()
    single=np.zeros((63,128,128),np.int64)
    for dy in range(3):
        for dx in range(3):single+=raw[:,dy::3,dx::3].astype(np.int64)
    triplet=single[0::3]+single[1::3]+single[2::3]
    for scale,sums,n,zs in [('single',single,9,range(63)),('triplet',triplet,27,range(0,63,3))]:
        for i,z in enumerate(zs):
            for yy,xx in [(0,0),(0,127),(127,0),(127,127),(64,64)]:
                value=sum(int(raw[z+dz,3*yy+dy,3*xx+dx]) for dz in range(1 if scale=='single' else 3) for dy in range(3) for dx in range(3))
                require(value==int(sums[i,yy,xx]),'Scalar pooling spot check');counts['scalar_spatial_depth_groups']+=1
    prior_pooled=np.load(verify(src['pooled']),allow_pickle=False)
    require(np.array_equal((triplet.astype(np.float64)/6939.).astype(np.float32),prior_pooled),'Prior FP32 pool scale contract')
    counts['pooled_FP32_values']+=prior_pooled.size
    independent={};arr=np.load(verify(c['features']),allow_pickle=False)
    for scale,sums,n in [('single',single,9),('triplet',triplet,27)]:
        box=np.zeros((sums.shape[0],64,64),np.int64)
        for dy in range(-4,5):
            for dx in range(-4,5):box+=sums[:,32+dy:96+dy,32+dx:96+dx]
        center=sums[:,32:96,32:96];nums=np.stack([center,81*center-box],axis=1)
        require(np.array_equal(nums,arr[scale+'_numerators']),'Direct-offset feature numerators')
        require(np.array_equal(arr[scale+'_denominators'],[n,81*n]),'Exact denominators')
        independent[scale]=(nums,[n,81*n]);counts['feature_numerators']+=nums.size
    expected_keys={(scale,depth,feat,pol,sup) for scale,depths in plan['scales'].items() for depth in depths
                   for feat in plan['features'] for pol in plan['polarities'] for sup in plan['supports']}
    actual_keys={(r['scale'],r['depth'],r['feature'],r['polarity'],r['support']) for r in c['metrics']}
    require(actual_keys==expected_keys and len(c['metrics'])==len(expected_keys),'All metric variants exactly once')
    for r in c['metrics']:
        nums,den=independent[r['scale']];di=r['depth']-1 if r['scale']=='single' else (r['depth']-2)//3;fi=['intensity','contrast'].index(r['feature'])
        signed=nums[di,fi]*r['polarity'];mask=masks[r['support']];known=mask&(y<2)
        pos=int(np.count_nonzero(known&(y==1)));neg=int(np.count_nonzero(known&(y==0)))
        expected=dict(selected=int(mask.sum()),unknown=int(np.count_nonzero(mask&(y==2))),known=pos+neg,missing_prediction=0,scored_positive=pos,scored_negative=neg)
        require(r['counts']==expected and r['prevalence']==pos/(pos+neg),'Metric support/prevalence')
        if pos and neg:
            labels=y.copy();labels[~mask]=2;rank=independent_rank(signed,labels)
            require(r['status']=='defined','Defined policy')
            for key in ('ap','auc'):
                d=abs(rank[key]-r[key]);max_rank=max(max_rank,d);require(d<1e-12,'Signed-integer rank mismatch');counts['scalar_rank_metrics']+=1
        else:require(r['status']=='undefined_requires_both_classes' and r['ap'] is None and r['auc'] is None,'Undefined policy')
        counts['metric_records']+=1
    require(len(c['distributions'])==336,'Full distribution family')
    seen=set()
    for r in c['distributions']:
        key=(r['scale'],r['depth'],r['feature'],r['support']);require(key not in seen,'Duplicate distribution');seen.add(key)
        nums,den=independent[r['scale']];di=r['depth']-1 if r['scale']=='single' else (r['depth']-2)//3;fi=['intensity','contrast'].index(r['feature']);mask=masks[r['support']]
        for cls,name in [(0,'clear'),(1,'ink')]:
            ints=np.sort(nums[di,fi][mask&(y==cls)]);stats=r['classes'][name];require(stats['count']==len(ints),'Distribution count')
            if len(ints):
                values=ints.astype(np.float64)/den[fi]
                refstats=dict(mean=float(np.sum(ints,dtype=np.int64)/len(ints)/den[fi]),q25=quantile_sorted(values,.25),median=quantile_sorted(values,.5),q75=quantile_sorted(values,.75))
                for k,expected in refstats.items():
                    diff=abs(stats[k]-expected);max_stat=max(max_stat,diff);require(diff<=1e-9,'Class distribution mismatch');counts['distribution_values']+=1
            else:require(all(stats[k] is None for k in ('mean','q25','median','q75')),'Undefined class distribution')
        counts['distribution_records']+=1
    expected_stat_keys={(a,b,c,e) for a,b,c,d,e in expected_keys};require(seen==expected_stat_keys,'Missing distribution variant')
    require(len(c['summary'])==16,'Full summary family')
    for s in c['summary']:
        subset=[r for r in c['metrics'] if all(r[k]==s[k] for k in ['scale','feature','polarity','support'])]
        values=[r for r in subset if r['status']=='defined']
        require(s['total_depths']==len(subset) and s['defined_depths']==len(values),'Summary depth counts')
        require(s['prevalence']==subset[0]['prevalence'],'Summary prevalence')
        for key,threshold in [('AUROC_gt_half',.5),('AUROC_gt_point6',.6)]:require(s[key]==sum(r['auc']>threshold for r in values),'AUROC threshold counts')
        for key in ('ap','auc'):
            vals=np.sort([r[key] for r in values])
            for suffix,q in [('q25',.25),('median',.5),('q75',.75)]:require(abs(s[key+'_'+suffix]-quantile_sorted(vals,q))<1e-12,'Summary quantile')
        counts['profile_summaries']+=1
    arr.close()
require(counts['metric_records']==2688 and counts['scalar_rank_metrics']==5376 and counts['distribution_records']==1344,'All results verified')
check_freeze();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;require(rss<=1024**3,'Verification RSS cap')
receipt=dict(status='PASS',created_at_utc=now(),verifier=pin(__file__),run=pin(HERE/'run-v1/COMPLETE.json'),counts=counts,
    maximum_rank_difference=max_rank,maximum_distribution_difference=max_stat,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=rss,
    original_primary_unchanged=True,independent_agent=False,new_model_forwards=0,new_source_requests=0)
save(HERE/'VERIFICATION.json',receipt);print(json.dumps(receipt,indent=2))
