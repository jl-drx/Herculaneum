import resource,time
import numpy as np
from scipy.ndimage import uniform_filter1d
from common import *
from depth_reference import infer
from synthetic import CASES,make_case

started=time.monotonic();go=json.loads((HERE/'GO.json').read_text())
for p in go['files']:verify(p)
out=HERE/'run-v1';out.mkdir(exist_ok=False)
synthetic=[]
for kind,seed in [(k,s) for k in CASES for s in range(8)]+[('flat',0)]:
    x,truth=make_case(kind,seed);_,m=infer(x);accept=m['accepted'];coverage=float(accept.mean())
    record=dict(case=kind,seed=seed,pixels=accept.size,accepted=int(accept.sum()),coverage=coverage)
    if kind in ['single','sloped']:
        tol=1 if kind=='single' else 2
        good=accept&(abs(m['depth']-truth)<=tol)
        record.update(tolerance_layers=tol,accepted_and_within_tolerance=int(good.sum()),within_tolerance_fraction=float(good.mean()),passed=coverage>=.95 and good.mean()>=.95)
    else:record['passed']=coverage<=.05
    synthetic.append(record)
gate=dict(status='PASS' if all(r['passed'] for r in synthetic) else 'FAIL',cases=synthetic,passed=sum(r['passed'] for r in synthetic),failed=sum(not r['passed'] for r in synthetic))
save(out/'SYNTHETIC_GATE.json',gate)
if gate['status']!='PASS':
    save(out/'STOP.json',dict(reason='Fixed synthetic gate failed; real-map computation not run',gate=pin(out/'SYNTHETIC_GATE.json')))
    raise SystemExit('Synthetic gate failed; preserved without tuning')
plan=json.loads((HERE/'PLAN.json').read_text());results=[]
for c in plan['contexts']:
    with np.load(verify(c['arrays']),allow_pickle=False) as a:x=a['CT_mean3']
    e,b=infer(x);smooth=uniform_filter1d(x,size=3,axis=0,mode='nearest');es,s=infer(smooth)
    _,affine=infer(1.25*x+1000)
    require(np.array_equal(b['accepted'],affine['accepted']) and np.array_equal(b['candidate'],affine['candidate']),'Positive affine map invariance')
    require(np.allclose(b['contrast'],affine['contrast'],rtol=1e-10,atol=1e-10),'Positive affine score invariance')
    arr=dict(energy=e,smooth_energy=es)
    arr.update({'baseline_'+k:v for k,v in b.items()});arr.update({'smooth_'+k:v for k,v in s.items()})
    p=out/(c['id']+'-arrays.npz');np.savez_compressed(p,**arr)
    variants={}
    for name,m in [('baseline',b),('smooth3',s)]:
        a=m['accepted'];d=m['depth'][a]
        variants[name]=dict(pixels=a.size,accepted=int(a.sum()),coverage=float(a.mean()),rising=int((m['orientation'][a]<0).sum()),falling=int((m['orientation'][a]>0).sum()),
            depth_quantiles_0_25_50_75_100=np.quantile(d,[0,.25,.5,.75,1]).tolist() if len(d) else None,
            depth_histogram={str(float(v)):int(n) for v,n in zip(*np.unique(d,return_counts=True))})
    joint=b['accepted']&s['accepted'];diff=abs(b['depth'][joint]-s['depth'][joint])
    results.append(dict(id=c['id'],arrays=pin(p),variants=variants,jointly_accepted=int(joint.sum()),lost_acceptance=int((b['accepted']&~s['accepted']).sum()),gained_acceptance=int((~b['accepted']&s['accepted']).sum()),
        joint_depth_MAE=float(diff.mean()) if len(diff) else None,joint_within_2_layers=float((diff<=2).mean()) if len(diff) else None,positive_affine_invariance=True))
    require(time.monotonic()-started<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<512*1024**2,'Worker resources')
for p in go['files']:verify(p)
require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<64*1024**2,'Phase disk cap')
receipt=dict(status='COMPLETED',created_at_utc=now(),GO=pin(HERE/'GO.json'),synthetic_gate=pin(out/'SYNTHETIC_GATE.json'),contexts=results,
    elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_source_requests=0,new_model_forwards=0,GPU_spend_GBP=0,
    limitation='Texture proxy only; no independent physical/ink/optical truth')
save(out/'COMPLETE.json',receipt);print(json.dumps(receipt,indent=2))
