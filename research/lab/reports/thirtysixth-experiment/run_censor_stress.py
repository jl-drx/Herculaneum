import resource,time
import numpy as np
from common import *
from depth_reference import boundary_from_energy
started=time.monotonic();require(not (HERE/'CENSOR_STRESS.json').exists(),'Write once')
for p in json.loads((HERE/'CENSOR_STRESS_GO.json').read_text())['files']:verify(p)
records=[]
for split in [1,3,5,6,16,32,47,57,58,60,62]:
    for reverse in [False,True]:
        e=np.zeros((63,1,1));e[:split]=100
        if reverse:e=100-e
        m=boundary_from_energy(e);candidate=float(m['candidate'][0,0]);accepted=bool(m['accepted'][0,0])
        high=lambda start: (5-max(0,min(start+5,split)-start)) if reverse else max(0,min(start+5,split)-start)
        means=[100*high(i)/5 for i in range(59)];scale=max(means)
        signed=[(means[i]-means[i+5])/scale for i in range(54)];index=max(range(54),key=lambda i:abs(signed[i]));peak=abs(signed[index])
        second=max(abs(s) for i,s in enumerate(signed) if abs(i-index)>5)
        analytic_accept=peak>=.5 and second/peak<.8
        require(candidate==index+5.5 and accepted==analytic_accept and abs(m['contrast'][0,0]-peak)<1e-12,'Independent interval arithmetic')
        censored=accepted and candidate in [5.5,58.5]
        records.append(dict(split=split,reverse=reverse,true_boundary=split+.5,candidate=candidate,accepted=accepted,absolute_error=abs(candidate-split-.5),search_censored=censored,advisory_usable=accepted and not censored))
complete=json.loads((HERE/'run-v2/COMPLETE.json').read_text());real=[]
for c in complete['contexts']:
    with np.load(verify(c['arrays']),allow_pickle=False) as a:
        b=a['baseline_accepted'];d=a['baseline_depth'];flag=b&((d==5.5)|(d==58.5))
        real.append(dict(context=c['id'],original_accepted=int(b.sum()),censored=int(flag.sum()),nonendpoint_accepted=int((b&~flag).sum()),original_support=b.size))
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;require(rss<128*1024**2 and time.monotonic()-started<30,'Stress resource limits')
r=dict(status='COMPLETED_LIMITATION_CONFIRMED',created_at_utc=now(),GO=pin(HERE/'CENSOR_STRESS_GO.json'),cases=records,real_baseline_counts=real,analytic_cases_verified=22,original_accepts_outside_search=sum(x['accepted'] and not 5.5<=x['true_boundary']<=58.5 for x in records),advisory_accepts_outside_search=sum(x['advisory_usable'] and not 5.5<=x['true_boundary']<=58.5 for x in records),original_maps_unchanged=True,no_physical_surface_truth=True,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=rss)
save(HERE/'CENSOR_STRESS.json',r);print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2))
