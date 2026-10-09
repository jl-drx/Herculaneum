"""Independent small NumPy/JSON checks; no scientific data, Torch or checkpoints."""
from pathlib import Path
import json
import socket
import sys
import tempfile

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
IMPL=ROOT.parent.parent/'inference'
sys.path.insert(0,str(IMPL))
def blocked(*a,**k):raise AssertionError('Network forbidden')
socket.socket.connect=socket.socket.connect_ex=socket.create_connection=blocked
import numpy as np
import comparison_common as common
import analyze_comparison as analysis

CASES=[]
def reject(name,fn):
    try:fn()
    except (ValueError,KeyError,TypeError):CASES.append(name);return
    raise AssertionError('Expected rejection: '+name)

def main():
    target=ROOT/'synthetic-fixture-result.json'
    assert not target.exists()
    source,_=common.sources()
    selector=common.module(source['cutoff'],'review16_selector')
    metrics=common.module(source['metrics'],'review16_metrics')
    for n in (1,19,20,21,39,40,101):
        scores=(np.arange(n)%6/5).astype(np.float32)
        candidates=np.nextafter(np.unique(scores),np.float32(np.inf),dtype=np.float32)
        expected=min(float(t) for t in candidates if int(np.count_nonzero(scores.astype(np.float64)>=float(t)))<=n//20)
        result=selector.choose_development_threshold(scores)
        assert result['threshold']==expected and result['maximum_false_positives']==n//20
        CASES.append('cutoff-bruteforce-complete-ties-N'+str(n))
    rng=np.random.default_rng(1601)
    for trial in range(20):
        y=np.array([True,False]*9);s=rng.integers(0,6,len(y)).astype(np.float32)/np.float32(5)
        m,_=metrics.binary_metrics(y,s)
        pairs=s[y,None]-s[None,~y]
        auc=(int((pairs>0).sum())+.5*int((pairs==0).sum()))/pairs.size
        ap=0.;previous_tp=0
        for threshold in sorted(set(s.tolist()),reverse=True):
            predicted=s>=threshold;tp=int((predicted&y).sum())
            ap+=(tp-previous_tp)/int(y.sum())*tp/int(predicted.sum());previous_tp=tp
        assert abs(m['roc_auc']-auc)<=2e-15 and abs(m['average_precision']-ap)<=2e-15
        CASES.append('pairwise-AUC-and-direct-group-AP-'+str(trial))
    y=np.array([True,False,True,False]);s=np.array([.1,.2,.3,.4],np.float32)
    empty=analysis.derived(y,s,float(np.nextafter(np.float32(1),np.float32(np.inf))),selector)
    assert empty['precision'] is None and empty['sensitivity']==0 and empty['specificity']==1 and empty['f1']==0 and empty['zero_predicted_positives']
    CASES.append('all-negative-is-specific-but-insensitive')
    rows=[dict(auc=x,ap=x,supplied_positive_prevalence=x,operating={k:dict(empty) for k in ('retained_fourteenth','frag3_only')}) for x in (.2,.8)]
    macro=analysis.macro(rows)
    assert macro['auc']==.5 and macro['operating']['frag3_only']['precision'] is None
    assert analysis.delta(macro,macro)['operating']['frag3_only']['precision'] is None
    CASES.append('equal-context-mean-and-undefined-precision-delta')
    with tempfile.TemporaryDirectory(dir=ROOT,prefix='synthetic-gates-') as tmp:
        tmp=Path(tmp)
        for name in ('descriptor','freeze','design'):
            common.write_json(tmp/(name+'.json'),{'synthetic':name})
        descriptor,freeze,design=[common.record(tmp/(n+'.json')) for n in ('descriptor','freeze','design')]
        go=dict(status='GO',code_freeze=freeze,research_design=design)
        ct=dict(status='PASS',descriptor=descriptor,all_CT_arrays_exact=True)
        def run(g=go,c=ct):
            common.write_json(tmp/'go.json',g);common.write_json(tmp/'ct.json',c)
            common.validate_ready_receipts(descriptor,freeze,design,common.record(tmp/'go.json'),common.record(tmp/'ct.json'))
        run();CASES.append('exact-source-and-descriptor-PASS-gates')
        reject('empty-source-GO',lambda:run(g={}))
        reject('wrong-source-freeze',lambda:run(g=dict(go,code_freeze=design)))
        reject('wrong-scientific-design',lambda:run(g=dict(go,research_design=freeze)))
        reject('wrong-source-status',lambda:run(g=dict(go,status='PASS')))
        reject('empty-CT-receipt',lambda:run(c={}))
        reject('wrong-CT-descriptor',lambda:run(c=dict(ct,descriptor=design)))
        reject('failed-CT-receipt',lambda:run(c=dict(ct,status='STOP')))
        reject('truthy-inexact-CT-receipt',lambda:run(c=dict(ct,all_CT_arrays_exact=1)))
    assert 'torch' not in sys.modules
    result=dict(status='PASS',count=len(CASES),cases=CASES,
                code=[common.record(p) for p in sorted(IMPL.glob('*.py'))],reviewer=common.record(__file__),
                scientific_arrays_read=0,checkpoint_reads=0,model_calls=0,network_calls=0)
    common.write_json(target,result);print(json.dumps(dict(status='PASS',count=len(CASES),receipt=common.record(target))))
if __name__=='__main__':main()
