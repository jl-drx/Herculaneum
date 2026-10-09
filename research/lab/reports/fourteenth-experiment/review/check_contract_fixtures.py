"""AST-extracted synthetic selector/screen checks; no model or retained-array reads."""
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
COMMON=HERE.parent/'implementation/calibration_common.py'
SELECTOR=HERE.parents[1]/'seventh-experiment/benchmark/operating_point.py'


def require(ok,message):
    if not ok:
        raise ValueError(message)


def extract(path,names,namespace):
    nodes=[node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name in names]
    require({node.name for node in nodes}==set(names),'Requested source functions missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


def record(path):
    data=path.read_bytes()
    return dict(path=str(path),size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def main():
    require(record(SELECTOR)['sha256']=='38a0ebbab0835845ee2213c2be9505cf9b41a129ab13dd6668ff797bfacacb73','Original selector source changed')
    ns=dict(np=np,Fraction=Fraction,require=require)
    extract(COMMON,['fraction_record','sensitivity','recovery_screen'],ns)
    extract(SELECTOR,['choose_development_threshold'],ns)
    screen=ns['recovery_screen'];select=ns['choose_development_threshold']
    def row(tp,n=50,fp=5,nn=100):return dict(tp=tp,fn=n-tp,fp=fp,tn=nn-fp)
    original={'a':row(10),'b':row(10)}
    exact=screen({'a':row(9),'b':row(11)},original)
    require(exact['passed'] and exact['per_context_sensitivity_checks']['a']['delta']['numerator']==-1
        and exact['per_context_sensitivity_checks']['a']['delta']['denominator']==50,'Inclusive rational boundary failed')
    require(9/50-1/5 < -.02,'Fixture must discriminate rounded-float subtraction')
    require(not screen({'a':row(8),'b':row(12)},original)['passed'],'Individual loss cap bypassed by mean')
    require(not screen({'a':row(9),'b':row(10)},original)['passed'],'Mean sensitivity floor omitted')
    for nn,expected in [(19,False),(20,True)]:
        base={name:row(10,nn=nn,fp=0) for name in ['a','b']}
        current={name:row(10,nn=nn,fp=1) for name in ['a','b']}
        require(screen(current,base)['passed'] is expected,'Exact specificity boundary differs')
    # Equal-context sensitivity improves while a wrongly pooled metric decreases.
    base={'a':row(0,n=1),'b':row(2,n=100)}
    current={'a':row(1,n=1),'b':row(0,n=100)}
    require(screen(current,base)['passed'] and Fraction(1,101)<Fraction(2,101),'Equal-context fixture fails to discriminate pooling')
    rejected=0
    invalid=[{'a':row(10,n=51),'b':row(10)}, {'a':row(10,nn=101),'b':row(10)},
             {'a':dict(tp=10.,fn=40,fp=5,tn=95),'b':row(10)}]
    for candidate in invalid:
        try:screen(candidate,original)
        except ValueError:rejected+=1
        else:raise ValueError('Invalid confusion support/type accepted')
    tied=np.array([.99]+[.95]*3+[.1]*36,dtype=np.float32)
    result=select(tied)
    require(result['maximum_false_positives']==2 and result['development_false_positives']==1
        and result['boundary_tie_count']==3 and result['threshold']>float(np.float32(.95)),'Whole crossing tie not excluded')
    ones=select(np.ones(20,dtype=np.float32))
    require(ones['threshold']>1 and ones['development_false_positives']==0,'Finite above-one threshold rejected')
    small=select(np.array([0],dtype=np.float32))
    require(np.float32(small['threshold']).view(np.uint32)==1 and small['development_false_positives']==0,'Smallest subnormal cutoff differs')
    output=dict(status='PASS',source=[record(COMMON),record(SELECTOR)],exact_screen_cases=6,
        rejected_confusion_cases=rejected,original_selector_cases=3,actual_retained_array_reads=0,
        checkpoint_reads=0,Torch_imports=0,model_calls=0,network_requests=0)
    target=HERE/'contract-fixtures.json';require(not target.exists(),'Preserve previous reviewer receipt')
    target.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps(record(target)))


if __name__=='__main__':main()
