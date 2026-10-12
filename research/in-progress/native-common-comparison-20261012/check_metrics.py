"""Pre-result qualification of evaluation and its separate recomputation."""
import argparse
import json
from pathlib import Path
import numpy as np
import evaluate
import verify_results as v
import runner as r


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cases=0
    def check(y,s,ap=None,auc=None):
        nonlocal cases
        one=evaluate.metrics(y,s);two=v.independent_metrics(y,s);v.same(one,two)
        if ap is not None:assert abs(one['ap']-ap) < 1e-14
        if auc is not None:assert abs(one['auc']-auc) < 1e-14
        cases+=1;return one
    check([0,0,1,1],[.1,.4,.35,.8],5/6,.75)
    check([0,0,1,1],[.5]*4,.5,.5)
    check([0,1,1],[0,1,1],1,1)
    check([0,1],[1,0],.5,0)
    for cls in [0,1]:
        m=check([cls]*20,np.linspace(0,1,20));assert m['ap'] is None and m['auc'] is None
    rng=np.random.default_rng(139343)
    for i in range(60):
        y=rng.integers(0,2,5+i*13);s=rng.integers(0,17,len(y))/16
        check(y,s)
        permutation=rng.permutation(len(y));v.same(evaluate.metrics(y,s),evaluate.metrics(y[permutation],s[permutation]))
    m=check([0,1],[0,1],1,1);assert m['brier']==0 and np.isfinite(m['bce'])
    check([0,1]*6,np.linspace(0,1,12,dtype=np.float32))
    for y,s in [([],[]),([0],[float('nan')]),([1],[1.1])]:
        for fn in [evaluate.metrics,v.independent_metrics]:
            try:fn(y,s)
            except AssertionError:pass
            else:raise AssertionError('Invalid score accepted')
    v.same_members(['PHerc0841','PHerc0009B'],['PHerc0009B','PHerc0841'])
    for names in [['PHerc0841'],['PHerc0841','PHerc0841']]:
        try:v.same_members(names,['PHerc0009B','PHerc0841'])
        except AssertionError:pass
        else:raise AssertionError('Incorrect parent membership accepted')
    result=dict(status='PASS',utc=r.now(),paired_metric_cases=cases,
        additional_checks=['60 permutation/tie invariance checks','six invalid-input rejections',
                           'one-class null policy','finite clipped BCE at zero/one',
                           'unordered parent membership; duplicates and omissions rejected'],
        real_model_scores_opened=False,
        code=[r.pin(Path(x.__file__)) for x in [evaluate,v]]+[r.pin(Path(__file__))])
    r.atomic(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
