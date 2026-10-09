"""Preparation fixtures use generated arrays and toy modules only."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from common import LAB,require,pin,write_json,check_coverage,encode,qualified_helpers,module
from metrics import rank_metrics,model_metrics,paired_difference,parent_summary
from run_native import Capture,reconstruct


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();rows=[]
    def check(name,value):
        if not value: raise AssertionError(name)
        rows.append({'name':name,'status':'PASS'})
    def reject(name,fn):
        try: fn()
        except (ValueError,RuntimeError): rows.append({'name':name,'status':'PASS'});return
        raise AssertionError(name)
    y=np.array([0,1,0,1],bool)
    for name,scores,auc,ap in [('perfect',[0,.7,.2,1],1,1),('reversed',[1,.2,.7,0],0,5/12),
        ('all_tied',[.5]*4,.5,.5),('mixed_ties',[.1,.4,.4,.8],.875,5/6)]:
        r=rank_metrics(y,np.array(scores));check(name+'_auc',abs(r['roc_auc']-auc)<1e-12);check(name+'_ap',abs(r['average_precision']-ap)<1e-12)
    r=rank_metrics(np.array([0,0,0,1],bool),np.array([.8,.5,.1,.8]));check('imbalanced_tie_auc',abs(r['roc_auc']-5/6)<1e-12);check('imbalanced_tie_ap',r['average_precision']==.5)
    for name,a,b in [('empty',np.array([],bool),np.array([])),('single_class',np.ones(4,bool),np.arange(4)),
        ('nonfinite',y,np.array([0,1,np.nan,0])),('shape',y,np.array([0,1,2])),('dtype',y.astype(np.uint8),np.arange(4))]:
        reject(name,lambda a=a,b=b:rank_metrics(a,b))
    scores=np.array([-.2,0,.25,.5,1,1.2],np.float32);byte,decoded=encode(scores)
    check('quantization_clipped_truncation',byte.tolist()==[0,0,63,127,255,255])
    check('author_decode_endpoints',decoded.tolist()==[0,0,0,63/128,1,1])
    stats=model_metrics(y,np.array([0,.25,.5,1],np.float32))
    check('diagnostic_endpoint_fractions',stats['decoded_zero_fraction']==.5 and stats['decoded_one_fraction']==.25)
    a=np.array([0,.5,1],np.float32);b=np.array([.5,.5,0],np.float32);d=paired_difference(a,b)
    check('paired_signed',abs(d['mean_signed']-1/6)<1e-12);check('paired_absolute',d['mean_absolute']==.5)
    check('paired_linear_p95',abs(d['p95_absolute']-.95)<1e-12);check('paired_max',d['max_absolute']==1)
    mask=np.array([[1,1],[0,1]],bool);evaluated=np.ones((2,2),bool);raw=np.ones((2,2),np.float32)*.5
    check_coverage(raw,evaluated,mask);check('full_coverage_passes',True)
    broken=evaluated.copy();broken[0,0]=False;reject('missing_primary_not_intersection',lambda:check_coverage(raw,broken,mask))
    invalid=raw.copy();invalid[0,0]=np.nan;reject('nonfinite_primary',lambda:check_coverage(invalid,evaluated,mask))
    invalid=raw.copy();invalid[0,0]=1.1;reject('outside_probability_range',lambda:check_coverage(invalid,evaluated,mask))
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,x):self.calls+=1;return {'ink':x[:,:,0].clone()}
    x=torch.arange(17*128*128,dtype=torch.float32).reshape(1,1,17,128,128)%17
    digest=hashlib.sha256(x.numpy().tobytes()).hexdigest();expected=[dict(bbox_roi_xywh=[0,0,128,128],input_sha256=digest)]
    toy=Toy().eval();capture=Capture(toy,expected).eval()
    with torch.inference_mode():capture(x)
    check('toy_one_forward',toy.calls==1 and capture.captured==expected)
    bad=Toy().eval();capture_bad=Capture(bad,[dict(expected[0],input_sha256='0'*64)]).eval()
    with torch.inference_mode():reject('hash_mismatch_before_forward',lambda:capture_bad(x))
    check('mismatch_zero_forwards',bad.calls==0)
    with torch.inference_mode():reject('extra_call_rejected',lambda:capture(x))
    check('extra_call_no_forward',toy.calls==1)
    with torch.inference_mode():reject('wrong_input_shape',lambda:capture_bad(x[:,:,:16]))
    phase22=LAB/'reports/twentysecond-experiment/model-parity'
    helperplan={'sources':{'phase22_factory_helper':pin(phase22/'parity_sources.py'),'phase22_state_source':pin(phase22/'run_parity.py')}}
    factory,state_receipt=qualified_helpers(helperplan)
    class Tied(torch.nn.Module):
        def __init__(self):super().__init__();self.a=torch.nn.Parameter(torch.tensor([1.,-0.]));self.b=self.a
    tied=Tied();state={k:v.clone() for k,v in tied.state_dict().items()}
    schema=dict(tensor_schema=[dict(name=k,shape=[2]) for k in ('a','b')],expected_factory_alias_groups=[dict(names=['a','b'])],
        retained_factory_unique_storage_groups=1,expected_named_tensor_count=2,expected_named_numel=4,retained_factory_unique_parameters=2)
    receipt=state_receipt(tied,state,schema);check('qualified_state_helper_toy_pass',receipt['unique_groups']==1)
    changed={k:v.clone() for k,v in state.items()};changed['b'][1]=0.
    reject('signed_zero_alias_mismatch',lambda:state_receipt(tied,changed,schema))
    reject('missing_state_key',lambda:state_receipt(tied,{'a':state['a']},schema))
    wrong=copy.deepcopy(schema);wrong['expected_factory_alias_groups']=[]
    reject('alias_partition_mismatch',lambda:state_receipt(tied,state,wrong))
    reference=module('phase23_toy_reference',pin(LAB/'run_ink_reference.py'))
    weights=reference.hann_blend_map();boxes=[[0,0,128,128],[64,0,128,128],[0,64,128,128],[64,64,128,128]]
    probs=np.stack([np.full((128,128),.25,np.float32) for _ in boxes])
    sums,denom,raw=reconstruct(probs,boxes,weights,(192,192))
    check('hann_full_coverage',bool((denom>0).all()));check('hann_constant_preserved',bool((raw==.25).all()))
    reject('blend_tile_count_mismatch',lambda:reconstruct(probs[:3],boxes,weights,(192,192)))
    contexts={str(i):dict(parent_papyrus='Paris1',models={role:dict(raw=dict(roc_auc=a,average_precision=a/2)) for role in ('native','seed42')}) for i,a in enumerate([.2,.8])}
    summary=parent_summary(contexts,'Paris1');check('equal_context_mean',summary['models']['native']==dict(roc_auc=.5,average_precision=.25))
    check('no_spurious_parent_delta',summary['native_minus_seed42']==dict(roc_auc=0.,average_precision=0.))
    reject('wrong_parent_count',lambda:parent_summary(contexts,'Paris2'))
    write_json(args.output,dict(status='PASS',checks=rows,check_count=len(rows),real_factory_constructions=0,
        real_checkpoint_loads=0,scientific_array_reads=0,real_model_forwards=0,toy_forward_calls=1,network_requests=0))
    print({'status':'PASS','checks':len(rows),'output':args.output})


if __name__=='__main__':main()
