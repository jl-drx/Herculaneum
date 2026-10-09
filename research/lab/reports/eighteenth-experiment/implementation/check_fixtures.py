"""Synthetic-only fixtures; no retained scientific array/model IO or training."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from depth_common import require, write_json, record, state_hash, isolation_state, module
from depth_adapter import ARMS, ResidualAdapter, features, compose, manual_terms


def rejected(fn):
    try:fn()
    except ValueError:return True
    raise AssertionError('Invalid synthetic input was accepted')


def run():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261008);np.random.seed(20261008);random.seed(20261008)
    checks={}
    x=torch.arange(17,dtype=torch.float32).view(1,1,17,1,1).expand(2,1,17,7,9).clone()
    raw=features(x,'raw_control');depth=features(x,'explicit_depth')
    require(torch.equal(raw[:,0,:,0,0],torch.tensor([0,*range(16)],dtype=torch.float32).expand(2,17))
            and torch.equal(raw[:,2,:,0,0],torch.tensor([*range(1,17),16],dtype=torch.float32).expand(2,17)),
            'Replicate-end raw neighbors differ')
    require(torch.equal(raw[:,1:2],x) and torch.equal(depth[:,0:1],x)
            and bool((depth[:,1]==1).all()) and torch.equal(depth[:,2,:,0,0],((torch.arange(17)-8)/8).expand(2,17)),
            'Linear derivative/center coordinate differs')
    q=x*x;qd=features(q,'explicit_depth')[:,1]
    require(torch.equal(qd[0,:,0,0],torch.tensor([1,*range(2,31,2),31],dtype=torch.float32)),
            'Quadratic centered/full-slope endpoint stencil differs')
    constant=features(torch.ones_like(x),'explicit_depth')
    require(bool((constant[:,1]==0).all()),'Constant derivative differs')
    checks['feature_channel_coordinate_endpoint_cases']=4
    image=torch.randn(2,1,17,7,9);original=image.clone()
    def augment(t,code):
        t=torch.rot90(t,code%4,(-2,-1))
        return torch.flip(t,(-1,)) if code>=4 else t
    for code in range(8):
        for arm in ARMS:
            require(torch.equal(features(augment(image,code),arm),augment(features(image,arm),code)),
                    'D4 does not commute with features')
    require(torch.equal(image,original),'Feature construction mutated X')
    checks['d4_commutation_cases']=16
    checks['invalid_input_rejection_cases']=sum([rejected(lambda:features(x.double(),'raw_control')),
                                                rejected(lambda:features(x[:,:,:16],'explicit_depth')),
                                                rejected(lambda:features(x,'unknown'))])
    models=[];optimizers=[];post_rng=[]
    for arm in ARMS:
        torch.manual_seed(20261008);adapter=ResidualAdapter().train();models.append(adapter)
        post_rng.append(hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest())
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,
                                    weight_decay=0,foreach=False,fused=False);optimizers.append(optimizer)
        require(sum(p.numel() for p in adapter.parameters())==801 and not list(adapter.named_buffers()),'Model capacity/buffers differ')
        expected=list(adapter.parameters());actual=optimizer.param_groups[0]['params']
        require(len(expected)==len(actual) and all(a is b for a,b in zip(expected,actual)),'Optimizer membership differs')
    require(state_hash(models[0].state_dict())==state_hash(models[1].state_dict()) and post_rng[0]==post_rng[1],
            'Matched initialization/RNG differs')
    initial_adapter_hash=state_hash(models[0].state_dict())
    checks['801_parameter_matched_init_rng_optimizer_cases']=2
    mask=torch.zeros((2,1,7,9),dtype=torch.bool);mask[:,:,:4,:]=True
    target=torch.zeros(mask.shape);target[:,:,:,::2]=1
    teacher=torch.linspace(-3,3,mask.numel(),dtype=torch.float32).reshape(mask.shape)
    for arm,adapter,optimizer in zip(ARMS,models,optimizers):
        rep=features(image,arm)
        z,residual=compose(adapter,rep,teacher)
        require(bool((residual==0).all()) and torch.equal(z.view(torch.int32),teacher.view(torch.int32)),
                'Zero projection did not preserve bitwise baseline')
        z.retain_grad();loss,values=manual_terms(z,teacher,mask,target);loss.backward()
        require(values['kl']==0 and bool((z.grad[~mask]==0).all())
                and all(bool((p.grad==0).all()) for p in adapter.stem.parameters())
                and any(bool((p.grad!=0).any()) for p in adapter.head.parameters()),'Manual gradient/zero projection differs')
        expected=(torch.sigmoid(z.detach())-(.25+.5*target))/mask.sum()
        require(torch.allclose(z.grad[mask],expected[mask],atol=1e-8,rtol=1e-6),'BCE initial derivative differs')
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True,foreach=False)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        z,_=compose(adapter,rep,teacher);manual_terms(z,teacher,mask,target)[0].backward()
        require(any(bool((p.grad!=0).any()) for p in adapter.stem.parameters()),'Hidden gradient failed to activate')
        before=isolation_state(adapter,optimizer)
        with torch.no_grad():
            for _ in range(2):compose(adapter,rep,teacher)
        require(before==isolation_state(adapter,optimizer),'Synthetic no_grad measurement mutated state/RNG')
    checks['bitwise_init_mask_gradient_hidden_activation_isolation_cases']=2
    # One representative full-shape forward/backward measures bounded CPU feasibility only.
    torch.manual_seed(20261008);adapter=ResidualAdapter();large=torch.randn(2,1,17,128,128)
    large_teacher=torch.zeros((2,1,128,128));large_mask=torch.ones_like(large_teacher,dtype=torch.bool)
    large_target=torch.zeros_like(large_teacher);large_target[:,:,::2]=1
    started=time.perf_counter();z,_=compose(adapter,features(large,'explicit_depth'),large_teacher)
    manual_terms(z,large_teacher,large_mask,large_target)[0].backward()
    checks['full_shape_synthetic_forward_backward_seconds']=time.perf_counter()-started
    checks['scientific_array_reads']=0;checks['checkpoint_reads']=0
    checks['synthetic_optimizer_steps']=2
    return dict(status='PASS',checks=checks,initial_adapter_state_sha256=initial_adapter_hash,
                runtime=dict(python=__import__('platform').python_version(),numpy=np.__version__,torch=torch.__version__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();receipt=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,receipt);print(json.dumps(dict(status='PASS',receipt=record(args.output))))
