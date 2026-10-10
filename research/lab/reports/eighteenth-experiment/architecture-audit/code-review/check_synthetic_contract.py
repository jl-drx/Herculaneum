"""Independent small synthetic check; never opens a scientific array/checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
LAB = Path('/workspace/herculaneum-lab')
IMPL = LAB / 'reports/eighteenth-experiment/implementation'
sys.path.insert(0, str(IMPL))

import numpy as np
import torch
from depth_adapter import ARMS, ResidualAdapter, features, compose, manual_terms


def pin(path):
    path = Path(path).resolve(strict=True)
    value = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(value).hexdigest(), size_bytes=len(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sources = {p.name:pin(p) for p in sorted(IMPL.glob('*.py'))}
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    rng = np.random.default_rng(90183)
    x = rng.normal(size=(2,1,17,7,5)).astype(np.float32)
    xx = torch.from_numpy(x.copy())
    expected_raw = np.concatenate((x[:,:,np.maximum(np.arange(17)-1,0)], x,
                                   x[:,:,np.minimum(np.arange(17)+1,16)]), axis=1)
    difference = np.empty_like(x)
    for j in range(17):
        lo, hi = max(j-1,0), min(j+1,16)
        difference[:,:,j] = (x[:,:,hi]-x[:,:,lo])/np.float32(hi-lo)
    relative = np.broadcast_to(((np.arange(17,dtype=np.float32)-8)/8)[None,None,:,None,None], x.shape)
    expected_depth = np.concatenate((x,difference,relative), axis=1)
    check('raw_clamped_neighbors_exact', np.array_equal(features(xx,ARMS[0]).numpy(), expected_raw))
    check('candidate_stencil_coordinate_exact', np.array_equal(features(xx,ARMS[1]).numpy(), expected_depth))
    check('feature_input_unchanged', np.array_equal(xx.numpy(), x))
    for code in range(8):
        def transform(a):
            a = np.rot90(a, code%4, axes=(-2,-1))
            return np.ascontiguousarray(np.flip(a,axis=-1) if code>=4 else a)
        for arm, expected in zip(ARMS,[expected_raw,expected_depth]):
            check('D4_commutes_%s_%s'%(arm,code),
                  np.array_equal(features(torch.from_numpy(transform(x)),arm).numpy(),transform(expected)))

    ramp = torch.arange(17,dtype=torch.float32)[None,None,:,None,None].expand(2,1,17,7,5).clone()
    check('linear_profile_full_slope_endpoints', bool((features(ramp,ARMS[1])[:,1] == 1).all()))
    check('constant_profile_zero_difference', bool((features(torch.ones_like(ramp),ARMS[1])[:,1] == 0).all()))
    constructed = []
    for arm in ARMS:
        torch.manual_seed(20261008)
        constructed.append(ResidualAdapter())
    a,b = constructed
    check('801_live_optimizer_parameters',sum(p.numel() for p in a.parameters() if p.requires_grad)==801)
    check('same_initial_parameter_tensors',all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters())))
    check('no_buffers',not list(a.named_buffers()))
    check('zero_final_projection',bool((a.head.weight==0).all()) and bool((a.head.bias==0).all()))
    check('nonzero_initial_hidden_weights',bool((a.stem.weight!=0).any()))
    teacher = torch.from_numpy(rng.normal(size=(2,1,7,5)).astype(np.float32))
    mask = torch.from_numpy(rng.random((2,1,7,5)) > .35)
    hard = torch.from_numpy(rng.integers(0,2,(2,1,7,5)).astype(np.float32))
    for arm,model in zip(ARMS,constructed):
        z,r = compose(model,features(xx,arm),teacher)
        check('zero_residual_'+arm,bool((r==0).all()))
        check('baseline_bitwise_'+arm,torch.equal(z.detach().view(torch.int32),teacher.view(torch.int32)))

    # Independent scalar loss arithmetic on deliberately nonidentical logits.
    test = (teacher + torch.from_numpy(rng.normal(0,.3,teacher.shape).astype(np.float32))).requires_grad_(True)
    total,row = manual_terms(test,teacher,mask,hard)
    zn = test.detach().numpy()[mask.numpy()].astype(np.float64)
    tn = teacher.numpy()[mask.numpy()].astype(np.float64)
    yn = hard.numpy()[mask.numpy()].astype(np.float64)*.5+.25
    independent_bce = np.mean(np.logaddexp(0,zn)-yn*zn)
    independent_kl = np.mean(np.logaddexp(0,zn)-np.logaddexp(0,tn)-(zn-tn)/(1+np.exp(-tn)))
    check('manual_bce_independent_numpy',abs(row['base']-independent_bce)<2e-7)
    check('manual_kl_independent_numpy',abs(row['kl']-independent_kl)<1e-12)
    check('manual_total_fp64',total.dtype==torch.float64)
    total.backward()
    check('unknown_output_gradient_zero',bool((test.grad[~mask]==0).all()))
    check('teacher_gradient_absent',teacher.grad is None and not teacher.requires_grad)

    # Only tiny synthetic adapter updates; never constructs/loads seed42.
    representation = features(xx,ARMS[1])
    optimizer = torch.optim.AdamW(a.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,
                                 weight_decay=0,foreach=False,fused=False)
    check('optimizer_exact_membership',set(map(id,a.parameters()))==set(map(id,optimizer.param_groups[0]['params'])))
    for step in [1,2]:
        optimizer.zero_grad(set_to_none=True)
        z,r = compose(a,representation,teacher)
        total,_ = manual_terms(z,teacher,mask,hard)
        total.backward()
        hidden = sum(float(p.grad.double().square().sum()) for p in a.stem.parameters())
        final = sum(float(p.grad.double().square().sum()) for p in a.head.parameters())
        check('synthetic_hidden_gradient_step_'+str(step),hidden==0 if step==1 else hidden>0)
        check('synthetic_final_gradient_step_'+str(step),final>0)
        torch.nn.utils.clip_grad_norm_(a.parameters(),1,error_if_nonfinite=True)
        optimizer.step()
    check('source_files_unchanged',sources=={p.name:pin(p) for p in sorted(IMPL.glob('*.py'))})
    result=dict(status='PASS',checks=checks,check_count=len(checks),source_pins=sources,
                synthetic_adapter_constructions=2,synthetic_adapter_forwards=4,
                synthetic_adapter_backwards=2,synthetic_optimizer_updates=2,
                scientific_array_reads=0,checkpoint_reads=0,backbone_constructions=0,
                backbone_forwards=0,network_calls=0,
                limitation='Synthetic arithmetic/API checks only; not scientific execution or runtime-envelope validation.')
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(status='PASS',checks=len(checks),result=pin(args.output))))


if __name__=='__main__':
    main()
