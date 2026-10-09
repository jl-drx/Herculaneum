"""Finite generated-tensor checks; no scientific arrays, models or optimizers."""
import argparse
import hashlib
import json
from pathlib import Path
import math
import numpy as np
import torch
from preservation_objectives import objective


def require(ok,message):
    if not ok:raise ValueError(message)


def record(path):
    p=Path(path);return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)


def sigmoid(x):return 1/(1+np.exp(-x))


def oracle(z,t,m,y,v,mode):
    z=z.astype(np.float64);t=t.astype(np.float64);nm=int(m.sum());nv=int(v.sum());u=v&~m;nu=int(u.sum())
    soft=.25+.5*y[m];bce=float(np.mean(np.logaddexp(0,z[m])-soft*z[m]))
    kl=np.logaddexp(0,z)-np.logaddexp(0,t)-sigmoid(t)*(z-t)
    weights=np.zeros_like(z)
    if mode=='manual_mean':weights[m]=1/nm
    elif mode=='support_mean':weights[v]=1/nv
    else:weights[m]=.5/nm;weights[u]=.5/nu
    grad=weights*(sigmoid(z)-sigmoid(t));grad[m]+=(sigmoid(z[m])-soft)/nm
    return bce+float(np.sum(weights[v]*kl[v])),grad,bce,float(np.sum(weights[v]*kl[v]))


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);checks=[];outcomes={}
    def check(name,value):require(value,name);checks.append(name)
    z=np.array([-.7,.2,.8,-.4,1.1,-1.2,.5,.9,-.3,1.5,-2.,.1],np.float32).reshape(2,1,2,3)
    t=np.array([-.9,.4,.1,.3,.7,-.8,-.1,.4,.2,.8,-1.2,.7],np.float32).reshape(z.shape)
    m=np.array([1,0,0,0,1,0,1,0,0,0,0,0],bool).reshape(z.shape)
    v=np.array([1,1,1,0,1,0,1,1,0,1,0,1],bool).reshape(z.shape)
    y=np.full(z.shape,np.nan,np.float32);y[m]=[0,1,1]
    for mode in ['manual_mean','support_mean','stratified_mean']:
        student=torch.tensor(z,requires_grad=True);teacher=torch.tensor(t,requires_grad=True)
        masks=torch.from_numpy(m);valid=torch.from_numpy(v);target=torch.from_numpy(y)
        loss,terms=objective(student,teacher,masks,target,valid,mode);loss.backward()
        wanted,gradient,bce,kl=oracle(z,t,m,y,v,mode)
        check(mode+'_numpy_loss',abs(float(loss.detach())-wanted)<=2e-7)
        check(mode+'_numpy_gradient',np.max(np.abs(student.grad.numpy()-gradient))<=2e-8)
        check(mode+'_teacher_detached',teacher.grad is None)
        check(mode+'_manual_bce_fixed',abs(float(terms['bce'])-bce)<=2e-7)
        check(mode+'_invalid_gradient_zero',bool((student.grad[~valid]==0).all()))
        if mode=='manual_mean':check(mode+'_extra_gradient_zero',bool((student.grad[valid&~masks]==0).all()))
        else:check(mode+'_extra_gradient_nonzero',bool((student.grad[valid&~masks]!=0).all()))
        changed=z.copy();changed[~v]=np.nan
        invalid_loss,_=objective(torch.tensor(changed,requires_grad=True),teacher,masks,target,valid,mode)
        check(mode+'_invalid_values_unread',float(invalid_loss.detach())==float(loss.detach()))
        changed_target=y.copy();changed_target[~m]=123456.
        replacement,_=objective(student.detach(),teacher,masks,torch.from_numpy(changed_target),valid,mode)
        check(mode+'_nonmanual_targets_ignored',float(replacement)==float(loss.detach()))
        equal=torch.tensor(t,requires_grad=True);zero_terms=objective(equal,teacher,masks,target,valid,mode)[1]
        check(mode+'_identity_kl_exact_zero',float(zero_terms['kl'])==0.)
        zero_terms['kl'].backward();check(mode+'_identity_kl_gradient_exact_zero',bool((equal.grad==0).all()))
        # Derivative check against scalar finite differences of independent FP64 oracle.
        finite=np.zeros_like(z,dtype=np.float64);epsilon=1e-5
        for index in np.ndindex(z.shape):
            a=z.astype(np.float64);b=a.copy();a[index]+=epsilon;b[index]-=epsilon
            finite[index]=(oracle(a,t,m,y,v,mode)[0]-oracle(b,t,m,y,v,mode)[0])/(2*epsilon)
        check(mode+'_finite_difference',float(np.max(np.abs(gradient-finite)))<1e-9)
        outcomes[mode]=dict(loss=float(loss.detach()),bce=float(terms['bce']),kl=float(terms['kl']),
            manual_count=int(m.sum()),support_count=int(v.sum()),extra_count=int((v&~m).sum()),
            gradient=student.grad.numpy().tolist(),max_oracle_error=float(np.max(np.abs(student.grad.numpy()-gradient))))
    # Uniform support mean is a count-weighted mixture, not the old manual mean plus new support.
    nm=int(m.sum());nv=int(v.sum());nu=nv-nm
    s=torch.tensor(z,requires_grad=True);teacher=torch.tensor(t)
    _,terms=objective(s,teacher,torch.from_numpy(m),torch.from_numpy(y),torch.from_numpy(v),'support_mean')
    check('uniform_support_mixture_identity',abs(float(terms['kl'])-(nm/nv*float(terms['manual_kl'])+nu/nv*float(terms['extra_kl'])))<1e-15)
    manual_preservation=(sigmoid(z.astype(float))-sigmoid(t.astype(float)))[m]/nm
    expanded_preservation=(sigmoid(z.astype(float))-sigmoid(t.astype(float)))[m]/nv
    check('manual_kl_derivative_dilution',np.allclose(expanded_preservation/manual_preservation,nm/nv,rtol=0,atol=1e-15))
    # Equal per-batch aggregation differs from pooling unequal manual participant counts.
    batch_means=[1.,3.];counts=[1,3]
    check('equal_batch_mean_not_pooled_pixel_mean',math.fsum(batch_means)/2==2. and sum(a*b for a,b in zip(batch_means,counts))/sum(counts)==2.5)
    for name,bad_m,bad_v,bad_mode in [('empty_manual',np.zeros_like(m),v,'manual_mean'),
            ('manual_outside_support',m,np.zeros_like(v),'support_mean'),('empty_extra',m,m,'stratified_mean')]:
        try:objective(torch.tensor(z),torch.tensor(t),torch.from_numpy(bad_m),torch.from_numpy(y),torch.from_numpy(bad_v),bad_mode)
        except ValueError:check(name+'_rejected',True)
        else:raise ValueError(name+' not rejected')
    output=dict(status='PASS',check_count=len(checks),checks=checks,outcomes=outcomes,
        fixture_inputs=dict(student=z.tolist(),teacher=t.tolist(),manual=m.tolist(),support=v.tolist(),manual_targets=[0,1,1]),
        uniform_support_manual_kl_scale_ratio=nm/nv,fixture_kl_weight=1.,fixture_stratum_weights=[.5,.5],
        reductions='BCE is one mean over all manual participants of the batch; KL each defined reduction; fixed diagnostic risk is equal mean over batches.',
        numerical_tolerances=dict(loss_absolute=2e-7,gradient_absolute=2e-8,finite_difference_absolute=1e-9),
        sources=[record(Path(__file__)),record(Path(__file__).with_name('preservation_objectives.py'))],
        scientific_array_reads=0,model_calls=0,optimizer_calls=0,network_calls=0,checkpoint_loads=0,
        authorization='Synthetic mechanics only; no scientific support validation or study-design choice.')
    with args.output.open('x') as f:json.dump(output,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PASS',checks=len(checks),manual_KL_scale_ratio=nm/nv,output=record(args.output))))

if __name__=='__main__':run()
