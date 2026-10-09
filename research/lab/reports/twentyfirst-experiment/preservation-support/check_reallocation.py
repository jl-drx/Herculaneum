"""Finite independent-formula fixtures for exact root-proposed group reallocation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from reallocation_objective import objective


def require(ok,message):
    if not ok:raise ValueError(message)


def record(path):
    p=Path(path);return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)


def sigmoid(x):return 1/(1+np.exp(-x))


def oracle(z,t,m,v,y,arm):
    z=z.astype(float);t=t.astype(float);nm=int(m.sum());nv=int(v.sum());soft=.25+.5*y[m]
    base=np.mean(np.logaddexp(0,z[m])-soft*z[m]);kl=np.logaddexp(0,z)-np.logaddexp(0,t)-sigmoid(t)*(z-t)
    a,b=(2.,0.) if arm=='manual_twice' else (1.,1.)
    weights=np.zeros(z.shape);weights[m]=a/nm;weights[v]=b/nv
    gradient=weights*(sigmoid(z)-sigmoid(t));gradient[m]+=(sigmoid(z[m])-soft)/nm
    return float(base+a*kl[m].mean()+b*kl[v].mean()),gradient,dict(base=float(base),manual_kl=float(kl[m].mean()),retired_validation_kl=float(kl[v].mean()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);checks=[];receipts={}
    def check(name,value):require(value,name);checks.append(name)
    z=np.linspace(-1.1,1.4,12,dtype=np.float32).reshape(2,1,2,3)
    t=np.linspace(.8,-.7,12,dtype=np.float32).reshape(z.shape)
    m=np.array([1,0,0,0,1,0,1,0,0,0,0,0],bool).reshape(z.shape)
    v=np.array([0,1,1,0,0,0,0,1,0,1,0,1],bool).reshape(z.shape)
    y=np.full(z.shape,np.nan,np.float32);y[m]=[0,1,1]
    for arm in ['manual_twice','manual_plus_retired_validation']:
        student=torch.tensor(z,requires_grad=True);teacher=torch.tensor(t,requires_grad=True)
        loss,terms=objective(student,teacher,torch.from_numpy(m),torch.from_numpy(v),torch.from_numpy(y),arm)
        loss.backward();wanted,gradient,diagnostics=oracle(z,t,m,v,y,arm)
        check(arm+'_loss_matches_numpy',abs(float(loss.detach())-wanted)<2e-7)
        check(arm+'_output_derivative_matches_numpy',np.max(np.abs(student.grad.numpy()-gradient))<2e-8)
        check(arm+'_manual_BCE_unchanged',abs(float(terms['base'].detach())-diagnostics['base'])<2e-7)
        check(arm+'_teacher_detached',teacher.grad is None)
        check(arm+'_coefficient_sum_two',terms['manual_kl_coefficient']+terms['retired_validation_kl_coefficient']==2.)
        check(arm+'_outside_roles_gradient_zero',bool((student.grad[torch.from_numpy(~(m|v))]==0).all()))
        if arm=='manual_twice':check(arm+'_retired_gradient_zero',bool((student.grad[torch.from_numpy(v)]==0).all()))
        else:check(arm+'_retired_gradient_nonzero',bool((student.grad[torch.from_numpy(v)]!=0).all()))
        changed_target=y.copy();changed_target[~m]=12345
        changed,_=objective(student.detach(),teacher,torch.from_numpy(m),torch.from_numpy(v),torch.from_numpy(changed_target),arm)
        check(arm+'_retired_and_unknown_targets_unread',float(changed)==float(loss.detach()))
        changed_z=z.copy();changed_z[~(m|v)]=np.nan
        changed,_=objective(torch.tensor(changed_z),teacher,torch.from_numpy(m),torch.from_numpy(v),torch.from_numpy(y),arm)
        check(arm+'_unsupported_logits_unread',float(changed)==float(loss.detach()))
        equal=torch.tensor(t,requires_grad=True)
        _,identity=objective(equal,teacher,torch.from_numpy(m),torch.from_numpy(v),torch.from_numpy(y),arm)
        preservation=identity['manual_kl_coefficient']*identity['manual_kl']+identity['retired_validation_kl_coefficient']*identity['retired_validation_kl']
        preservation.backward()
        check(arm+'_identity_KL_zero',float(preservation.detach())==0.)
        check(arm+'_identity_KL_derivative_roundoff',float(equal.grad.abs().max())<=1e-15)
        finite=np.zeros(z.shape,dtype=float);epsilon=1e-5
        for index in np.ndindex(z.shape):
            plus=z.astype(float);minus=plus.copy();plus[index]+=epsilon;minus[index]-=epsilon
            finite[index]=(oracle(plus,t,m,v,y,arm)[0]-oracle(minus,t,m,v,y,arm)[0])/(2*epsilon)
        check(arm+'_scalar_finite_difference',np.max(np.abs(finite-gradient))<1e-9)
        receipts[arm]=dict(loss=float(loss.detach()),diagnostics=diagnostics,
            gradient=student.grad.numpy().tolist(),manual_count=int(m.sum()),retired_count=int(v.sum()),
            identity_KL_gradient_max=float(equal.grad.abs().max()))
    nm=int(m.sum());nv=int(v.sum());delta=sigmoid(z.astype(float))-sigmoid(t.astype(float))
    check('candidate_manual_KL_derivative_is_half_control',np.allclose((delta[m]/nm)/(2*delta[m]/nm),.5,atol=0,rtol=0))
    manual_kl=receipts['manual_twice']['diagnostics']['manual_kl'];retired_kl=receipts['manual_twice']['diagnostics']['retired_validation_kl']
    check('candidate_is_not_double_union_mean',abs((manual_kl+retired_kl)-2*(nm*manual_kl+nv*retired_kl)/(nm+nv))>1e-5)
    for name,bad_m,bad_v in [('empty_manual',np.zeros_like(m),v),('empty_retired',m,np.zeros_like(v)),('overlap',m,v|m)]:
        for arm in ['manual_twice','manual_plus_retired_validation']:
            try:objective(torch.tensor(z),torch.tensor(t),torch.from_numpy(bad_m),torch.from_numpy(bad_v),torch.from_numpy(y),arm)
            except ValueError:check(name+'_'+arm+'_rejected',True)
            else:raise ValueError('Invalid role support accepted')
    out=dict(status='PASS',check_count=len(checks),checks=checks,receipts=receipts,
        exact_candidate_definitions=dict(manual_twice='BCE_M + 2*mean(KL_M)',manual_plus_retired_validation='BCE_M + mean(KL_M) + mean(KL_V)'),
        interpretation='One unit of preservation is reallocated from manual to explicitly retired validation membership; manual KL derivative halves. Sum of group coefficients remains 2; this is not constant manual strength or a uniform union mean.',
        fixture_inputs=dict(student=z.tolist(),teacher=t.tolist(),manual=m.tolist(),retired_validation=v.tolist(),manual_targets=[0,1,1]),
        numerical_tolerances=dict(loss_absolute=2e-7,output_gradient_absolute=2e-8,identity_gradient_absolute=1e-15,finite_difference_absolute=1e-9),
        sources=[record(Path(__file__)),record(Path(__file__).with_name('reallocation_objective.py'))],
        scientific_array_reads=0,model_calls=0,optimizer_calls=0,checkpoint_loads=0,network_calls=0,
        no_scientific_protocol_frozen_by_this_fixture=True,no_geometry_validity_claim=True)
    with args.output.open('x') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PASS',checks=len(checks),output=record(args.output))))

if __name__=='__main__':main()
