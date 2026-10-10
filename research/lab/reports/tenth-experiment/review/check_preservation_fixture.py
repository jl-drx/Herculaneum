"""Independent tiny-logit KL checks. No model, checkpoint, or network use."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket

HERE = Path(__file__).resolve().parent
LOSS = HERE.parent / 'implementation/preservation.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    e = math.exp(value)
    return e / (1 + e)


def softplus(value):
    return max(value, 0) + math.log1p(math.exp(-abs(value)))


def independent(z, t, mask):
    """Probability-weighted log-probability differences, scalar Python."""
    count = sum(mask)
    values, gradient = [], []
    for a, b, participate in zip(z, t, mask):
        q = sigmoid(b)
        # Log probabilities stay finite at extreme logits; no log(0).
        d = q * (softplus(-a) - softplus(-b)) + (1-q) * (softplus(a) - softplus(b))
        if participate:
            values.append(d)
        gradient.append((sigmoid(a)-q)/count if participate else 0.)
    return math.fsum(values)/count, gradient


def fixture():
    before = [record(__file__), record(LOSS)]
    spec = importlib.util.spec_from_file_location('tenth_fixture_loss', LOSS)
    loss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loss)
    import torch
    torch.set_num_threads(4)
    shape = (2, 1, 2, 4)
    manual = [1,1,0,0,1,0,0,0, 1,0,1,1,0,1,0,0]
    pseudo = [0,0,1,0,0,1,1,0, 0,1,0,0,1,0,0,0]
    m = torch.tensor(manual, dtype=torch.bool).reshape(shape)
    p = torch.tensor(pseudo, dtype=torch.bool).reshape(shape)
    hard = [0,1,0,0,1,1,0,0, 1,1,0,1,0,0,0,0]
    cases = {
        'mixed': ([-100.,-3.,-.5,0.,.25,2.,100.,-1., -100.,-2.,-.25,.5,1.,3.,100.,-4.],
                  [100.,-1.,.3,2.,-.5,1.,-100.,0., -80.,2.,.25,-.5,-1.,-3.,0.,-4.]),
        'near_equal': ([x+1e-7 for x in [-20.,-3.,-.5,0.,.25,2.,20.,-1., -10.,-2.,-.25,.5,1.,3.,10.,-4.]],
                       [-20.,-3.,-.5,0.,.25,2.,20.,-1., -10.,-2.,-.25,.5,1.,3.,10.,-4.]),
        'equal': ([-1000.,-21.,-.5,0.,.25,21.,1000.,-1., -100.,-2.,-.25,.5,1.,3.,100.,-4.],)*2,
    }
    checks = []
    for dtype in (torch.float64, torch.float32):
        for case, (student, teacher) in cases.items():
            z = torch.tensor(student, dtype=dtype).reshape(shape).requires_grad_()
            t = torch.tensor(teacher, dtype=dtype).reshape(shape).requires_grad_()
            z_values, t_values = z.detach().reshape(-1).tolist(), t.detach().reshape(-1).tolist()
            dm, dp = loss.preservation_losses(z, t, m, p)
            require(dm.dtype == dp.dtype == torch.float64, 'KL must accumulate in float64')
            expected = {'D_M': independent(z_values, t_values, manual),
                        'D_P': independent(z_values, t_values, pseudo)}
            terms = {'D_M': dm, 'D_P': dp, 'A_preservation': dm,
                     'B_preservation': dm + .25*dp}
            expected['A_preservation'] = expected['D_M']
            expected['B_preservation'] = (
                expected['D_M'][0]+.25*expected['D_P'][0],
                [a+.25*b for a,b in zip(expected['D_M'][1],expected['D_P'][1])])
            for arm in ('A','B'):
                y = torch.tensor(hard, dtype=dtype).reshape(shape)*.5+.25
                u = torch.nn.functional.binary_cross_entropy_with_logits(z[m],y[m])
                ps = torch.nn.functional.binary_cross_entropy_with_logits(z[p],y[p])
                terms[arm+'_full'] = u+dm if arm=='A' else u+.25*ps+dm+.25*dp
                value, gradient = expected[arm+'_preservation']
                value = float(value); gradient = list(gradient)
                for support, weight in ((manual,1.),(pseudo,0. if arm=='A' else .25)):
                    n = sum(support)
                    value += weight*math.fsum(softplus(a)-(.25+.5*yh)*a
                                             for a,yh,used in zip(z_values,hard,support) if used)/n
                    for i,used in enumerate(support):
                        if used:
                            gradient[i] += weight*(sigmoid(z_values[i])-(.25+.5*hard[i]))/n
                expected[arm+'_full'] = value,gradient
            for name, term in terms.items():
                dz,dt = torch.autograd.grad(term,(z,t),retain_graph=True,allow_unused=True)
                value,gradient = expected[name]
                e_value = abs(term.item()-value)
                e_gradient = max(abs(a-b) for a,b in zip(dz.reshape(-1).tolist(),gradient))
                value_tol = 5e-5 if dtype==torch.float32 and name.endswith('_full') else 2e-12
                grad_tol = 5e-8 if dtype==torch.float32 else 2e-12
                require(e_value<=value_tol and e_gradient<=grad_tol,'Independent value/gradient differs: '+case+'/'+name)
                require(dt is None and t.grad is None,'Teacher acquired a gradient')
                require(bool((dz[~(m|p)]==0).all()),'Unknown outputs acquired gradients')
                if name.startswith('A') or name=='D_M':
                    require(bool((dz[p]==0).all()),'Manual-only term used pseudo outputs')
                if case=='equal' and not name.endswith('_full'):
                    require(term.item()==0. and float(dz.abs().max())<=1e-12,'Equality sentinel differs')
                checks.append(dict(case=case,dtype=str(dtype),term=name,value=term.item(),
                                   independent_value=value,value_error=e_value,gradient_max_error=e_gradient,
                                   teacher_gradient_absent=True,unknown_gradient_exactly_zero=True,
                                   value_tolerance=value_tol,gradient_tolerance=grad_tol))
    z=torch.zeros(shape);t=torch.ones(shape)
    bad=[]
    for name,args in [
        ('shape',(z,t[..., :3],m,p)),('mask_dtype',(z,t,m.float(),p)),
        ('overlap',(z,t,m,m|p)),('empty_manual',(z,t,torch.zeros_like(m),p)),
        ('empty_pseudo',(z,t,m,torch.zeros_like(p))),
        ('nonfinite_student',(torch.full_like(z,float('nan')),t,m,p)),
        ('nonfinite_teacher',(z,torch.full_like(t,float('inf')),m,p))]:
        try: loss.preservation_losses(*args)
        except ValueError: bad.append(dict(case=name,rejected=True))
        else: raise ValueError('Invalid fixture accepted: '+name)
    after=[record(__file__),record(LOSS)]
    require(before==after,'Fixture source changed')
    return dict(status='PASS',source_receipts_before=before,source_receipts_after=after,
                checks=checks,invalid_input_guards=bad,formula='q*(log(q)-log(p))+(1-q)*(log(1-q)-log(1-p)), using finite log-sigmoid',
                derivative='(sigmoid(student)-sigmoid(teacher))/participating_count',
                theta_teacher_detached=True,torch_version=torch.__version__,
                model_constructions=0,model_forward_calls=0,checkpoint_loads=0,optimizer_steps=0,network_requests=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True,type=Path);args=parser.parse_args()
    require(args.receipt.resolve().parent==HERE and not args.receipt.exists(),'Fresh receipt in assigned review directory required')
    def blocked(*a,**k): raise RuntimeError('Fixture network access prohibited')
    socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked;socket.getaddrinfo=blocked;socket.socket.sendto=blocked
    receipt=fixture();args.receipt.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',receipt=record(args.receipt),checks=len(receipt['checks']),guards=len(receipt['invalid_input_guards']),
                         max_value_error=max(x['value_error'] for x in receipt['checks']),max_gradient_error=max(x['gradient_max_error'] for x in receipt['checks']))))
