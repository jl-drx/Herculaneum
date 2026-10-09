"""Finite allocation-only toy fixtures; never load scientific inputs or seed42."""
import argparse
import ast
import hashlib
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn
from depth_common import require,record,write_json,state_hash,isolation_state
from depth_adapter import ARMS,manual_terms
from fusion_adapter import FrozenFusionModel,matched_adapter,depth_features
from run_experiment import strict_reload
from depth_data import load_support_array


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    root=Path(__file__).resolve().parent;checks=[]
    def check(name,value):require(value,name);checks.append(name)
    for source in root.glob('*.py'):ast.parse(source.read_text())
    check('sources_parse',True)
    check('unchanged_phase19_architecture',record(root/'fusion_adapter.py')['sha256']=='a7004b8d06f163b8a20bfad87cd00dfb3d015d36c6be0833362255ee68a53d24')
    with tempfile.TemporaryDirectory(prefix='allocation-masks-',dir=root) as scratch:
        path=Path(scratch)/'mask.npy';values=np.array([[True,False],[False,True]],np.bool_);np.save(path,values,allow_pickle=False)
        descriptor=dict(**record(path),shape=[2,2],dtype='bool',raw_sha256=hashlib.sha256(values.tobytes()).hexdigest())
        check('extended_support_descriptor_loads',np.array_equal(load_support_array(descriptor,(2,2),np.bool_),values))
        bad=dict(descriptor,raw_sha256='0'*64)
        try:load_support_array(bad,(2,2),np.bool_)
        except ValueError:check('corrupt_support_raw_hash_rejected',True)
        else:raise ValueError('Corrupt raw hash accepted')
    class Body(nn.Module):
        def __init__(self):super().__init__();self.layer=nn.Conv2d(32,1,1)
        def forward(self,x):return {'ink':self.layer(x)}
    class Fusion(nn.Module):
        def __init__(self):super().__init__();self.layer=nn.Conv2d(1,32,1)
        def forward(self,x):return self.layer(x.mean(2))
    class Model(nn.Module):
        def __init__(self):super().__init__();self.depth_fusion=Fusion();self.network=Body()
        def forward(self,x):return self.network(self.depth_fusion(x))
    torch.manual_seed(20261008);model=Model().eval().requires_grad_(False)
    original=state_hash(model.state_dict());image=torch.randn(2,1,17,8,8)
    manual=torch.zeros(2,1,8,8,dtype=torch.bool);manual[:,:,:3]=True
    retired=torch.zeros_like(manual);retired[:,:,3:6]=True
    target=torch.full(manual.shape,float('nan'));target[manual]=(torch.arange(int(manual.sum()))%2).float()
    with torch.no_grad():teacher=model(image)['ink']
    initial_states=[];representations=[];first_updates=[]
    for arm in ARMS:
        before_rng=torch.random.get_rng_state().clone();adapter=matched_adapter(20261008)
        check(arm+'_forked_rng_unchanged',torch.equal(before_rng,torch.random.get_rng_state()))
        initial_states.append(state_hash(adapter.state_dict()))
        wrapper=FrozenFusionModel(model,adapter,'explicit_depth').train()
        representation=depth_features(image,'explicit_depth');representations.append(hashlib.sha256(representation.numpy().tobytes()).hexdigest())
        fused=wrapper.frozen_features(image);z,residual,adapted=wrapper.forward_from_fused(image,fused)
        check(arm+'_initial_original_identity',torch.equal(z,teacher) and torch.equal(adapted,fused) and not bool(residual.any()))
        z.retain_grad();adapted.retain_grad();loss,terms=manual_terms(z,teacher,manual,target,retired,arm)
        loss.backward()
        objective_support=manual if arm=='manual_double' else manual|retired
        check(arm+'_output_gradient_support',bool((z.grad[~objective_support]==0).all()))
        check(arm+'_decoder_input_gradients',float(adapted.grad.abs().sum())>0)
        check(arm+'_hidden_initial_gradient_zero',all(not bool(p.grad.any()) for p in adapter.stem.parameters()))
        check(arm+'_projection_initial_gradient_nonzero',sum(float(p.grad.square().sum()) for p in adapter.projection.parameters())>0)
        check(arm+'_separate_initial_KL_zero',terms['kl_manual']==terms['kl_retired']==0)
        check(arm+'_initial_shared_objectives_equal',terms['objective_A']==terms['objective_B']==terms['full']==terms['base'])
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False)
        torch.nn.utils.clip_grad_norm_(list(adapter.parameters()),1.,foreach=False,error_if_nonfinite=True);optimizer.step()
        first_updates.append(np.concatenate([p.detach().numpy().ravel() for p in adapter.parameters()]))
        optimizer.zero_grad(set_to_none=True)
        second,_,_=wrapper.forward_from_fused(image,fused);second.retain_grad()
        loss2,terms2=manual_terms(second,teacher,manual,target,retired,arm);loss2.backward()
        check(arm+'_second_hidden_gradient_positive',sum(float(p.grad.square().sum()) for p in adapter.stem.parameters())>0)
        sigmoid=lambda x:1/(1+np.exp(-x))
        a=second.detach().numpy().astype(np.float64);t=teacher.numpy().astype(np.float64);m=manual.numpy();v=retired.numpy()
        q=.25+.5*target.numpy()[m];expected=np.zeros(a.shape)
        expected[m]=(sigmoid(a[m])-q)/m.sum()
        expected[m]+=(2. if arm=='manual_double' else 1.)*(sigmoid(a[m])-sigmoid(t[m]))/m.sum()
        if arm=='manual_retired':expected[v]+=(sigmoid(a[v])-sigmoid(t[v]))/v.sum()
        check(arm+'_independent_analytic_gradient',np.max(np.abs(second.grad.numpy()-expected))<2e-8)
        km=float(np.mean(np.logaddexp(0,a[m])-np.logaddexp(0,t[m])-sigmoid(t[m])*(a[m]-t[m])))
        kv=float(np.mean(np.logaddexp(0,a[v])-np.logaddexp(0,t[v])-sigmoid(t[v])*(a[v]-t[v])))
        check(arm+'_independent_separate_KL',abs(terms2['kl_manual']-km)<1e-12 and abs(terms2['kl_retired']-kv)<1e-12)
        check(arm+'_own_full_selection',terms2['full']==terms2['objective_A' if arm=='manual_double' else 'objective_B'])
        changed_target=target.clone();changed_target[~manual]=12345
        changed=manual_terms(second.detach(),teacher,manual,changed_target,retired,arm)[1]
        check(arm+'_retired_targets_ignored',changed==terms2)
        adapter.zero_grad(set_to_none=True);before=isolation_state(adapter,optimizer)
        with torch.no_grad():probe=wrapper.forward_from_fused(image,fused);manual_terms(probe[0],teacher,manual,target,retired,arm)
        check(arm+'_measurement_isolated',before==isolation_state(adapter,optimizer))
        check(arm+'_backbone_unchanged',state_hash(model.state_dict())==original and all(p.grad is None and not p.requires_grad for p in model.parameters()) and not any(part.training for part in model.modules()))
        with tempfile.TemporaryDirectory(prefix='allocation-toy-',dir=root) as scratch:
            path=Path(scratch)/'adapter.pth';torch.save(adapter.state_dict(),path)
            check(arm+'_strict_compact_adapter_reload',strict_reload(adapter,path,state_hash(adapter.state_dict()))['parameter_count']==5296)
        for label,m2,v2 in [('empty_M',torch.zeros_like(manual),retired),('empty_V',manual,torch.zeros_like(retired)),('overlap',manual,retired|manual)]:
            try:manual_terms(second.detach(),teacher,m2,target,v2,arm)
            except ValueError:check(arm+'_'+label+'_rejected',True)
            else:raise ValueError('Invalid role support accepted')
    check('matched_initial_parameters',initial_states[0]==initial_states[1])
    check('matched_explicit_depth_features',representations[0]==representations[1])
    difference=first_updates[1].astype(float)-first_updates[0].astype(float)
    write_json(args.output,dict(status='PASS',checks=checks,check_count=len(checks),
        first_step_difference_diagnostic=dict(max_absolute=float(np.abs(difference).max()),l2=float(np.sqrt(np.square(difference).sum())),no_acceptance_threshold=True),
        source_code={p.name:record(p) for p in sorted(root.glob('*.py'))},
        scientific_array_reads=0,pretrained_checkpoint_loads=0,network_calls=0,gpu_calls=0,
        actual_model='tiny synthetic interface fixture only; never seed42',
        limits='Two synthetic updates, two backwards per arm; temporary tiny adapter checkpoints removed. No accuracy/timing claim.'))
    print({'status':'PASS','checks':len(checks)})

if __name__=='__main__':main()
