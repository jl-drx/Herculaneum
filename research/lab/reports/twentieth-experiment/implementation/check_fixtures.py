"""Finite synthetic mechanism fixtures only: no checkpoints/scans/annotations read."""
import argparse
import ast
import hashlib
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn
from depth_common import require,record,write_json,state_hash,isolation_state
from depth_adapter import FusionAdapter,FrozenFusionModel,features,manual_terms
from frozen_backbone import tensor_receipt,bitwise_equal
from run_experiment import strict_reload


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.manual_seed(20261008)
    checks=[]
    def check(name,value):require(value,name);checks.append(name)
    root=Path(__file__).resolve().parent
    for path in root.glob('*.py'):ast.parse(path.read_text())
    check('all_python_sources_parse',True)
    source=Path('/workspace/herculaneum-lab/reports/nineteenth-experiment/fusion-prototype/fusion_adapter.py')
    check('unchanged_phase19_fusion_source',record(root/'fusion_adapter.py')['sha256']==record(source)['sha256'])
    class Body(nn.Module):
        def __init__(self):super().__init__();self.layer=nn.Conv2d(32,1,1)
        def forward(self,x):return {'ink':self.layer(x)}
    class Fusion(nn.Module):
        def __init__(self):super().__init__();self.layer=nn.Conv2d(1,32,1)
        def forward(self,x):return self.layer(x.mean(2))
    class Model(nn.Module):
        def __init__(self):super().__init__();self.depth_fusion=Fusion();self.network=Body()
        def forward(self,x):return self.network(self.depth_fusion(x))
    model=Model().eval().requires_grad_(False);original=state_hash(model.state_dict())
    image=torch.randn(2,1,17,8,8);mask=torch.zeros(2,1,8,8,dtype=torch.bool);mask[:,:,:4]=True
    target=(torch.arange(128).reshape(2,1,8,8)%2).float()
    teacher=model(image)['ink'].detach();initials=[]
    for arm in ['raw_control','explicit_depth']:
        torch.manual_seed(20261008);adapter=FusionAdapter();initials.append(state_hash(adapter.state_dict()))
        wrapper=FrozenFusionModel(model,adapter,arm).train();fused=wrapper.frozen_features(image)
        z,residual,adapted=wrapper.forward_from_fused(image,fused)
        check(arm+'_capacity',sum(p.numel() for p in adapter.parameters())==5296)
        check(arm+'_exact_zero_identity',bitwise_equal(z,teacher) and bitwise_equal(adapted,fused) and not bool(residual.any()))
        z.retain_grad();adapted.retain_grad();loss,terms=manual_terms(z,teacher,mask,target)
        loss.backward()
        check(arm+'_manual_logit_support',bool((z.grad[~mask]==0).all()))
        check(arm+'_decoder_input_gradient',bool((adapted.grad!=0).any()))
        check(arm+'_first_hidden_gradient_zero',all(not bool(p.grad.any()) for p in adapter.stem.parameters()))
        check(arm+'_projection_gradient_positive',sum(float(p.grad.square().sum()) for p in adapter.projection.parameters())>0)
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False)
        norm=torch.nn.utils.clip_grad_norm_(list(adapter.parameters()),1.,foreach=False,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        second,_,_=wrapper.forward_from_fused(image,fused);manual_terms(second,teacher,mask,target)[0].backward()
        check(arm+'_second_hidden_gradient_positive',sum(float(p.grad.square().sum()) for p in adapter.stem.parameters())>0)
        check(arm+'_original_unchanged',state_hash(model.state_dict())==original and all(p.grad is None and not p.requires_grad for p in model.parameters()) and not any(m.training for m in model.modules()))
        adapter.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory(prefix='phase20-synthetic-',dir=root) as scratch:
            path=Path(scratch)/'adapter.pth';torch.save(adapter.state_dict(),path)
            check(arm+'_strict_compact_reload',strict_reload(adapter,path,state_hash(adapter.state_dict()))['parameter_count']==5296)
        before=isolation_state(adapter,optimizer)
        with torch.no_grad():probe=wrapper.forward_from_fused(image,fused)
        check(arm+'_measurement_isolation',before==isolation_state(adapter,optimizer))
        check(arm+'_feature_receipt_shape',tensor_receipt(probe[1])['shape']==[2,32,8,8])
        # The original tile inference wrapper enters inference_mode. The new
        # boundary must disable it for ordinary fusion caches, with no autograd.
        with torch.inference_mode():
            with torch.inference_mode(False),torch.no_grad():
                ordinary=wrapper.frozen_features(image)
                output=wrapper.forward_from_fused(image,ordinary)
                check(arm+'_nested_inference_boundary',not torch.is_inference(ordinary) and not output[0].requires_grad)
    check('matched_initial_states',initials[0]==initials[1])
    ramp=torch.arange(17,dtype=torch.float32).reshape(1,1,17,1,1).expand(1,1,17,3,3)
    depth=features(ramp,'explicit_depth');raw=features(ramp,'raw_control')
    check('full_endpoint_and_center_differences',bool((depth[:,1]==1).all()))
    check('nominal_depth_endpoints',float(depth[0,2,0,0,0])==-1 and float(depth[0,2,-1,0,0])==1)
    check('raw_endpoint_clamping',float(raw[0,0,0,0,0])==0 and float(raw[0,2,-1,0,0])==16)
    write_json(args.output,dict(status='PASS',checks=checks,check_count=len(checks),
        source_code={p.name:record(p) for p in sorted(root.glob('*.py'))},
        scientific_array_reads=0,pretrained_checkpoint_loads=0,network_calls=0,gpu_calls=0,
        actual_model='tiny synthetic interface fixture; not original seed42',
        limitations='No scientific learning, timing or efficacy claim. Temporary tiny synthetic checkpoints deleted.'))
    print({'status':'PASS','checks':len(checks)})


if __name__=='__main__':main()
