"""One fixed offline native-model inference; exact retained seed42 inputs required."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
from common import (HERE,PHASE,FORBIDDEN,require,pin,verify,read_json,write_json,module,
                    qualified_helpers,load_array,load_plan,verify_runtime_sources,check_coverage,encode)
import numpy as np
import torch
import safetensors
from safetensors.torch import load_file


def forbidden_torch_load(*args,**kwargs):
    FORBIDDEN['torch_load']+=1
    raise RuntimeError('torch.load forbidden in phase23')


torch.load=forbidden_torch_load


class Capture(torch.nn.Module):
    def __init__(self,model,expected):
        super().__init__();self.model=model;self.expected=expected
        self.logits=[];self.probabilities=[];self.captured=[]
    def forward(self,image):
        require(not torch.is_grad_enabled(),'Autograd enabled')
        index=len(self.logits)
        require(index<len(self.expected) and image.dtype==torch.float32
            and image.device.type=='cpu' and tuple(image.shape)==(1,1,17,128,128),'Input shape/count/type differs')
        require(bool(torch.isfinite(image).all()),'Nonfinite normalized input')
        digest=hashlib.sha256(image.numpy().tobytes(order='C')).hexdigest()
        require(digest==self.expected[index]['input_sha256'],'Normalized input differs BEFORE model forward')
        outputs=self.model(image)
        require(isinstance(outputs,dict) and set(outputs)=={'ink'},'Model output mapping differs')
        logits=outputs['ink']
        if isinstance(logits,(tuple,list)):
            require(len(logits)==1,'Unexpected deep supervision');logits=logits[0]
        require(logits.dtype==torch.float32 and tuple(logits.shape)==(1,1,128,128)
            and bool(torch.isfinite(logits).all()),'Output logits differ')
        require(hashlib.sha256(image.numpy().tobytes(order='C')).hexdigest()==digest,'Model mutated input')
        self.logits.append(logits.numpy()[0,0].copy())
        self.probabilities.append(logits.float().sigmoid().numpy()[0,0].copy())
        self.captured.append(dict(bbox_roi_xywh=self.expected[index]['bbox_roi_xywh'],input_sha256=digest))
        return outputs


def reconstruct(probabilities,boxes,weights,shape=(384,384)):
    sums=np.zeros(shape,np.float32);denominator=np.zeros(shape,np.float32)
    require(len(probabilities)==len(boxes),'Reconstruction tile count differs')
    for prob,box in zip(probabilities,boxes):
        x,y,w,h=box
        require(prob.shape==weights.shape==(h,w),'Reconstruction tile shape differs')
        sums[y:y+h,x:x+w]+=prob*weights
        denominator[y:y+h,x:x+w]+=weights
    raw=np.zeros_like(sums);np.divide(sums,denominator,out=raw,where=denominator>0)
    return sums,denominator,raw


def save_array(path,array):
    require(not Path(path).exists(),'Refuse overwrite')
    np.save(path,array,allow_pickle=False)
    return {**pin(path),'shape':list(array.shape),'dtype':array.dtype.str,
        'raw_c_order_sha256':hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def worker(args):
    started=time.monotonic();plan,contract=load_plan(args.plan,args.release);limits=plan['limits']
    signal.alarm(limits['worker_wall_seconds'])
    resource.setrlimit(resource.RLIMIT_CPU,(limits['worker_cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_AS,(limits['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(limits['phase_output_bytes'],)*2)
    require(dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,safetensors=safetensors.__version__)==plan['runtime'],'Runtime differs')
    require(not torch.cuda.is_available(),'Unexpected CUDA runtime')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(plan['seed'])
    directory=Path(args.output)/'native';directory.mkdir(exist_ok=False)
    factory,state_receipt=qualified_helpers(plan)
    phase22=read_json(plan['sources']['phase22_plan']);verify_runtime_sources(plan)
    model,normalizer,unused_baseline,config=factory.build_factory('current',phase22['sources'])
    require(Path(normalizer.__code__.co_filename).resolve()==verify(plan['sources']['normalization']),
        'Actual normalization source differs')
    schema=read_json(phase22['sources']['reference_schema'])
    require(schema['expected_named_tensor_count']==508 and schema['retained_factory_unique_storage_groups']==264
        and schema['retained_factory_unique_parameters']==34546498 and len(schema['expected_factory_alias_groups'])==244,'Qualified schema changed')
    checkpoint=phase22['stage1']['artifact'];state=load_file(str(verify(checkpoint)),device='cpu')
    expected={item['name']:item for item in schema['tensor_schema']}
    require(set(state)==set(expected) and len(state)==508,'Native key set differs')
    for name,value in state.items():
        require(value.dtype==torch.float32 and list(value.shape)==expected[name]['shape'],'Native dtype/shape differs before strict load')
    loaded=model.load_state_dict(state,strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys,'Strict load incompatibility')
    initial=state_receipt(model,state,schema)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    require(all(not m.training for m in model.modules()) and all(p.grad is None for p in model.parameters()),'Initial eval/gradient state differs')
    write_json(directory/'load-verification.json',dict(status='PASS',strict=True,missing_keys=[],unexpected_keys=[],
        checkpoint=checkpoint,config=config,model_class=type(model).__module__+'.'+type(model).__qualname__,**initial))
    reference=module('phase23_retained_inference',plan['sources']['inference_adapter'])
    weights=reference.hann_blend_map();runs={};counter=[0]
    def full_forward(module,args,output): counter[0]+=1
    hook=model.register_forward_hook(full_forward)
    for case in contract['cases']:
        target=directory/case['id'];target.mkdir()
        surface=load_array(case['arrays']['surface']);primary=load_array(case['arrays']['primary_mask'])
        require(surface.shape==(17,384,384) and surface.dtype==np.float32 and primary.shape==(384,384),'Audited input shape differs')
        require(case['selected_surface_planes']==list(range(17)) and case['layer_start']==0 and case['direction']=='forward','Audited depth differs')
        expected_tiles=case['tile_schedule_and_captured_inputs']
        boxes=[row['bbox_roi_xywh'] for row in expected_tiles]
        require(boxes==[[x,y,128,128] for y in range(0,257,64) for x in range(0,257,64)],'Audited tile order differs')
        capture=Capture(model,expected_tiles).eval()
        raw,inference,evaluated=reference.infer_surface(capture,surface,0,'forward',normalizer,None)
        require(inference['tile_count']==inference['scored_tile_count']==len(capture.logits)==25,'Inference tile count differs')
        require([t['bbox_roi_xywh'] for t in inference['tiles']]==boxes and capture.captured==expected_tiles,'Inference order/input hashes differ')
        require(raw.dtype==np.float32 and evaluated.dtype==np.bool_ and np.isfinite(raw).all()
            and ((raw>=0)&(raw<=1)).all(),'Raw canvas values invalid')
        check_coverage(raw,evaluated,primary)
        logits=np.stack(capture.logits);probabilities=np.stack(capture.probabilities)
        sums,denominator,reconstructed=reconstruct(probabilities,boxes,weights)
        require(reconstructed.tobytes()==raw.tobytes() and np.array_equal(evaluated,denominator>0),'Recorded tile reconstruction differs')
        byte,decoded=encode(raw)
        arrays={}
        for key,value in [('tile_logits',logits),('tile_probabilities',probabilities),('raw_scores',raw),
            ('probability_sum',sums),('weight_sum',denominator),('hann',weights),('evaluated_mask',evaluated),
            ('uint8',byte),('decoded',decoded)]:
            arrays[key]=save_array(target/(key.replace('_','-')+'.npy'),value)
        item=dict(status='PASS',id=case['id'],arrays=arrays,inference=inference,captured_tiles=capture.captured,
            input_contract=plan['sources']['input_contract'],surface=case['arrays']['surface'],
            primary_mask=case['arrays']['primary_mask'],primary_pixels=int(primary.sum()),
            primary_evaluated_pixels=int(evaluated[primary].sum()),input_fingerprints_exact_before_forward=True,
            surface_unchanged=True,role='reused_development_diagnostic')
        verify(case['arrays']['surface']);verify(case['arrays']['primary_mask'])
        write_json(target/'metadata.json',item);runs[case['id']]=item
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=limits['rss_bytes'],'RSS exceeded')
        require(sum(p.stat().st_size for p in PHASE.rglob('*') if p.is_file())<=limits['phase_output_bytes'],'Phase output cap exceeded')
        del surface,primary,raw,evaluated,capture,logits,probabilities,sums,denominator,reconstructed;gc.collect()
    hook.remove()
    require(counter[0]==150,'Exactly150 full-model calls required')
    require(all(not m.training for m in model.modules()) and all(p.grad is None for p in model.parameters()),'Final eval/gradient state differs')
    final=state_receipt(model,state,schema);require(final==initial,'Native state changed')
    write_json(directory/'final-state-verification.json',dict(status='PASS',**final))
    verify_runtime_sources(plan);require(not any(FORBIDDEN.values()),'Forbidden action attempted')
    load_plan(args.plan,args.release)
    result=dict(status='PASS',plan=pin(args.plan),input_contract=plan['sources']['input_contract'],runs=runs,
        load_verification=pin(directory/'load-verification.json'),final_state_verification=pin(directory/'final-state-verification.json'),
        model_constructions=1,strict_loads=1,full_model_calls=counter[0],training_steps=0,gradients=0,
        source_hashes_unchanged=True,state_unchanged=True,forbidden_calls=FORBIDDEN,
        elapsed_seconds=time.monotonic()-started,
        cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(directory/'result.json',result)
    print(json.dumps(dict(status='PASS',result=pin(directory/'result.json'))))


def controller(args):
    started=time.monotonic();plan,contract=load_plan(args.plan,args.release)
    out=Path(args.output);out.mkdir(exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--plan',str(Path(args.plan).resolve()),
        '--release',str(Path(args.release).resolve()),'--output',str(out.resolve()),'--worker']
    with (out/'native.log').open('xb') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,
            'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
        stop=None;peak=0
        while child.poll() is None:
            elapsed=time.monotonic()-started
            if elapsed>plan['limits']['external_wall_seconds']: stop='external_wall'
            if sum(p.stat().st_size for p in PHASE.rglob('*') if p.is_file())>plan['limits']['phase_output_bytes']: stop='phase_output_bytes'
            try:
                text=Path('/proc/%d/status'%child.pid).read_text()
                rss=next(int(line.split()[1])*1024 for line in text.splitlines() if line.startswith('VmRSS:'))
                peak=max(peak,rss)
                if rss>plan['limits']['rss_bytes']: stop='worker_rss'
            except (FileNotFoundError,StopIteration): pass
            if stop:
                child.kill();child.wait();break
            time.sleep(.1)
    result=dict(status='PASS' if not stop and child.returncode==0 else 'STOPPED',stop=stop,returncode=child.returncode,
        elapsed_seconds=time.monotonic()-started,external_peak_rss_bytes=peak,no_retry=True,plan=pin(args.plan))
    if result['status']=='PASS': result['result']=pin(out/'native/result.json')
    write_json(out/'controller-result.json',result)
    require(result['status']=='PASS','Native scientific execution stopped; no retry')
    print(json.dumps(result))


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--release',required=True)
    p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');args=p.parse_args()
    (worker if args.worker else controller)(args)


if __name__=='__main__':main()
