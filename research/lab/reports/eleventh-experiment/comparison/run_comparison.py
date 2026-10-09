"""Offline CPU inference for five retained states; capture unchanged tile logits."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time


def offline(*args,**kwargs): raise RuntimeError('Network disabled in reference comparison')
socket.create_connection=offline; socket.socket.connect=offline; socket.socket.connect_ex=offline

import numpy as np
from prepare_reference import ROLES,DIRECTIONS,load_plan,load_arrays,record,verify,write_json,module,require


def rss(): return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


class Capture:
    """Capture original outputs; never alter logits, inputs or the adapter."""
    def __init__(self,model): self.model=model; self.logits=[]; self.input_hashes=[]
    def __call__(self,tensor):
        import torch
        require(tensor.dtype==torch.float32 and tuple(tensor.shape)==(1,1,17,128,128),'Normalized model input differs')
        digest=hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
        output=self.model(tensor); logits=output['ink']
        if isinstance(logits,(list,tuple)): logits=logits[0]
        require(logits.dtype==torch.float32 and tuple(logits.shape)==(1,1,128,128)
            and bool(torch.isfinite(logits).all()),'Native tile logit contract differs')
        self.logits.append(logits.detach().cpu().numpy()[0,0].copy())
        require(hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()==digest,'Model mutated captured input')
        self.input_hashes.append(digest)
        return output


def worker(plan,path,role,directory):
    limits=plan['limits']; resource.setrlimit(resource.RLIMIT_AS,(limits['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(limits['worker_cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(200*1024**2,)*2)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Reference worker wall cap')))
    signal.alarm(limits['worker_wall_seconds']); started=time.perf_counter(); directory.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261008)
    shared=module(plan['shared'],'eleventh_state_hasher'); helper=module(plan['helper'],'eleventh_model_contract')
    runtime=helper.load_runtime(); initializer=plan['initializer']; model,modelmeta=helper.build_model(verify(initializer),initializer,runtime)
    require(shared.state_hash(model.state_dict())==plan['models']['frozen_seed42']['state_sha256'],'Initializer changed')
    state_record=plan['models'][role]['checkpoint']; storage=None
    if state_record is not None:
        state=torch.load(verify(state_record),map_location='cpu',weights_only=True); expected=model.state_dict()
        require(isinstance(state,dict) and set(state)==set(expected),'Saved state key contract differs')
        for name,tensor in state.items():
            require(isinstance(tensor,torch.Tensor) and tensor.layout==torch.strided and tensor.dtype==expected[name].dtype
                and tensor.shape==expected[name].shape and bool(torch.isfinite(tensor).all()),'Saved tensor contract differs: '+name)
        storage=helper.verify_storage_contract(state,expected,torch)
        require(shared.state_hash(state)==plan['models'][role]['state_sha256'],'Saved state semantic hash differs')
        strict=model.load_state_dict(state,strict=True)
        require(not strict.missing_keys and not strict.unexpected_keys,'Saved state strict load differs')
        del state,expected; gc.collect()
    model.eval(); require(not model.training and all(p.grad is None for p in model.parameters()),'Inference model mode differs')
    state_hash=shared.state_hash(model.state_dict()); require(state_hash==plan['models'][role]['state_sha256'],'Role state differs')
    require(rss()<=limits['rss_bytes'],'Model load exceeds RSS acceptance'); runs={}; forwards=0
    for direction in DIRECTIONS:
        runs[direction]={}
        for case in plan['cases']:
            arrays=load_arrays(case); capture=Capture(model); target=directory/direction/case['id']; target.mkdir(parents=True,exist_ok=False)
            scores,meta,evaluated=runtime['reference'].infer_surface(capture,arrays['surface'],0,direction,runtime['normalize_robust'],None)
            scored=[tile for tile in meta['tiles'] if tile['status']=='scored']
            require(meta['tile_count']==25 and len(capture.logits)==len(scored)==meta['scored_tile_count'],'Tile capture alignment differs')
            require(evaluated[arrays['primary_mask']].all() and np.isfinite(scores[arrays['primary_mask']]).all(),
                'Fixed primary has unevaluated/nonfinite predictions; stop without shrinking')
            logits=np.stack(capture.logits) if capture.logits else np.empty((0,128,128),dtype=np.float32)
            np.save(target/'tile-logits.npy',logits,allow_pickle=False); np.save(target/'scores.npy',scores,allow_pickle=False)
            np.save(target/'evaluated-mask.npy',evaluated,allow_pickle=False)
            result=dict(scores=record(target/'scores.npy'),evaluated=record(target/'evaluated-mask.npy'),logits=record(target/'tile-logits.npy'),
                inference=meta,captured_tiles=[dict(bbox_roi_xywh=tile['bbox_roi_xywh'],input_sha256=digest)
                    for tile,digest in zip(scored,capture.input_hashes)],
                raw_logits_semantics='Original per-scored-tile FP32 logits before sigmoid; ordered like scored tiles, not a blended logit mosaic',
                adapter_layer_start=0,adapter_direction=direction,float32_input_without_uint8_quantization=True,
                input_fingerprint_before_model_call=True,model_input_tensor_unchanged=True,
                normalization='unchanged robust_mad1/99 over full17x128x128 patch; no reference mask enters normalization')
            write_json(target/'metadata.json',result); runs[direction][case['id']]=result; forwards+=len(scored)
            require(rss()<=limits['rss_bytes'],'Inference exceeds RSS acceptance'); del capture,arrays,logits; gc.collect()
    require(forwards<=100 and shared.state_hash(model.state_dict())==state_hash
        and all(p.grad is None for p in model.parameters()),'Forward budget or unchanged state check failed')
    load_plan(path,execute=True); result=dict(status='PASS',role=role,plan=record(path),state_sha256=state_hash,
        source_model=plan['models'][role],initializer_contract=modelmeta,saved_state_storage_contract=storage,
        runs=runs,tile_forward_count=forwards,wall_seconds=time.perf_counter()-started,peak_rss_bytes=rss(),
        training_calls=0,threshold_fitting_calls=0,model_state_unchanged=True,source_file_pins_unchanged_before_after=True)
    write_json(directory/'result.json',result); signal.alarm(0); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True); parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--preflight',action='store_true'); parser.add_argument('--worker-role',choices=ROLES)
    args=parser.parse_args(); plan,path=load_plan(args.plan,execute=True); before=record(path)
    if args.preflight:
        for case in plan['cases']: load_arrays(case)
        print(json.dumps(dict(status='PASS',plan=before,model_calls=0))); return
    require(args.output_dir is not None,'Output directory required')
    if args.worker_role: worker(plan,path,args.worker_role,args.output_dir); return
    started=time.perf_counter(); args.output_dir.mkdir(parents=True,exist_ok=False); workers=[]; results={}
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for role in ROLES:
        remaining=plan['limits']['total_wall_seconds']-(time.perf_counter()-started)
        require(remaining>0,'Total comparison wall cap exhausted')
        command=[sys.executable,str(Path(__file__).resolve()),'--plan',str(path),'--output-dir',str(args.output_dir/role),'--worker-role',role]
        with (args.output_dir/(role+'.log')).open('xb') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,
                timeout=min(remaining,plan['limits']['worker_wall_seconds']+5))
        workers.append(dict(role=role,exit_code=completed.returncode)); require(completed.returncode==0,'Worker stopped; no retry or substitution')
        item=record(args.output_dir/role/'result.json'); receipt=json.loads(verify(item).read_text())
        require(receipt['status']=='PASS' and receipt['role']==role and receipt['plan']==before
            and receipt['state_sha256']==plan['models'][role]['state_sha256'],'Worker provenance differs'); results[role]=item
        print(json.dumps(dict(role=role,status='PASS',tile_forwards=receipt['tile_forward_count'])),flush=True)
    require(sum(json.loads(verify(r).read_text())['tile_forward_count'] for r in results.values())<=500,'Total tile budget exceeded')
    load_plan(path,execute=True); require(record(path)==before,'Frozen plan changed during comparison')
    write_json(args.output_dir/'results.json',dict(status='PASS',plan=before,models=results,workers=workers,
        wall_seconds=time.perf_counter()-started,retries=0,all_fixed_roles_and_directions_retained=True))
    require(sum(f.stat().st_size for f in args.output_dir.rglob('*') if f.is_file())<=plan['limits']['output_bytes'],'Run output cap exceeded')
    print(json.dumps(dict(status='PASS',results=record(args.output_dir/'results.json'))))


if __name__=='__main__': main()
