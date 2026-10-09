"""Conditional offline inference of eligible final states, forward only."""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time


def offline(*args, **kwargs): raise RuntimeError('Network disabled in retention worker')
socket.create_connection=offline; socket.socket.connect=offline; socket.socket.connect_ex=offline

from retention_common import ARMS, load_plan, load_case, module, verify, record, read_json, write_json, require


def rss(): return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def worker(plan, path, arm, directory):
    limits=plan['limits']; signal.alarm(limits['worker_wall_seconds'])
    resource.setrlimit(resource.RLIMIT_AS,(limits['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(limits['worker_cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(limits['output_bytes'],)*2)
    started=time.perf_counter(); directory.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    require(sys.version_info[:3] == (3,12,14) and np.__version__ == '2.5.3' and torch.__version__ == '2.12.0+cpu', 'Runtime differs')
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261008)
    shared=module(plan['shared'],'thirteenth_state_hasher'); helper=module(plan['helper'],'thirteenth_model_contract')
    runtime=helper.load_runtime(); initializer=plan['initializer']; model,modelmeta=helper.build_model(verify(initializer),initializer,runtime)
    require(shared.state_hash(model.state_dict()) == plan['initial_state_sha256'], 'Initializer state differs')
    source=plan['models'][arm]; state=torch.load(verify(source['checkpoint']),map_location='cpu',weights_only=True)
    expected=model.state_dict(); require(isinstance(state,dict) and set(state)==set(expected),'Final state keys differ')
    for key,value in state.items():
        require(isinstance(value,torch.Tensor) and value.layout==torch.strided and value.shape==expected[key].shape
            and value.dtype==expected[key].dtype and bool(torch.isfinite(value).all()),'Final tensor schema differs: '+key)
    storage=helper.verify_storage_contract(state,expected,torch)
    require(shared.state_hash(state)==source['state_sha256'],'Final semantic state differs')
    strict=model.load_state_dict(state,strict=True)
    require(not strict.missing_keys and not strict.unexpected_keys,'Final state strict load differs')
    del state,expected; gc.collect(); model.eval()
    require(not list(model.named_buffers()) and all(p.grad is None for p in model.parameters()),'Inference state/mode differs')
    require(modelmeta['embedded_config']['targets']['ink']['activation']=='none','Output activation differs')
    state_hash=shared.state_hash(model.state_dict()); require(state_hash==source['state_sha256'],'Loaded state differs')
    require(rss()<=limits['rss_bytes'],'Model load RSS exceeds acceptance')
    # Reuse the archived capture wrapper, whose outputs are returned unchanged.
    sys.path.insert(0,str(verify(plan['legacy_prepare']).parent))
    legacy=module(plan['legacy_worker'],'thirteenth_archived_capture'); runs={}; forwards=0
    for case in plan['cases']:
        surface,labels,primary=load_case(plan,case); capture=legacy.Capture(model)
        scores,meta,evaluated=runtime['reference'].infer_surface(capture,surface,case['layer_start'],'forward',runtime['normalize_robust'],None)
        scored=[tile for tile in meta['tiles'] if tile['status']=='scored']
        require(meta['tile_count']==meta['scored_tile_count']==len(scored)==len(capture.logits)==25,'Fixed tile coverage differs')
        require(scores.dtype==np.float32 and scores.shape==(384,384) and evaluated.dtype==np.bool_
            and evaluated.shape==(384,384) and evaluated[primary].all() and np.isfinite(scores[primary]).all(),
            'Canonical mask has invalid predictions; stop without shrinkage')
        captured=[dict(bbox_roi_xywh=t['bbox_roi_xywh'],input_sha256=h) for t,h in zip(scored,capture.input_hashes)]
        baseline=case['historical_maps']['frozen_seed42']; old_scored=[t['bbox_roi_xywh'] for t in baseline['inference']['tiles'] if t['status']=='scored']
        require([t['bbox_roi_xywh'] for t in captured]==old_scored,'Archived tile placement differs')
        fingerprint_match=None
        if 'captured_tiles' in baseline:
            require(captured==baseline['captured_tiles'],'Archived normalized tile input differs'); fingerprint_match=True
        target=directory/case['id']; target.mkdir()
        np.save(target/'scores.npy',scores,allow_pickle=False); np.save(target/'evaluated-mask.npy',evaluated,allow_pickle=False)
        np.save(target/'tile-logits.npy',np.stack(capture.logits),allow_pickle=False)
        item=dict(scores=record(target/'scores.npy'),evaluated=record(target/'evaluated-mask.npy'),logits=record(target/'tile-logits.npy'),
            inference=meta,captured_tiles=captured,adapter_layer_start=case['layer_start'],adapter_direction='forward',
            archived_normalized_input_fingerprint_match=fingerprint_match,
            archived_input_limit=None if fingerprint_match else 'Historical PHerc0814 maps lack cached tile input fingerprints; equality is bounded by frozen source/depth/normalizer/adapter pins and tile placement.',
            model_input_tensor_unchanged=True,normalization=plan['inference']['normalization'])
        write_json(target/'metadata.json',item); runs[case['id']]=item; forwards+=len(scored)
        require(rss()<=limits['rss_bytes'],'Inference RSS exceeds acceptance'); del capture,surface,labels,primary,scores,evaluated; gc.collect()
    require(forwards==100 and shared.state_hash(model.state_dict())==state_hash and all(p.grad is None for p in model.parameters()),'Forward budget/state differs')
    load_plan(path,execute=True)
    output=dict(status='PASS',arm=arm,plan=record(path),source_model=source,state_sha256=state_hash,
        initializer_contract=modelmeta,saved_state_storage_contract=storage,runs=runs,tile_forward_count=forwards,
        peak_rss_bytes=rss(),wall_seconds=time.perf_counter()-started,training_calls=0,threshold_fitting_calls=0,
        baseline_forward_calls=0,model_state_unchanged=True,network_requests=0)
    write_json(directory/'result.json',output); signal.alarm(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True); parser.add_argument('--worker-arm',choices=ARMS)
    args=parser.parse_args(); plan,path=load_plan(args.plan,execute=True); before=record(path)
    if args.worker_arm:
        require(args.worker_arm in plan['eligible_arms'],'Unqualified state inference prohibited')
        worker(plan,path,args.worker_arm,args.output_dir); return
    args.output_dir.mkdir(parents=True,exist_ok=False); started=time.perf_counter(); results={}; workers=[]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for arm in plan['eligible_arms']:
        remaining=plan['limits']['total_wall_seconds']-(time.perf_counter()-started); require(remaining>0,'Combined wall cap exhausted')
        command=[sys.executable,str(Path(__file__).resolve()),'--plan',str(path),'--worker-arm',arm,'--output-dir',str(args.output_dir/arm)]
        with (args.output_dir/(arm+'.log')).open('xb') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=min(remaining,125))
        workers.append(dict(arm=arm,returncode=completed.returncode)); require(completed.returncode==0,'Worker failed; no retry/substitution')
        item=record(args.output_dir/arm/'result.json'); result=read_json(item)
        require(result['status']=='PASS' and result['arm']==arm and result['plan']==before and result['tile_forward_count']==100
            and result['state_sha256']==plan['models'][arm]['state_sha256'],'Worker receipt differs'); results[arm]=item
        require(sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())<=plan['limits']['output_bytes'],'Combined output cap exceeded')
    load_plan(path,execute=True); require(record(path)==before,'Plan changed during retention')
    total=sum(read_json(item)['tile_forward_count'] for item in results.values()); require(total<=200,'Combined forward cap exceeded')
    require(time.perf_counter()-started<=plan['limits']['total_wall_seconds'],'Final combined wall cap exceeded')
    write_json(args.output_dir/'results.json',dict(status='PASS' if results else 'SKIPPED',plan=before,models=results,
        eligible_arms=plan['eligible_arms'],workers=workers,tile_forward_count=total,wall_seconds=time.perf_counter()-started,
        retries=0,skip_reason=None if results else 'Neither final state passed both fixed training-loss improvement requirements; no inference or substitute state.',
        complete_fixed_contexts_per_eligible_state=True))
    require(sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())<=plan['limits']['output_bytes'],'Final combined output cap exceeded')
    print(json.dumps(dict(status='PASS' if results else 'SKIPPED',result=record(args.output_dir/'results.json'))))


if __name__=='__main__': main()
