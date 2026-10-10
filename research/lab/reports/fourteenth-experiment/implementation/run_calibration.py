"""Offline calibration inference only; publish both locks before diagnostic analysis."""
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


def offline(*args,**kwargs): raise RuntimeError('Network disabled in calibration control')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline

from calibration_common import ARMS,BASELINE_CUTOFF,load_plan,development_data,selection_from_map,module,verify,record,read_json,write_json,require


def rss():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def worker(plan,path,arm,directory):
    limits=plan['limits'];signal.alarm(limits['worker_wall_seconds'])
    resource.setrlimit(resource.RLIMIT_CPU,(limits['worker_cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_AS,(limits['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(limits['output_bytes'],)*2)
    started=time.perf_counter();directory.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    require(sys.version_info[:3]==(3,12,14) and np.__version__=='2.5.3' and torch.__version__=='2.12.0+cpu','Runtime differs')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261008)
    shared=module(plan['shared'],'fourteenth_state_hasher');helper=module(plan['helper'],'fourteenth_model_contract')
    runtime=helper.load_runtime();model,modelmeta=helper.build_model(verify(plan['initializer']),plan['initializer'],runtime)
    require(shared.state_hash(model.state_dict())==plan['initial_state_sha256'],'Initializer semantic state differs')
    source=plan['models'][arm];state=torch.load(verify(source['checkpoint']),map_location='cpu',weights_only=True);expected=model.state_dict()
    require(isinstance(state,dict) and set(state)==set(expected),'Final state keys differ')
    for key,value in state.items():
        require(isinstance(value,torch.Tensor) and value.layout==torch.strided and value.shape==expected[key].shape
            and value.dtype==expected[key].dtype and bool(torch.isfinite(value).all()),'Final tensor contract differs: '+key)
    storage=helper.verify_storage_contract(state,expected,torch);require(shared.state_hash(state)==source['state_sha256'],'Final state semantic hash differs')
    strict=model.load_state_dict(state,strict=True);require(not strict.missing_keys and not strict.unexpected_keys,'Strict final state load differs')
    del state,expected;gc.collect();model.eval()
    require(not model.training and not list(model.named_buffers()) and all(p.grad is None for p in model.parameters()),'Inference mode/buffers/gradients differ')
    initial_hash=shared.state_hash(model.state_dict());require(initial_hash==source['state_sha256'],'Loaded state differs')
    require(rss()<=limits['rss_bytes'],'Load RSS exceeds acceptance')
    sys.path.insert(0,str(verify(plan['legacy_prepare']).parent));legacy=module(plan['legacy_worker'],'fourteenth_archived_capture')
    selector=module(plan['cutoff'],'fourteenth_original_selector');maps={};regional={};forwards=0
    for case,expected_tiles in zip(plan['development'],plan['inference']['tiles_by_case']):
        surface,labels,primary=development_data(plan,case);capture=legacy.Capture(model)
        scores,meta,evaluated=runtime['reference'].infer_surface(capture,surface,2,'forward',runtime['normalize_robust'],None)
        scored=[tile for tile in meta['tiles'] if tile['status']=='scored']
        require(meta['tile_count']==meta['scored_tile_count']==len(scored)==len(capture.logits)==expected_tiles,'Calibration tile coverage differs')
        require(scores.dtype==np.float32 and scores.shape==primary.shape and evaluated.dtype==np.bool_ and evaluated.shape==primary.shape
            and evaluated[primary].all() and np.isfinite(scores[primary]).all(),'Fixed calibration support unscored/nonfinite; stop without shrinking')
        captured=[dict(bbox_roi_xywh=t['bbox_roi_xywh'],input_sha256=h) for t,h in zip(scored,capture.input_hashes)]
        baseline_metadata=read_json(case['baseline_maps']['metadata'])
        baseline_tiles=[t['bbox_roi_xywh'] for t in baseline_metadata['inference']['tiles'] if t['status']=='scored']
        require([t['bbox_roi_xywh'] for t in captured]==baseline_tiles,'Archived calibration tile placement differs')
        target=directory/case['id'];target.mkdir()
        np.save(target/'scores.npy',scores,allow_pickle=False);np.save(target/'evaluated-mask.npy',evaluated,allow_pickle=False)
        np.save(target/'tile-logits.npy',np.stack(capture.logits),allow_pickle=False);np.save(target/'fixed-primary-mask.npy',primary,allow_pickle=False)
        item=dict(scores=record(target/'scores.npy'),evaluated=record(target/'evaluated-mask.npy'),logits=record(target/'tile-logits.npy'),
            fixed_primary_mask=record(target/'fixed-primary-mask.npy'),inference=meta,captured_tiles=captured,
            adapter_layer_start=2,adapter_direction='forward',normalization=plan['inference']['normalization'],
            model_input_tensor_unchanged=True,archived_input_limit='Original calibration baselines lack cached normalized tile fingerprints; source/depth/normalizer/adapter pins and tile placements are preserved.')
        write_json(target/'metadata.json',item);maps[case['id']]=item
        regional[case['id']]=dict(selection=selector.choose_development_threshold(scores[primary&~labels]),maps=item)
        require(regional[case['id']]['selection']['negative_count']==plan['negative_counts'][case['id']],'Canonical negative count differs')
        forwards+=len(scored);require(rss()<=limits['rss_bytes'],'Inference RSS exceeds acceptance')
        del surface,labels,primary,capture,scores,evaluated;gc.collect()
    require(forwards==59 and shared.state_hash(model.state_dict())==initial_hash and all(p.grad is None for p in model.parameters()),'State/forward budget changed')
    lock=dict(arm=arm,plan=record(path),state_sha256=initial_hash,development=regional,
        primary=max(row['selection']['threshold'] for row in regional.values()),secondary=regional['pherc1667-w029']['selection']['threshold'],
        selector=plan['cutoff'],scope='Only original three development-negative vectors; no diagnostic arrays loaded or diagnostic fitting',diagnostic_arrays_loaded=False)
    write_json(directory/'thresholds.json',lock);load_plan(path,execute=True)
    output=dict(status='PASS',arm=arm,plan=record(path),state_sha256=initial_hash,source_model=source,initializer_contract=modelmeta,
        saved_state_storage_contract=storage,development=maps,threshold_lock=record(directory/'thresholds.json'),tile_forward_count=forwards,
        wall_seconds=time.perf_counter()-started,peak_rss_bytes=rss(),model_state_unchanged=True,parameter_gradients_absent=True,
        actual_calls=dict(calibration_tile_forwards=59,baseline_forwards=0,diagnostic_forwards=0,backwards=0,updates=0),network_requests=0)
    write_json(directory/'result.json',output);signal.alarm(0)


def reconstruct_baseline(plan,directory):
    selector=module(plan['cutoff'],'fourteenth_baseline_selector');original=read_json(plan['baseline_lock']);regional={}
    for case in plan['development']:
        selection=selection_from_map(plan,case,case['baseline_maps'],selector)
        require(selection==original['development'][case['id']]['selection'],'Baseline regional threshold reconstruction differs')
        regional[case['id']]=selection
    primary=max(row['threshold'] for row in regional.values());secondary=regional['pherc1667-w029']['threshold']
    require(primary==secondary==BASELINE_CUTOFF==original['primary']==original['secondary'],'Literal baseline lock reconstruction differs')
    target=directory/'baseline-reconstruction.json';write_json(target,dict(status='PASS',original_lock=plan['baseline_lock'],regional=regional,
        primary=primary,secondary=secondary,original_bytes_unchanged=True,baseline_forwards=0,diagnostic_arrays_loaded=False))
    return record(target)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--worker-arm',choices=ARMS)
    args=parser.parse_args();plan,path=load_plan(args.plan,execute=True);before=record(path)
    if args.worker_arm:worker(plan,path,args.worker_arm,args.output_dir);return
    args.output_dir.mkdir(parents=True,exist_ok=False);started=time.perf_counter();baseline=reconstruct_baseline(plan,args.output_dir)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    results={};locks={};workers=[]
    for arm in ARMS:
        remaining=plan['limits']['total_wall_seconds']-(time.perf_counter()-started);require(remaining>0,'Combined wall cap exhausted')
        command=[sys.executable,str(Path(__file__).resolve()),'--plan',str(path),'--worker-arm',arm,'--output-dir',str(args.output_dir/arm)]
        with (args.output_dir/(arm+'.log')).open('xb') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=min(remaining,95))
        workers.append(dict(arm=arm,returncode=completed.returncode));require(completed.returncode==0,'Calibration worker failed; no retry/substitution')
        item=record(args.output_dir/arm/'result.json');receipt=read_json(item)
        require(receipt['status']=='PASS' and receipt['arm']==arm and receipt['plan']==before
            and receipt['state_sha256']==plan['models'][arm]['state_sha256'] and receipt['tile_forward_count']==59,'Worker receipt differs')
        results[arm]=item;locks[arm]=receipt['threshold_lock'];verify(locks[arm])
        require(sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())<=plan['limits']['output_bytes'],'Combined run artifact cap exceeded')
    load_plan(path,execute=True);require(record(path)==before and time.perf_counter()-started<=plan['limits']['total_wall_seconds'],'Final plan/wall check failed')
    output=dict(status='PASS',plan=before,models=results,threshold_locks=locks,baseline_reconstruction=baseline,workers=workers,
        tile_forward_count=118,retries=0,wall_seconds=time.perf_counter()-started,
        both_new_locks_written_before_diagnostic_analysis=True,diagnostic_arrays_loaded=False)
    write_json(args.output_dir/'locks-ready.json',output)
    require(sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())<=plan['limits']['output_bytes'],'Final run artifact cap exceeded')
    print(json.dumps(dict(status='PASS',locks_ready=record(args.output_dir/'locks-ready.json'))))


if __name__=='__main__':main()
