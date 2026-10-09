"""Offline two-phase inference; all Frag3 locks precede every Paris2 prediction."""
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
from datetime import datetime, timezone


def offline(*args,**kwargs):raise RuntimeError('Network disabled in fixed-state comparison')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline
sys.dont_write_bytecode=True

from comparison_common import (ROLES,CALIBRATION_IDS,EVALUATION_IDS,load_plan,load_arrays,
    module,verify,record,read_json,write_json,require,verify_locks_ready,require_primary_coverage)


def rss():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
def utc():return datetime.now(timezone.utc).isoformat()


def worker(plan,path,role,phase,directory,barrier=None,barrier_sha=None):
    limits=plan['limits'];signal.alarm(limits['worker_wall_seconds'])
    resource.setrlimit(resource.RLIMIT_CPU,(limits['worker_cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_AS,(limits['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(limits['output_bytes'],)*2)
    started=time.perf_counter();directory.mkdir(parents=True,exist_ok=False)
    ready=verify_locks_ready(plan,barrier,barrier_sha)[1] if phase=='evaluation' else None
    import numpy as np
    import torch
    require(sys.version_info[:3]==(3,12,14) and np.__version__=='2.5.3' and torch.__version__=='2.12.0+cpu','Runtime differs')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261008)
    shared=module(plan['shared'],'sixteenth_state_hasher');helper=module(plan['helper'],'sixteenth_model_contract')
    runtime=helper.load_runtime();model,modelmeta=helper.build_model(verify(plan['initializer']),plan['initializer'],runtime)
    require(shared.state_hash(model.state_dict())==plan['models']['frozen_seed42']['state_sha256'],'Initializer differs')
    source=plan['models'][role];storage=None
    if source['checkpoint'] is not None:
        state=torch.load(verify(source['checkpoint']),map_location='cpu',weights_only=True);expected=model.state_dict()
        require(isinstance(state,dict) and set(state)==set(expected),'Raw state keys differ')
        for key,value in state.items():
            require(isinstance(value,torch.Tensor) and value.layout==torch.strided and value.shape==expected[key].shape
                and value.dtype==expected[key].dtype and bool(torch.isfinite(value).all()),'Raw tensor differs: '+key)
        storage=helper.verify_storage_contract(state,expected,torch)
        require(shared.state_hash(state)==source['state_sha256'],'Saved semantic state differs')
        strict=model.load_state_dict(state,strict=True)
        require(not strict.missing_keys and not strict.unexpected_keys,'Raw state strict load differs')
        del state,expected;gc.collect()
    model.eval();initial_hash=shared.state_hash(model.state_dict())
    require(initial_hash==source['state_sha256'] and not model.training and not list(model.named_buffers())
        and all(p.grad is None and p.dtype==torch.float32 and p.device.type=='cpu' for p in model.parameters()),'Inference state/mode differs')
    require(rss()<=limits['rss_bytes'],'Model load RSS exceeds acceptance')
    module(plan['legacy_prepare'],'prepare_reference');legacy=module(plan['legacy_worker'],'sixteenth_capture')
    selector=module(plan['cutoff'],'sixteenth_cutoff_selector');runs={};regional={};forwards=0
    cases=plan['cases'][:2] if phase=='calibration' else plan['cases'][2:]
    for case in cases:
        arrays=load_arrays(plan,case);capture=legacy.Capture(model)
        scores,meta,evaluated=runtime['reference'].infer_surface(capture,arrays['surface'],0,'forward',runtime['normalize_robust'],None)
        scored=[tile for tile in meta['tiles'] if tile['status']=='scored'];primary=arrays['primary_mask']
        require(meta['tile_count']==meta['scored_tile_count']==len(capture.logits)==25,'Scheduled tile coverage differs; no rescue')
        require_primary_coverage(scores,evaluated,primary)
        target=directory/case['id'];target.mkdir()
        np.save(target/'tile-logits.npy',np.stack(capture.logits),allow_pickle=False)
        np.save(target/'scores.npy',scores,allow_pickle=False);np.save(target/'evaluated-mask.npy',evaluated,allow_pickle=False)
        item=dict(scores=record(target/'scores.npy'),evaluated=record(target/'evaluated-mask.npy'),logits=record(target/'tile-logits.npy'),
            inference=meta,captured_tiles=[dict(bbox_roi_xywh=t['bbox_roi_xywh'],input_sha256=h) for t,h in zip(scored,capture.input_hashes)],
            adapter_layer_start=0,adapter_direction='forward',fixed_primary_mask=case['arrays']['primary_mask'],
            normalization=plan['inference']['normalization'],input_fingerprint_before_model_call=True,model_input_tensor_unchanged=True,
            raw_logits_semantics='FP32 pre-sigmoid scored-tile logits in row-major schedule; no blended logit mosaic')
        write_json(target/'metadata.json',item);runs[case['id']]=item;forwards+=len(scored)
        if phase=='calibration':regional[case['id']]=selector.choose_development_threshold(scores[primary&(arrays['labels']==0)])
        require(rss()<=limits['rss_bytes'],'Inference RSS exceeds acceptance');del arrays,capture,scores,evaluated;gc.collect()
    lock=None
    if phase=='calibration':
        for case in cases:require(regional[case['id']]['negative_count']==case['primary_negative'],'Calibration negative count differs')
        value=dict(role=role,plan=plan['_plan_record'],state_sha256=initial_hash,regional=regional,
            primary=max(row['threshold'] for row in regional.values()),original_cutoff=source['original_cutoff'],
            selector=plan['cutoff'],scope='Only two Frag3 primary negative vectors; no Frag1/2 inference',evaluation_predictions=0)
        write_json(directory/'thresholds.json',value);lock=record(directory/'thresholds.json')
    else:verify_locks_ready(plan,barrier,barrier_sha)
    require(forwards==(50 if phase=='calibration' else 100) and shared.state_hash(model.state_dict())==initial_hash
        and all(p.grad is None for p in model.parameters()),'State/gradient/forward count changed')
    load_plan(path,execute=True)
    result=dict(status='PASS',role=role,phase=phase,plan=plan['_plan_record'],state_sha256=initial_hash,source_model=source,
        initializer_contract=modelmeta,saved_state_storage_contract=storage,runs=runs,threshold_lock=lock,calibration_barrier=ready,
        tile_forward_count=forwards,wall_seconds=time.perf_counter()-started,peak_rss_bytes=rss(),model_state_unchanged=True,
        model_load_operations=dict(final_role_worker_constructions=1,initializer_strict_loads=1,
            candidate_raw_state_strict_loads=int(source['checkpoint'] is not None),
            total_load_state_dict_calls=1+int(source['checkpoint'] is not None)),
        parameter_gradients_absent=True,training_calls=0,network_calls=0,source_pins_unchanged_before_after=True)
    write_json(directory/'result.json',result);signal.alarm(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--worker-role',choices=ROLES);parser.add_argument('--phase',choices=['calibration','evaluation'])
    parser.add_argument('--locks-ready',type=Path);parser.add_argument('--locks-ready-sha256');args=parser.parse_args()
    plan,path=load_plan(args.plan,execute=True);before=record(path);plan['_plan_record']=before
    if args.preflight:
        for case in plan['cases']:load_arrays(plan,case)
        print(json.dumps(dict(status='PASS',plan=before,model_calls=0)));return
    require(args.output_dir is not None,'Output directory required')
    if args.worker_role:
        require(args.phase is not None,'Worker phase required')
        worker(plan,path,args.worker_role,args.phase,args.output_dir,args.locks_ready,args.locks_ready_sha256);return
    args.output_dir.mkdir(parents=True,exist_ok=False);started=time.perf_counter();results={};workers=[];barrier=None;load_counts={}
    order_path=args.output_dir/'phase-order.jsonl'
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for phase in ['calibration','evaluation']:
        results[phase]={}
        for role in ROLES:
            remaining=plan['limits']['total_wall_seconds']-(time.perf_counter()-started);require(remaining>0,'Aggregate wall cap exhausted')
            target=args.output_dir/phase/role;target.parent.mkdir(exist_ok=True)
            command=[sys.executable,str(Path(__file__).resolve()),'--plan',str(path),'--worker-role',role,'--phase',phase,'--output-dir',str(target)]
            if barrier:command+=['--locks-ready',barrier['path'],'--locks-ready-sha256',barrier['sha256']]
            launched=dict(event='launch',role=role,phase=phase,index=len(workers),utc=utc(),elapsed_seconds=time.perf_counter()-started,
                calibration_barrier=barrier,command=command)
            with order_path.open('a') as log:log.write(json.dumps(launched,sort_keys=True)+'\n');log.flush()
            with (args.output_dir/(phase+'-'+role+'.log')).open('xb') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=min(remaining,125))
            require(completed.returncode==0,'Worker stopped; no retry or substitution')
            item=record(target/'result.json');receipt=read_json(item)
            require(receipt['status']=='PASS' and receipt['phase']==phase and receipt['role']==role and receipt['plan']==before
                and receipt['state_sha256']==plan['models'][role]['state_sha256']
                and receipt['tile_forward_count']==(50 if phase=='calibration' else 100),'Worker lineage differs')
            completed_record=dict(event='complete',role=role,phase=phase,index=len(workers),exit_code=0,result=item,
                utc=utc(),elapsed_seconds=time.perf_counter()-started)
            with order_path.open('a') as log:log.write(json.dumps(completed_record,sort_keys=True)+'\n');log.flush()
            results[phase][role]=item;workers.append(dict(launch=launched,complete=completed_record))
            expected_loads=dict(final_role_worker_constructions=1,initializer_strict_loads=1,
                candidate_raw_state_strict_loads=int(plan['models'][role]['checkpoint'] is not None),
                total_load_state_dict_calls=1+int(plan['models'][role]['checkpoint'] is not None))
            require(receipt['model_load_operations']==expected_loads,'Model-load accounting differs')
            for key,value in expected_loads.items():load_counts[key]=load_counts.get(key,0)+value
            require(sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())<=plan['limits']['output_bytes'],'Run byte cap exceeded')
        if phase=='calibration':
            locks={role:read_json(item)['threshold_lock'] for role,item in results[phase].items()}
            locked=dict(status='PASS',plan=before,roles=ROLES,models=results[phase],threshold_locks=locks,
                calibration_tile_forwards=150,evaluation_tile_forwards=0,all_three_locks_written_before_evaluation=True,
                locked_at_utc=utc(),locked_at_elapsed_seconds=time.perf_counter()-started,calibration_workers=workers.copy())
            temporary=args.output_dir/'locks-ready.pending.json';write_json(temporary,locked)
            os.replace(temporary,args.output_dir/'locks-ready.json')
            barrier=record(args.output_dir/'locks-ready.json');verify_locks_ready(plan,barrier['path'],barrier['sha256'])
    load_plan(path,execute=True);verify_locks_ready(plan,barrier['path'],barrier['sha256'])
    require(record(path)==before and time.perf_counter()-started<=plan['limits']['total_wall_seconds'],'Plan/aggregate wall cap differs')
    write_json(args.output_dir/'results.json',dict(status='PASS',plan=before,models=results,locks_ready=barrier,workers=workers,
        wall_seconds=time.perf_counter()-started,tile_forward_count=450,model_load_operations=load_counts,
        retries=0,phase_order=record(order_path),all_fixed_states_and_contexts_retained=True))
    require(sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())<=plan['limits']['output_bytes'],'Final byte cap exceeded')
    print(json.dumps(dict(status='PASS',results=record(args.output_dir/'results.json'))))


if __name__=='__main__':main()
