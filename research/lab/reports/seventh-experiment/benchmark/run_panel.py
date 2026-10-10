#!/usr/bin/env python3
"""Run the frozen other-scroll panel sequentially; never fit reserved thresholds."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

import numpy as np

from panel_common import ARMS, base, load_case, load_plan


def worker(plan_path, case_id, role, output):
    started=time.perf_counter()
    plan,plan_path,_=load_plan(plan_path,execute=True)
    case=next(c for c in plan['cases'] if c['id']==case_id)
    arrays,_,_,_,preflight=load_case(case)
    artifact=next(m for m in plan['models'] if m['label']==role)
    output=Path(output).resolve();output.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024*1024,8*1024*1024))
    def offline(*args,**kwargs):
        raise ValueError('Offline panel attempted network access; no retry')
    socket.socket.connect=offline;socket.socket.connect_ex=offline
    socket.socket.sendto=offline;socket.create_connection=offline;socket.getaddrinfo=offline
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    helper=base.import_helper(plan);runtime=helper.load_runtime()
    model,model_metadata=helper.build_model(artifact['path'],artifact,runtime)
    scores,inference,evaluated=runtime['reference'].infer_surface(model,arrays['surface'],
        case['data']['model_layer_start'],'forward',runtime['normalize_robust'],valid_mask=None)
    base.require(scores.dtype==np.float32 and scores.shape==arrays['surface'].shape[1:]
        and evaluated.dtype==np.bool_ and evaluated.shape==scores.shape, 'Invalid output shape/dtype')
    base.require(np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1)), 'Invalid raw scores')
    np.save(output/'scores.npy',scores,allow_pickle=False)
    np.save(output/'evaluated-mask.npy',evaluated,allow_pickle=False)
    base.write_json(output/'metadata.json',dict(case_id=case_id,role=role,plan=base.record(plan_path),
        source_inputs=case['inputs'],source_provenance=case['provenance'],data_contract=case['data'],
        threshold_lock=plan['threshold_lock'],execution_code=plan['code'],model=model_metadata,
        raw_scores=base.record(output/'scores.npy'),evaluated_mask=base.record(output/'evaluated-mask.npy'),
        preflight=preflight,inference=inference,
        runtime=dict(device='cpu',precision='float32',batch_size=1,threads=4,interop_threads=1,
            deterministic_algorithms=True,wall_seconds=time.perf_counter()-started,
            peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            package_versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','timm','numpy','einops']}),
        canvas_semantics='valid_mask=None is all-ones canvas only; valid_pixel_count is not authenticated geometry.',
        interpretation='Raw uncalibrated score; supplied-reference developmental comparison, no reading.'))
    print(json.dumps(dict(case_id=case_id,role=role,status='complete',tiles=inference['scored_tile_count'],seconds=inference['inference_seconds'])))


def verify_all(plan):
    summaries={case['id']:load_case(case)[-1] for case in plan['cases']}
    helper=base.import_helper(plan)
    for model in plan['models']:
        helper.verify_artifact(model['path'],model)
    return summaries


def run(plan_path,output,execute):
    plan,plan_path,lock=load_plan(plan_path,execute=execute)
    summaries=verify_all(plan)
    if not execute:
        print(json.dumps(dict(status='preflight_passed_no_inference',plan=base.record(plan_path),cases=summaries),indent=2));return
    base.require(output is not None,'Fresh --output-dir required')
    output=Path(output).resolve();output.mkdir(exist_ok=False)
    (output/'frozen-plan.json').write_bytes(plan_path.read_bytes())
    (output/'threshold-lock.json').write_bytes(Path(plan['threshold_lock']['path']).read_bytes())
    base.write_json(output/'preflight.json',dict(plan=base.record(plan_path),cases=summaries,threshold_lock=plan['threshold_lock']))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    runs=[]
    try:
        for case in plan['cases']:
            case_output=output/case['id'];case_output.mkdir(exist_ok=False)
            for role in ARMS:
                log=case_output/f'{role}.log';started=time.perf_counter()
                with log.open('w') as stream:
                    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--plan',str(plan_path),
                        '--worker',case['id'],role,'--output-dir',str(case_output/role)],
                        stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=plan['limits']['per_worker_wall_seconds'])
                runs.append(dict(case_id=case['id'],role=role,exit_code=result.returncode,wall_seconds=time.perf_counter()-started,log=base.record(log)))
                base.write_json(output/'execution.json',dict(status='running',runs=runs,plan=base.record(plan_path)))
                base.require(result.returncode==0,'STOP: worker failed; no retry')
                base.require(base.output_bytes(output)<=plan['limits']['maximum_output_bytes'],'STOP: output budget exceeded')
        from analyze_panel import analyze
        analysis=analyze(plan_path,output)
        final_plan,_,_=load_plan(plan_path,execute=True);verify_all(final_plan)
        base.require((output/'frozen-plan.json').read_bytes()==plan_path.read_bytes(),'Frozen plan changed')
        base.require(base.output_bytes(output)<=plan['limits']['maximum_output_bytes'],'STOP: output budget exceeded')
        base.write_json(output/'execution.json',dict(status='complete',runs=runs,plan=base.record(plan_path),analysis=analysis,output_bytes=base.output_bytes(output)))
        print(json.dumps(dict(status='complete',output=str(output),analysis=analysis)))
    except Exception as error:
        base.write_json(output/'execution.json',dict(status='stopped_no_retry',runs=runs,reason=str(error),exception_type=type(error).__name__))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',nargs=2,metavar=('CASE','MODEL'),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        base.require(args.worker[1] in ARMS and args.output_dir is not None,'Invalid worker arguments')
        worker(args.plan,*args.worker,args.output_dir)
    else:
        run(args.plan,args.output_dir,args.execute)


if __name__=='__main__':
    main()
