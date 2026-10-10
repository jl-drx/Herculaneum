"""Additive v2 offline audit: unique storage aliases and pinned stored step74999."""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
from model_contract_v2 import HERE, ContractError, load_plan, require, load_runtime, build_model, audit_matched_pair, config_differences


def disable_network():
    def blocked(*args,**kwargs):
        raise ContractError('Offline audit attempted socket access; stopped')
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.socket.sendto = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked


def worker(plan_path,plan_sha,output):
    disable_network()  # Before torch/vendor imports or checkpoint deserialization.
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    plan = load_plan(plan_path,plan_sha)
    runtime = load_runtime()
    records = {}
    for artifact in plan['artifacts']:
        model,record = build_model(artifact['local_path'],artifact,runtime)
        require(record['parameter_count'] == 34546498, 'Parameter count differs from retained hybrid model')
        records[artifact['label']] = record
        del model
        gc.collect()
    require(len({r['tensor_schema_sha256'] for r in records.values()}) == 1,
            'Four arms do not have exactly the same tensor keys/shapes/dtypes')
    matched = audit_matched_pair(records['russo_control75k']['embedded_config'],records['russo_ex01675k']['embedded_config'])
    published = {role:json.loads(Path(path).read_text()) for role,path in plan['published_configs'].items()}
    audit_matched_pair(published['russo_control75k'],published['russo_ex01675k'])
    source_differences = {}
    for role,source_config in published.items():
        embedded = records[role]['embedded_config']
        # Published fields are checked in full; canonical/default-only additions are reported separately.
        source_differences[role] = config_differences(source_config,{k:embedded[k] for k in source_config if k in embedded})
        require(not source_differences[role], 'Embedded config differs from published author config')
    baseline_differences = {role:config_differences(records['seed42']['embedded_config'],records[role]['embedded_config'])
                            for role in records if role != 'seed42'}
    load_plan(plan_path,plan_sha)
    result = dict(schema_version=1,status='passed',plan_path=str(plan_path),plan_sha256=plan_sha,
        artifacts=plan['artifacts'],models=records,matched_pair=matched,
        embedded_versus_published_differences=source_differences,
        baseline_full_config_differences=baseline_differences,
        inference_performed=False,network_requests=0,cpu_threads=4,
        tensor_schema_identical_across_four_arms=True,
        physical_independence='Not established by checkpoint contract',
        limitations=plan['interpretation_limits'])
    (output/'model-contract.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    output = args.output_dir.resolve()
    require(output.parent == HERE, 'Audit outputs must stay in owned directory')
    if args.worker:
        worker(plan_path,args.plan_sha256,output)
        return 0
    load_plan(plan_path,args.plan_sha256)
    output.mkdir(exist_ok=False)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    command = [sys.executable,str(Path(__file__).resolve()),'--worker','--plan',str(plan_path),
               '--plan-sha256',args.plan_sha256,'--output-dir',str(output)]
    result = dict(schema_version=1,plan_path=str(plan_path),plan_sha256=args.plan_sha256,
                  wall_timeout_seconds=180,cpu_seconds_limit=120,address_space_limit_bytes=8*1024**3,
                  network_requests=0,inference_performed=False,status='stopped')
    try:
        with (output/'audit-stdout.log').open('w') as stdout,(output/'audit-stderr.log').open('w') as stderr:
            completed = subprocess.run(command,env=env,stdout=stdout,stderr=stderr,timeout=180)
        result['returncode'] = completed.returncode
        load_plan(plan_path,args.plan_sha256)
        if completed.returncode == 0:
            require((output/'model-contract.json').exists(), 'Successful worker lacked contract result')
            result['status'] = 'passed'
    except subprocess.TimeoutExpired:
        result['stop_reason'] = 'CPU audit exceeded wall limit; no automatic rerun'
    except Exception as error:
        result['stop_reason'] = str(error) if isinstance(error,ContractError) else type(error).__name__
    (output/'audit-execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return 0 if result['status']=='passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
