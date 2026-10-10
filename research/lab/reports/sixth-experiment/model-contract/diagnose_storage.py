"""Additive offline diagnosis of state tensor names, storage aliases and factory schema."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import zipfile
import model_contract as original
from audit_checkpoints import disable_network

HERE = Path(__file__).resolve().parent


def measure(state,torch):
    original.require(len(state) <= 4096,'Diagnostic tensor-count cap')
    groups = defaultdict(list)
    schema = []
    for key in sorted(state):
        tensor = state[key]
        original.require(isinstance(tensor,torch.Tensor) and tensor.layout == torch.strided and
                         tensor.device.type == 'cpu' and not tensor.is_quantized and not tensor.is_complex(),
                         'Diagnostic requires ordinary dense CPU tensors')
        storage = tensor.untyped_storage()
        groups[storage._cdata].append(key)
        schema.append(dict(key=key,shape=list(tensor.shape),dtype=str(tensor.dtype),
            stride=list(tensor.stride()),storage_offset=tensor.storage_offset(),
            numel=tensor.numel(),named_bytes=tensor.numel()*tensor.element_size()))
    named = sum(row['named_bytes'] for row in schema)
    result_groups = []
    for names in sorted(groups.values(),key=lambda names:names[0]):
        first = state[names[0]]
        exact_view = all(state[k].dtype == first.dtype and state[k].shape == first.shape and
                         state[k].stride() == first.stride() and state[k].storage_offset() == first.storage_offset()
                         for k in names)
        equal = exact_view and all(torch.equal(first,state[k]) for k in names)
        result_groups.append(dict(keys=names,storage_bytes=first.untyped_storage().nbytes(),
            exact_same_view=exact_view,alias_values_equal=equal,
            named_bytes=sum(state[k].numel()*state[k].element_size() for k in names)))
    encoded = json.dumps(schema,sort_keys=True,separators=(',',':')).encode()
    return dict(tensor_count=len(state),named_tensor_numel=sum(t.numel() for t in state.values()),
        named_tensor_bytes=named,unique_storage_count=len(result_groups),
        unique_storage_bytes=sum(g['storage_bytes'] for g in result_groups),
        duplicate_named_bytes=named-sum(g['storage_bytes'] for g in result_groups),
        all_finite=all(bool(torch.isfinite(t).all()) for t in state.values()),
        storage_groups=result_groups,alias_groups=[g for g in result_groups if len(g['keys'])>1],
        schema=schema,schema_sha256=hashlib.sha256(encoded).hexdigest())


def worker(plan_path,plan_sha,output):
    disable_network()
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    plan = original.load_plan(plan_path,plan_sha)
    frozen = json.loads((HERE/'package-freeze.json').read_text())
    preserved = frozen['files'] + [dict(path=str(HERE/'audit'/name),size_bytes=(HERE/'audit'/name).stat().st_size,
        sha256=original.sha(HERE/'audit'/name)) for name in ('audit-execution.json','audit-stderr.log','audit-stdout.log')]
    for row in preserved:
        original.require(Path(row['path']).stat().st_size == row['size_bytes'] and original.sha(row['path']) == row['sha256'],
                         'Original frozen file or failed audit changed before diagnosis')
    runtime = original.load_runtime()
    records = {}
    for artifact in plan['artifacts']:
        path = Path(artifact['local_path'])
        original.verify_artifact(path,artifact)
        with zipfile.ZipFile(path) as archive:
            original.require(len(archive.infolist()) <= 8192 and
                             sum(entry.file_size for entry in archive.infolist()) <= original.MAX_ARCHIVE_BYTES,
                             'Diagnostic expanded ZIP cap exceeded')
        payload = torch.load(path,map_location='cpu',weights_only=True)
        original.require(isinstance(payload,dict) and isinstance(payload.get('config'),dict),'Config mapping absent')
        authored = payload['config']
        original.validate_authored_config(authored)
        config = runtime['InkConfig'].from_mapping(authored)
        container,state = runtime['reference'].selected_state(payload)
        original.require(container == 'model','Diagnostic weight container differs')
        measured = measure(state,torch)
        factory = runtime['make_model'](config)
        factory_state = factory.state_dict()
        factory_measured = measure(factory_state,torch)
        missing = sorted(set(factory_state)-set(state))
        unexpected = sorted(set(state)-set(factory_state))
        shape_dtype = [key for key in sorted(set(state)&set(factory_state)) if
                       state[key].shape != factory_state[key].shape or state[key].dtype != factory_state[key].dtype]
        def partition(measurement):
            return sorted(tuple(group['keys']) for group in measurement['storage_groups'])
        partition_match = not missing and not unexpected and partition(measured)==partition(factory_measured)
        strict_load = None
        loaded = None
        if not missing and not unexpected and not shape_dtype:
            loaded = factory.load_state_dict(state,strict=True)
            strict_load = not loaded.missing_keys and not loaded.unexpected_keys
        factory_alias_checks = []
        for group in factory_measured['alias_groups']:
            names = group['keys']
            available = all(k in state for k in names)
            consistent = available and all(torch.equal(state[names[0]].contiguous().reshape(-1).view(torch.uint8),
                                                       state[k].contiguous().reshape(-1).view(torch.uint8)) for k in names)
            factory_alias_checks.append(dict(keys=names,all_checkpoint_keys_present=available,
                                             checkpoint_values_bitwise_equal=consistent))
        record = dict(artifact=artifact,step=payload.get('step'),selected_container=container,
            checkpoint=measured,factory=factory_measured,missing_keys=missing,unexpected_keys=unexpected,
            shape_or_dtype_mismatch_keys=shape_dtype,alias_partition_matches_factory=partition_match,
            expected_factory_alias_value_checks=factory_alias_checks,
            expected_factory_alias_values_all_consistent=all(g['checkpoint_values_bitwise_equal'] for g in factory_alias_checks),
            strict_factory_load_passed=strict_load,
            unique_parameter_numel=sum(p.numel() for p in factory.parameters()),
            unique_parameter_bytes=sum(p.numel()*p.element_size() for p in factory.parameters()),
            named_parameter_numel_with_duplicates=sum(p.numel() for _,p in factory.named_parameters(remove_duplicate=False)),
            embedded_config=authored,original_named_byte_cap=original.MAX_STATE_BYTES,
            original_named_byte_cap_exceeded=measured['named_tensor_bytes']>original.MAX_STATE_BYTES,
            original_unique_storage_byte_cap_exceeded=measured['unique_storage_bytes']>original.MAX_STATE_BYTES)
        original.verify_artifact(path,artifact)
        records[artifact['label']] = record
        del payload,state,factory,factory_state,loaded
        gc.collect()
    for row in preserved:
        original.require(Path(row['path']).stat().st_size == row['size_bytes'] and original.sha(row['path']) == row['sha256'],
                         'Original frozen file or failed audit changed across diagnosis')
    original.load_plan(plan_path,plan_sha)
    result = dict(schema_version=1,status='diagnosed',network_requests=0,inference_calls=0,
        original_frozen_and_failed_files_preserved=preserved,plan_sha256=plan_sha,
        diagnostic_code_sha256=original.sha(__file__),models=records,
        interpretation='Storage accounting and strict local factory correspondence only; no ink or scientific benchmark conclusion.')
    (output/'storage-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    original.require(output.parent == HERE,'Diagnostic output must stay in owned directory')
    if args.worker:
        worker(args.plan.resolve(),args.plan_sha256,output)
        return 0
    original.load_plan(args.plan,args.plan_sha256)
    output.mkdir(exist_ok=False)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    command = [sys.executable,str(Path(__file__).resolve()),'--worker','--plan',str(args.plan.resolve()),
               '--plan-sha256',args.plan_sha256,'--output-dir',str(output)]
    with (output/'stdout.log').open('w') as stdout,(output/'stderr.log').open('w') as stderr:
        completed = subprocess.run(command,env=env,stdout=stdout,stderr=stderr,timeout=180)
    record = dict(returncode=completed.returncode,network_requests=0,inference_calls=0,
        wall_timeout_seconds=180,cpu_seconds_limit=120,address_space_limit_bytes=8*1024**3,
        diagnostic_code_sha256=original.sha(__file__))
    (output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
