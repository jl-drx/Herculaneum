"""Two isolated CPU workers; no execution without source-bound root release."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
FORBIDDEN_COUNTS = {'network': 0, 'torch_load': 0}


def offline(*args, **kwargs):
    FORBIDDEN_COUNTS['network'] += 1
    raise RuntimeError('Network forbidden during phase22 parity')


socket.create_connection = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline
socket.getaddrinfo = offline

import numpy as np
import torch
import safetensors
from safetensors.torch import load_file
from parity_sources import (HERE, PHASE, require, pin, verify, read_json, build_factory,
                            STUB_COUNTS, extraction_records)
from parity_inference import synthetic_volume, historical_case, current_case, save_case, array_record


def forbidden_torch_load(*args, **kwargs):
    FORBIDDEN_COUNTS['torch_load'] += 1
    raise RuntimeError('torch.load forbidden; only qualified safetensors allowed')


torch.load = forbidden_torch_load


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def load_plan(path, release):
    plan = json.loads(Path(path).read_text())
    require(plan['kind'] == 'phase22-native-cpu-synthetic-parity-v1', 'Unexpected plan')
    for item in plan['sources'].values():
        verify(item)
    for name, item in plan['code'].items():
        require(verify(item) == HERE / name, 'Executable path mismatch')
    require(set(plan['code']) == {p.name for p in HERE.glob('*.py')}, 'Executable set mismatch')
    freeze = read_json(plan['code_freeze'])
    require(freeze['code'] == plan['code'] and freeze['protocol'] == plan['sources']['protocol'], 'Freeze mismatch')
    go = json.loads(Path(release).read_text())
    require(go['status'] == 'GO' and go['plan'] == pin(path)
            and go['code_freeze'] == plan['code_freeze'], 'Exact root GO required')
    require(read_json(plan['stage1']['author_receipt'])['status'] == 'FULL_FILE_INTEGRITY_AND_VALUES_QUALIFIED_LOAD_UNVERIFIED',
            'Stage1 author qualification not PASS')
    require(read_json(plan['stage1']['independent_receipt'])['status'] == 'PASS', 'Stage1 independent qualification not PASS')
    verify(plan['stage1']['artifact'])
    return plan


def state_receipt(model, expected_state, schema):
    state = model.state_dict()
    expected_schema = {item['name']: item for item in schema['tensor_schema']}
    require(set(state) == set(expected_schema) == set(expected_state), 'Exactly508 state keys required')
    groups = {}
    value_hashes = {}
    named_elements = 0
    for name, value in state.items():
        expected = expected_state[name]
        require(value.dtype == expected.dtype == torch.float32 and value.device.type == 'cpu'
                and list(value.shape) == expected_schema[name]['shape'] == list(expected.shape), 'State tensor shape/dtype mismatch')
        raw = value.detach().contiguous().numpy().tobytes()
        require(raw == expected.detach().contiguous().numpy().tobytes(), 'Loaded parameter bytes differ: ' + name)
        require(bool(torch.isfinite(value).all()), 'Nonfinite loaded tensor')
        storage = value.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes(), value.storage_offset(), tuple(value.shape), tuple(value.stride()))
        groups.setdefault(key, []).append(name)
        value_hashes[name] = hashlib.sha256(raw).hexdigest()
        named_elements += value.numel()
    aliases = sorted([sorted(names) for names in groups.values() if len(names) > 1])
    expected_aliases = sorted([sorted(item['names']) for item in schema['expected_factory_alias_groups']])
    require(aliases == expected_aliases and len(groups) == schema['retained_factory_unique_storage_groups'], 'Factory alias partition differs')
    require(len(state) == schema['expected_named_tensor_count'] and named_elements == schema['expected_named_numel'], 'Named state count differs')
    require(sum(p.numel() for p in model.parameters()) == schema['retained_factory_unique_parameters'], 'Unique parameter count differs')
    digest = hashlib.sha256(json.dumps(value_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(named_tensors=len(state), named_elements=named_elements, alias_pairs=aliases,
        unique_groups=len(groups), unique_parameters=sum(p.numel() for p in model.parameters()),
        all_loaded_bytes_exact=True, all_finite=True, named_value_sha256=value_hashes,
        semantic_value_sha256=digest)


def worker(args):
    started = time.monotonic()
    plan = load_plan(args.plan, args.release)
    limits = plan['limits']
    resource.setrlimit(resource.RLIMIT_CPU, (limits['worker_cpu_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_AS, (limits['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits['stage_output_bytes'],)*2)
    signal.alarm(limits['worker_wall_seconds'])
    require(platform.python_version() == plan['runtime']['python'] and np.__version__ == plan['runtime']['numpy']
            and torch.__version__ == plan['runtime']['torch'] and safetensors.__version__ == plan['runtime']['safetensors']
            and not torch.cuda.is_available(), 'Runtime differs')
    torch.set_num_threads(limits['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    random.seed(plan['seed'])
    directory = Path(args.output) / args.worker
    directory.mkdir(parents=False, exist_ok=False)
    model, normalizer, implementation, config = build_factory(args.worker, plan['sources'])
    construction = {'model_class':type(model).__module__+'.'+type(model).__qualname__,
        'module_classes': {name:type(part).__module__+'.'+type(part).__qualname__ for name,part in model.named_modules()},
        'input_depth': model.input_depth,
        'network_properties': {key: getattr(model.network,key,None) for key in
            ('op_dims','num_stages','in_channels','features_per_stage','n_blocks_per_stage','kernel_sizes',
             'strides','enable_deep_supervision','architecture_type')},
        'module_count':len(list(model.modules()))}
    schema = read_json(plan['sources']['reference_schema'])
    require(schema['expected_named_tensor_count']==508 and schema['expected_named_numel']==68175426
        and schema['retained_factory_unique_storage_groups']==264
        and schema['retained_factory_unique_parameters']==34546498
        and len(schema['expected_factory_alias_groups'])==244, 'Pinned real factory schema differs')
    state = load_file(str(verify(plan['stage1']['artifact'])), device='cpu')
    # Safetensors does no dtype conversion. Stage1 independently proved byte-equal aliases.
    expected_schema = {item['name']: item for item in schema['tensor_schema']}
    require(set(state) == set(expected_schema) and len(state) == 508, 'Native508 keys differ')
    for name, value in state.items():
        require(value.dtype == torch.float32 and list(value.shape) == expected_schema[name]['shape'], 'Native schema differs')
    strict = model.load_state_dict(state, strict=True)
    require(not strict.missing_keys and not strict.unexpected_keys, 'Strict load returned incompatibility')
    loaded = state_receipt(model, state, schema)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    require(all(not part.training for part in model.modules()), 'Model not fully eval')
    write_json(directory / 'load-verification.json', dict(status='PASS', factory=args.worker,
        strict=True, missing_keys=[], unexpected_keys=[], dtype_conversion=False,
        checkpoint_format='safetensors', config=config, construction=construction, **loaded))
    results = []
    model_forwards = [0]
    def count_forward(module, args, output):
        model_forwards[0] += 1
    handle = model.register_forward_hook(count_forward)
    for case in plan['cases']:
        volume = synthetic_volume(*case['shape'], zero=case.get('zero',False))
        before = array_record(volume)
        arrays, metadata = (historical_case if args.worker == 'historical' else current_case)(
            model, normalizer, implementation, case, volume)
        require(before == array_record(volume), 'Synthetic uint8 source mutated')
        results.append(save_case(directory, case['name'], arrays, metadata))
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= limits['rss_bytes'], 'Worker RSS exceeded')
        require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) <= limits['stage_output_bytes'], 'Stage output budget exceeded')
    handle.remove()
    require(model_forwards[0] == plan['forwards_per_worker'] == sum(c['forwards'] for c in results), 'Forward total differs')
    require(all(p.grad is None for p in model.parameters()), 'Gradient materialized')
    require(all(not part.training for part in model.modules()), 'Model mode changed')
    final = state_receipt(model, state, schema)
    require(final == loaded, 'Loaded model state changed')
    write_json(directory/'final-state-verification.json',dict(status='PASS',factory=args.worker,**final))
    for item in plan['sources'].values():
        verify(item)
    closure = read_json(plan['sources']['historical_closure'])
    for item in closure['modules']:
        verify(item)
    for key in ('normalization_module_for_separate_input_parity',
                'historical_padding_module_for_separate_input_parity',
                'historical_inference_source_for_reviewed_pure_extraction', 'authored_configuration'):
        verify(closure[key])
    vendor = read_json(plan['sources']['current_vendor_manifest'])
    vendor_dir = Path(plan['sources']['current_vendor_manifest']['path']).parent
    for item in vendor['files']:
        require(pin(vendor_dir/item['path'])['sha256']==item['sha256'], 'Current source changed')
    require(not any(STUB_COUNTS['runtime_calls'].values()) and not any(FORBIDDEN_COUNTS.values()), 'Forbidden operation attempted')
    result = dict(status='PASS', factory=args.worker, model_constructions=1,
        exact_strict_load=pin(directory/'load-verification.json'), model_forwards=model_forwards[0],
        final_state_verification=pin(directory/'final-state-verification.json'), source_hashes_unchanged=True,
        training_steps=0, gradients=0, cases=results, stub_accounting=STUB_COUNTS,
        forbidden_calls=FORBIDDEN_COUNTS, state_unchanged=True,
        wall_seconds=time.monotonic()-started,
        cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    write_json(directory/'result.json', result)
    print(json.dumps({'status':'PASS','factory':args.worker,'result':pin(directory/'result.json')}))


def controller(args):
    started = time.monotonic()
    plan = load_plan(args.plan, args.release)
    out = Path(args.output)
    out.mkdir(parents=False, exist_ok=False)
    results = []
    for which in ('historical','current'):
        command = [sys.executable, str(Path(__file__).resolve()), '--plan', str(Path(args.plan).resolve()),
            '--release',str(Path(args.release).resolve()),'--output',str(out.resolve()),'--worker',which]
        child_started = time.monotonic()
        with (out/(which+'.log')).open('xb') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4',
                     'MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
            stop = None
            peak_rss = 0
            while child.poll() is None:
                if time.monotonic()-child_started > plan['limits']['worker_external_seconds']:
                    stop = 'worker_external_wall'
                if time.monotonic()-started > plan['limits']['controller_wall_seconds']:
                    stop = 'controller_wall'
                if sum(p.stat().st_size for p in PHASE.rglob('*') if p.is_file()) > plan['limits']['phase_output_bytes']:
                    stop = 'phase_output_bytes'
                if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) > plan['limits']['stage_output_bytes']:
                    stop = 'stage_output_bytes'
                try:
                    status = Path('/proc/%d/status'%child.pid).read_text()
                    rss = next(int(line.split()[1])*1024 for line in status.splitlines() if line.startswith('VmRSS:'))
                    peak_rss = max(peak_rss,rss)
                    if rss > plan['limits']['rss_bytes']:
                        stop = 'worker_rss'
                except (FileNotFoundError,StopIteration):
                    pass
                if stop:
                    child.kill()
                    child.wait()
                    break
                time.sleep(.1)
        item = dict(factory=which, returncode=child.returncode, stop=stop,
            elapsed_seconds=time.monotonic()-child_started, external_peak_rss_bytes=peak_rss)
        results.append(item)
        if stop or child.returncode:
            write_json(out/'STOPPED.json', dict(status='STOPPED', workers=results, no_retry=True))
            raise RuntimeError('Parity worker stopped: '+which)
        item['result'] = pin(out/which/'result.json')
    write_json(out/'controller-result.json', dict(status='PASS',workers=results,
        wall_seconds=time.monotonic()-started, model_forwards=2*plan['forwards_per_worker'],
        model_constructions=2, training_steps=0, no_retry=True))
    print(json.dumps({'status':'PASS','result':pin(out/'controller-result.json')}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--release',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--worker',choices=['historical','current'])
    args=parser.parse_args()
    if args.worker:
        worker(args)
    else:
        controller(args)


if __name__ == '__main__':
    main()
