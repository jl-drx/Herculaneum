#!/usr/bin/env python3
"""Bounded four-arm CPU comparison; inference is gated by a reviewed frozen plan."""
from __future__ import annotations

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

from comparison_common import (ARMS, class_counts, import_helper, load_data, load_plan,
                               output_bytes, record, require, sha, verify_record, write_json)


def worker(plan_path, label, output):
    started = time.perf_counter()
    plan, plan_path = load_plan(plan_path, execute=True)
    require(label in ARMS, 'Unknown worker arm')
    arrays, _, _, _, preflight = load_data(plan)
    artifact = next(item for item in plan['models'] if item['label'] == label)
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    # A per-file bound supplements the aggregate output check; no memory/GPU claim.
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    def blocked_network(*args, **kwargs):
        raise ValueError('Offline comparison attempted network access; stopped without retry')
    socket.socket.connect = blocked_network
    socket.socket.connect_ex = blocked_network
    socket.socket.sendto = blocked_network
    socket.create_connection = blocked_network
    socket.getaddrinfo = blocked_network
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    helper = import_helper(plan)
    runtime = helper.load_runtime()
    model, model_metadata = helper.build_model(artifact['path'], artifact, runtime)
    # The unchanged adapter receives no annotation/occupancy/geometry mask.
    scores, inference_metadata, evaluated = runtime['reference'].infer_surface(
        model, arrays['surface'], 2, 'forward', runtime['normalize_robust'], valid_mask=None)
    require(scores.dtype == np.float32 and scores.shape == arrays['surface'].shape[1:]
            and evaluated.dtype == np.bool_ and evaluated.shape == scores.shape,
            'Unexpected inference array contract')
    require(np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)), 'Invalid raw score map')
    np.save(output/'scores.npy', scores, allow_pickle=False)
    np.save(output/'evaluated-mask.npy', evaluated, allow_pickle=False)
    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'torchvision', 'timm', 'numpy', 'einops']}
    metadata = dict(label=label, plan=record(plan_path), inputs=plan['inputs'],
        model=model_metadata, inference=inference_metadata, preflight=preflight,
        raw_scores=record(output/'scores.npy'), evaluated_mask=record(output/'evaluated-mask.npy'),
        source_contract=plan['data'], execution_code=plan['code'],
        model_contract_helper=plan['model_contract_helper'],
        runtime=dict(device='cpu', precision='float32', batch_size=1, threads=4, interop_threads=1,
                     deterministic_algorithms=True, package_versions=versions,
                     worker_wall_seconds=time.perf_counter()-started,
                     peak_resident_memory_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),
        canvas_mask_semantics='valid_mask=None creates an all-ones canvas only. The adapter valid_pixel_count is not authenticated geometry support.',
        output_semantics='Unquantized float32 sigmoid model scores, unchanged robust-MAD tiled inference; not calibrated confidence or ink authenticity.')
    write_json(output/'metadata.json', metadata)
    print(json.dumps(dict(label=label, status='complete', inference_seconds=inference_metadata['inference_seconds'],
        worker_wall_seconds=metadata['runtime']['worker_wall_seconds'], scored_tiles=inference_metadata['scored_tile_count'])))


def run(plan_path, output, execute):
    plan, plan_path = load_plan(plan_path, execute=execute)
    arrays, labels, primary, secondary, preflight = load_data(plan)
    helper = import_helper(plan)
    for artifact in plan['models']:
        helper.verify_artifact(artifact['path'], artifact)
    if not execute:
        print(json.dumps(dict(status='preflight_passed_no_inference', plan_sha256=sha(plan_path), preflight=preflight), indent=2))
        return
    require(output is not None, '--output-dir is required for execution')
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    (output/'frozen-plan.json').write_bytes(plan_path.read_bytes())
    np.save(output/'primary-before-predictions-mask.npy', primary, allow_pickle=False)
    np.save(output/'secondary-before-predictions-mask.npy', secondary, allow_pickle=False)
    write_json(output/'preflight.json', dict(plan=record(plan_path), inputs=plan['inputs'], **preflight))
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    runs = []
    try:
        for label in ARMS:
            log_path = output/f'{label}.log'
            started = time.perf_counter()
            with log_path.open('w') as log:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--plan', str(plan_path),
                    '--worker', label, '--output-dir', str(output/label)], stdout=log, stderr=subprocess.STDOUT,
                    env=env, timeout=plan['limits']['per_model_timeout_seconds'], check=False)
            runs.append(dict(label=label, exit_code=process.returncode, wall_seconds=time.perf_counter()-started,
                             log=record(log_path)))
            write_json(output/'execution.json', dict(status='running', plan=record(plan_path), runs=runs))
            require(process.returncode == 0, f'STOP: {label} failed; no automatic retry')
            require(output_bytes(output) <= plan['limits']['maximum_output_bytes'], 'STOP: output byte budget exceeded')
        from analyze_comparison import analyze
        analysis = analyze(plan_path, output)
        # Recheck frozen identities after analysis, before claiming completion.
        final_plan, _ = load_plan(plan_path, execute=True)
        require((output/'frozen-plan.json').read_bytes() == plan_path.read_bytes(), 'Frozen plan changed during execution')
        load_data(final_plan)
        for artifact in final_plan['models']:
            helper.verify_artifact(artifact['path'], artifact)
        write_json(output/'execution.json', dict(status='complete', plan=record(plan_path), runs=runs,
                    analysis=analysis, output_bytes=output_bytes(output)))
        require(output_bytes(output) <= plan['limits']['maximum_output_bytes'], 'STOP: output byte budget exceeded')
        print(json.dumps(dict(status='complete', output_dir=str(output), analysis=analysis)))
    except Exception as error:
        write_json(output/'execution.json', dict(status='stopped_no_retry', plan=record(plan_path), runs=runs,
            reason=str(error), exception_type=type(error).__name__, output_bytes=output_bytes(output)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--execute', action='store_true', help='Requires root-reviewed ready_for_execution=true frozen plan')
    parser.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        require(args.output_dir is not None, 'Worker requires fresh output directory')
        worker(args.plan, args.worker, args.output_dir)
    else:
        run(args.plan, args.output_dir, args.execute)


if __name__ == '__main__':
    main()
