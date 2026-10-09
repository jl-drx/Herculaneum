"""Root-only bounded controller for the two frozen first-step diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan', type=Path, required=True)
parser.add_argument('--plan-sha256', required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
plan_record = record(args.plan)
assert plan_record['sha256'] == args.plan_sha256
plan = json.loads(args.plan.read_text())
assert plan['ready_for_execution'] is True
assert record(__file__) in plan['reviews'], 'Controller must be bound before execution'
worker = Path(__file__).resolve().parent / 'probe_worker.py'
assert record(worker) == plan['code']['probe_worker.py']
assert plan['limits']['total_wall_seconds'] == 300
args.output_dir.mkdir(parents=True, exist_ok=False)
started = time.perf_counter()
workers = []
status = 'PASS'
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
           MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
for objective in ['A', 'B']:
    remaining = 300 - (time.perf_counter() - started)
    if remaining <= 0:
        status = 'STOP'
        break
    command = [sys.executable, str(worker), '--plan', str(args.plan),
               '--plan-sha256', args.plan_sha256, '--objective', objective,
               '--output-dir', str(args.output_dir / objective)]
    before = time.perf_counter()
    timed_out = False
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True,
                                timeout=min(125, remaining))
        returncode, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = None
        stdout = error.stdout or b''
        stderr = error.stderr or b''
        if isinstance(stdout, bytes):
            stdout = stdout.decode('utf-8', errors='replace')
        if isinstance(stderr, bytes):
            stderr = stderr.decode('utf-8', errors='replace')
    (args.output_dir / (objective + '-stdout.txt')).write_text(stdout)
    (args.output_dir / (objective + '-stderr.txt')).write_text(stderr)
    entry = dict(objective=objective, returncode=returncode, timed_out=timed_out,
                 wall_seconds=time.perf_counter()-before)
    if returncode == 0:
        receipt = args.output_dir / objective / 'result.json'
        value = json.loads(receipt.read_text())
        assert value['status'] == 'PASS' and value['plan'] == plan_record
        assert value['actual_calls'] == dict(model_forwards=4, backwards=1,
                                            independent_fresh_optimizer_steps=3)
        entry['result'] = record(receipt)
    else:
        status = 'STOP'
    workers.append(entry)
    print(json.dumps(entry), flush=True)
    if status != 'PASS':
        break
elapsed = time.perf_counter() - started
output_bytes = sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())
assert elapsed <= 300 and output_bytes < 64*1024**2
assert record(args.plan) == plan_record and record(worker) == plan['code']['probe_worker.py']
receipt = dict(status=status, plan=plan_record, controller=record(__file__), workers=workers,
               wall_seconds=elapsed, output_bytes_before_controller_receipt=output_bytes,
               retries=0, checkpoint_outputs=0, network_calls=0)
(args.output_dir / 'run-results.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
print(json.dumps(dict(status=status, receipt=record(args.output_dir / 'run-results.json'))))
raise SystemExit(0 if status == 'PASS' and len(workers) == 2 else 1)
