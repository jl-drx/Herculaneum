#!/usr/bin/env python3
"""Execute the frozen controls sequentially after the parent launch gate."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT = Path('/workspace/herculaneum-lab')
PLAN = ROOT / 'reports/third-experiment/fixed-controls-plan.json'
EXPECTED_PLAN_SHA = '39c5a2f5cda2051a6c4d173f182f7d62607dd1249739ee3bf50311df05142b75'
DEST = ROOT / 'reports/third-experiment/fixed-controls-execution'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    if sha256(PLAN) != EXPECTED_PLAN_SHA:
        raise ValueError('Frozen plan changed.')
    if not (ROOT / 'reports/third-experiment/ink9um-large-forward/metadata.json').is_file():
        raise ValueError('Primary completed metadata is required before controls.')
    if DEST.exists():
        raise ValueError('Execution records already exist; do not overwrite them.')
    plan = json.loads(PLAN.read_text())
    DEST.mkdir(parents=True, exist_ok=False)
    records = []
    for condition in plan['runs']:
        for item in list(plan['source_inputs'].values()) + [plan['runner'], plan['checkpoint']]:
            if sha256(Path(item['path'])) != item['sha256']:
                raise ValueError('Frozen source, runner or checkpoint changed.')
        if Path(condition['output_dir']).exists():
            raise ValueError('A planned output directory already exists.')
        label = condition['condition']
        log_path = DEST / f'{label}.log'
        started_utc, started = now(), time.perf_counter()
        print(f'Starting {label} sequentially with four CPU threads.', flush=True)
        with log_path.open('x') as log:
            result = subprocess.run(condition['argv'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record = {'condition': label, 'argv': condition['argv'], 'started_at_utc': started_utc, 'completed_at_utc': now(), 'wall_seconds': time.perf_counter() - started, 'exit_code': result.returncode, 'log_path': str(log_path), 'log_sha256': sha256(log_path), 'frozen_plan_sha256': EXPECTED_PLAN_SHA, 'execution_code_sha256': sha256(Path(__file__))}
        meta = Path(condition['output_dir']) / 'metadata.json'
        if meta.is_file():
            record['output_metadata'] = {'path': str(meta), 'sha256': sha256(meta)}
        (DEST / f'{label}.json').write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        print(f'Finished {label}: exit {result.returncode}, wall {record["wall_seconds"]:.2f} seconds.', flush=True)
        if result.returncode:
            raise RuntimeError(f'Control failed; retained log: {log_path}')
    for item in list(plan['source_inputs'].values()) + [plan['runner'], plan['checkpoint']]:
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('Frozen source changed during control execution.')
    (DEST / 'summary.json').write_text(json.dumps({'completed_at_utc': now(), 'frozen_plan_sha256': EXPECTED_PLAN_SHA, 'runs': records, 'source_hashes_unchanged': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()
