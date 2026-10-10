"""Execute the four predeclared CPU conditions without retries or new downloads."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

LAB = Path('/workspace/herculaneum-lab')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--runs-dir', type=Path, required=True)
    args = ap.parse_args()
    plan_path = Path(__file__).with_name('inference-plan.json')
    plan = json.loads(plan_path.read_text())
    for record in plan['verified_inputs']:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Inference source/model changed after plan freeze.')
    if args.runs_dir.exists():
        raise ValueError('Preserve existing runs; choose fresh output.')
    prov = args.data_dir / 'provenance.json'
    if not prov.is_file():
        raise ValueError('Complete acquisition provenance is required.')
    args.runs_dir.mkdir(parents=True, exist_ok=False)
    record = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        inference_plan_path=str(plan_path.resolve()), inference_plan_sha256=sha(plan_path),
        source_provenance_path=str(prov.resolve()), source_provenance_sha256=sha(prov),
        execution_code_sha256=sha(__file__), conditions=[], completed=False,
        paid_resources_created=False, transcription=None, translation=None)
    result_path = args.runs_dir / 'execution.json'
    result_path.write_text(json.dumps(record, indent=2) + '\n')
    for condition in plan['conditions']:
        label = condition['name']
        command = [sys.executable, str(LAB / 'run_ink_baseline.py'),
            str(args.data_dir / 'surface.npy'), '--provenance', str(prov),
            '--mask', str(args.data_dir / 'geometry-mask.npy'),
            '--output-dir', str(args.runs_dir / label),
            '--layer-start', str(condition['layer_start']),
            '--direction', condition['direction'], '--threads', '4']
        for seed in condition['seeds']:
            command += [f'--{seed}-checkpoint', str(LAB / 'data/models' / f'ink9um-{seed}-step075000.pth')]
        started = time.monotonic()
        entry = dict(name=label, argv=command, started_at_utc=datetime.now(timezone.utc).isoformat())
        record['conditions'].append(entry)
        result_path.write_text(json.dumps(record, indent=2) + '\n')
        with (args.runs_dir / f'{label}.log').open('w') as log:
            try:
                result = subprocess.run(command, cwd=LAB, stdout=log, stderr=subprocess.STDOUT, timeout=120)
                entry['returncode'] = result.returncode
            except subprocess.TimeoutExpired:
                entry['returncode'] = 'timeout'
        entry['wall_seconds'] = time.monotonic() - started
        result_path.write_text(json.dumps(record, indent=2) + '\n')
        if entry['returncode'] != 0:
            raise RuntimeError(f'{label} failed; retained log, no automatic retry.')
        metadata = args.runs_dir / label / 'metadata.json'
        entry['metadata_sha256'] = sha(metadata)
        print(f'Completed {label}: {entry["wall_seconds"]:.2f}s', flush=True)
    if sha(prov) != record['source_provenance_sha256']:
        raise ValueError('Acquisition provenance changed during inference.')
    for item in plan['verified_inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen model/source changed during inference.')
    record['completed'] = True
    record['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    result_path.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
