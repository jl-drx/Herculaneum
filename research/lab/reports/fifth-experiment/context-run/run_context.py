"""Execute frozen post hoc context conditions on CPU, without retries or downloads."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from acquire_context_v2 import require, sha, verify_inputs, validate_decision

LAB = Path('/workspace/herculaneum-lab')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--runs-dir', type=Path, required=True)
    parser.add_argument('--inference-plan-sha256', required=True)
    args = parser.parse_args()
    plan_path = Path(__file__).with_name('inference-plan.json')
    require(sha(plan_path) == args.inference_plan_sha256, 'Frozen inference plan SHA differs')
    plan = json.loads(plan_path.read_text())
    verify_inputs(plan['verified_inputs'])
    require(plan['device'] == 'cpu' and plan['threads'] == 4 and plan['automatic_retries'] == 0
            and plan['per_condition_timeout_seconds'] == 120
            and plan['maximum_scheduled_tiles'] == 605, 'Frozen CPU bounds differ')
    require(not args.runs_dir.exists(), 'Preserve existing runs; choose fresh output')
    provenance_path = args.data_dir / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    require(provenance['inference_plan_sha256'] == args.inference_plan_sha256
            and provenance['frozen_plan_sha256'] == plan['acquisition_plan']['sha256']
            and sha(plan['acquisition_plan']['path']) == provenance['frozen_plan_sha256'],
            'Acquisition used a different frozen plan')
    require(provenance['shape_zyx'] == [31, 768, 768]
            and provenance['bbox_source_xywh'] == [768, 256, 768, 768], 'Wrong acquired context')
    validate_decision(Path(provenance['context_decision_path']), provenance['context_decision_sha256'],
                      Path(plan['acquisition_plan']['path']), plan['acquisition_plan']['sha256'])
    require(provenance['context_decision_sha256'] == plan['context_decision']['sha256'], 'Decision pin differs')
    source_records = [dict(path=str(args.data_dir / name), sha256=record['sha256'],
                          size_bytes=(args.data_dir / name).stat().st_size)
                      for name, record in provenance['derived_outputs'].items()]
    verify_inputs(source_records)
    args.runs_dir.mkdir(parents=True, exist_ok=False)
    record = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        inference_plan_path=str(plan_path.resolve()), inference_plan_sha256=args.inference_plan_sha256,
        source_provenance_path=str(provenance_path.resolve()), source_provenance_sha256=sha(provenance_path),
        execution_code_sha256=sha(__file__), context_decision_path=provenance['context_decision_path'],
        context_decision_sha256=provenance['context_decision_sha256'], conditions=[], completed=False,
        paid_resources_created=False, letter=None, transcription=None, translation=None)
    result_path = args.runs_dir / 'execution.json'
    result_path.write_text(json.dumps(record, indent=2) + '\n')
    for condition in plan['conditions']:
        name = condition['name']
        command = [sys.executable, str(LAB / 'run_ink_baseline.py'),
            str(args.data_dir / 'surface.npy'), '--provenance', str(provenance_path),
            '--mask', str(args.data_dir / 'geometry-mask.npy'), '--output-dir', str(args.runs_dir / name),
            '--layer-start', str(condition['layer_start']), '--direction', condition['direction'], '--threads', '4']
        for seed in condition['seeds']:
            command += [f'--{seed}-checkpoint', str(LAB / 'data/models' / f'ink9um-{seed}-step075000.pth')]
        entry = dict(name=name, argv=command, started_at_utc=datetime.now(timezone.utc).isoformat())
        record['conditions'].append(entry)
        result_path.write_text(json.dumps(record, indent=2) + '\n')
        started = time.monotonic()
        with (args.runs_dir / f'{name}.log').open('w') as log:
            try:
                result = subprocess.run(command, cwd=LAB, stdout=log, stderr=subprocess.STDOUT, timeout=120)
                entry['returncode'] = result.returncode
            except subprocess.TimeoutExpired:
                entry['returncode'] = 'timeout'
        entry['wall_seconds'] = time.monotonic() - started
        result_path.write_text(json.dumps(record, indent=2) + '\n')
        require(entry['returncode'] == 0, f'{name} failed; retained log, no automatic retry')
        entry['metadata_sha256'] = sha(args.runs_dir / name / 'metadata.json')
        print(f'Completed {name}: {entry["wall_seconds"]:.2f}s', flush=True)
    require(sha(provenance_path) == record['source_provenance_sha256'], 'Source provenance changed during inference')
    require(sha(plan_path) == args.inference_plan_sha256, 'Inference plan changed during execution')
    verify_inputs(source_records)
    verify_inputs(plan['verified_inputs'])
    record['completed'] = True
    record['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    result_path.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
