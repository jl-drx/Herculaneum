"""Three sequential offline selectors; no retries, fragment replacements or model calls."""
from pathlib import Path
import argparse
import json
import signal
import socket
import subprocess
import sys
import time
from label_common import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--acquisition', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    acquisition = Path(args.acquisition).resolve(strict=True)
    root = Path(args.output_dir).resolve()
    require(not root.exists(), 'Fresh selection controller directory required')
    root.mkdir(parents=True)
    def no_network(*a, **k):
        raise RuntimeError('Controller network is disabled')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = no_network
    started = time.monotonic()
    result = dict(schema_version=1, status='STARTED', plan=record(args.plan),
                  acquisition=record(acquisition), workers=[], selected=[], automatic_retries=0,
                  network_requests=0, model_calls=0, CT_pixel_decodes=0)
    try:
        for fragment in ('frag1', 'frag2', 'frag3'):
            remaining = LIMITS['selection_wall_seconds']-(time.monotonic()-started)
            require(remaining > 0, 'Combined selection wall cap exhausted')
            argv = [sys.executable, str(ROOT/'select_labels.py'), '--plan', str(Path(args.plan).resolve()),
                    '--acquisition', str(acquisition), '--fragment', fragment,
                    '--output-dir', str(root/fragment)]
            worker = {'fragment': fragment, 'argv': argv, 'started_seconds': time.monotonic()-started}
            result['workers'].append(worker)
            with (root/(fragment+'-stdout.log')).open('wb') as out, (root/(fragment+'-stderr.log')).open('wb') as err:
                run = subprocess.run(argv, stdout=out, stderr=err,
                                     timeout=min(LIMITS['worker_wall_seconds']+5, remaining), check=False)
            worker['returncode'] = run.returncode
            worker['result'] = record(root/fragment/'selection-result.json')
            require(run.returncode == 0, 'Fragment selection stopped; no substitution/retry')
            receipt = json.loads(verify(worker['result']).read_text())
            require(receipt['status'] == 'PASS' and len(receipt['selected']) == 2, 'Fragment construction incomplete')
            result['selected'].extend(receipt['selected'])
            require(directory_bytes(root) <= LIMITS['selection_output_bytes'], 'Selection output cap exceeded')
            require(directory_bytes(root)+directory_bytes(acquisition.parent) <= LIMITS['total_output_bytes'],
                    'Combined acquisition/selection output cap exceeded')
        load_plan(args.plan)
        verify(result['acquisition'])
        result.update(status='PASS', selected_context_count=6, selected_array_count=24,
                      independence='Three physical fragments, two parents; evaluation fragments share Paris2',
                      CT_ready=False, benchmark_predictions=0)
    except Exception as error:
        result.update(status='INCOMPLETE', failure_type=type(error).__name__)
        raise
    finally:
        result['wall_seconds'] = time.monotonic()-started
        save(root/'panel-provenance.json', result)
    print(json.dumps({'status': 'PASS', 'provenance': record(root/'panel-provenance.json')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'STOP', 'failure_type': type(error).__name__}))
        raise SystemExit(1)
