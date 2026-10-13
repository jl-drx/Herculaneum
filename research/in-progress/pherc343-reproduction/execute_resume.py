"""Supervise the approved fixed reproduction; stop on failure, retain live status.

No credentials, network requests, paid resources, training, or model selection.
Completed computation still requires scientific review before sealing the study.
"""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
W = P / 'work'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def pin(path):
    path = Path(path)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def atomic_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    lock = (W / 'fixed-pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for gate in ['RENDER_VERIFICATION.json', 'INFERENCE_QUALIFICATION.json']:
        assert json.loads((P / gate).read_text())['status'] == 'PASS', gate
    assert shutil.disk_usage(W).free > 5 * 1024**3
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = W / ('pipeline-' + stamp)
    run.mkdir()
    paths = ['PROTOCOL.md', 'RENDER_VERIFICATION.json', 'INFERENCE_QUALIFICATION.json',
             'FIXED_DISPLAY_STATS.json', 'run_fixed_depth.py', 'compose_and_compare.py',
             'execute_pipeline.py', 'run_fixed_depth_resume.py', 'execute_resume.py']
    frozen = {name: pin(P / name) for name in paths}
    atomic_json(run / 'INPUTS.json', frozen)
    state = {'status': 'RUNNING', 'started_at_utc': now(), 'supervisor_pid': os.getpid(),
             'run_directory': str(run), 'gpu_spend_GBP': 0, 'stages': [],
             'scope': 'Fixed d=+1 four-model reading-side reproduction; review still required.'}

    def save():
        state['updated_at_utc'] = now()
        atomic_json(run / 'STATUS.json', state)
        atomic_json(P / 'PIPELINE_STATUS.json', state)

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4')
    save()
    for script in ['run_fixed_depth_resume.py', 'compose_and_compare.py']:
        try:
            for name, identity in frozen.items():
                assert pin(P / name) == identity, f'Frozen input changed: {name}'
            record = {'script': script, 'started_at_utc': now(),
                      'log': str(run / (script + '.log'))}
            state['stages'].append(record)
            with Path(record['log']).open('xb') as output:
                child = subprocess.Popen([sys.executable, '-B', str(P / script)], cwd=P,
                                         env=env, stdout=output, stderr=subprocess.STDOUT)
                record['pid'] = child.pid
                save()
                while child.poll() is None:
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        record['log_bytes'] = Path(record['log']).stat().st_size
                        save()
                record['exit_code'] = child.returncode
            record['finished_at_utc'] = now()
            if record['exit_code']:
                raise RuntimeError(f'{script} exited {record["exit_code"]}; inspect its retained log')
            save()
        except Exception as exc:
            state['status'] = 'FAILED_REQUIRES_INSPECTION'
            state['error'] = str(exc)
            save()
            raise
    state['status'] = 'COMPUTE_COMPLETE_AWAITING_REVIEW'
    state['finished_at_utc'] = now()
    state['outputs'] = [pin(P / name) for name in ['FIXED_DEPTH_INFERENCE.json', 'IMAGE_COMPARISON.json']]
    save()
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
