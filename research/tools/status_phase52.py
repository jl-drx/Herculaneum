"""Read process identity and retained phase52 progress; never starts inference."""
import argparse
import datetime
import hashlib
import itertools
import json
from pathlib import Path


def inspect(study):
    p = Path(study)
    w = p / 'work'
    state_file = w / 'RUN_STATE.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else None
    pid = state['pid'] if state else None
    alive = False
    if state:
        try:
            raw = Path(f'/proc/{pid}/stat').read_text()
            fields = raw.rsplit(')', 1)[1].split()
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
            alive = (fields[0] != 'Z' and fields[19] == str(state['start_ticks'])
                     and str(p / 'worker.py').encode() in cmd)
        except (FileNotFoundError, ProcessLookupError):
            pass
    go_file = p / 'INFERENCE_GO.json'
    go_sha = hashlib.sha256(go_file.read_bytes()).hexdigest() if go_file.exists() else None
    if state and go_sha:
        assert state['go_sha256'] == go_sha, 'Run state/GO mismatch'
    progress_file = w / 'inference/PROGRESS.json'
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else None
    maps, identities = [], set()
    for f in sorted((w / 'inference').glob('*/COMPLETE.json')):
        r = json.loads(f.read_text())
        assert go_sha and r['status'] == 'PASS' and r['go_sha256'] == go_sha, 'Invalid completion receipt'
        identity = (r['model'], r['phase'], r['direction'], r['depth'])
        assert identity not in identities, 'Duplicate completed map identity'
        identities.add(identity)
        maps.append(dict(model=r['model'], phase=r['phase'], direction=r['direction'],
                         depth=r['depth'], seconds=r['seconds'], tiles=r['counts']['model_tiles']))
    expected = set(itertools.product(['dnative', 'hybrid_3d2d-seed42', 'hybrid_3d2d-seed43', 'hecate'],
                                    ['plain', 'padded'], ['forward', 'reverse'], [-15, 15]))
    assert identities <= expected, 'Unexpected completed map identity'
    result_file = p / 'INFERENCE_RESULTS.json'
    final = json.loads(result_file.read_text()) if result_file.exists() else None
    complete = bool(final and final['status'] == 'PASS'
                    and final['all_saved_logit_replays_exact'] and identities == expected)
    status = 'COMPLETE_RECEIPT_PRESENT' if complete else 'RUNNING' if alive else (
        'STOPPED_INCOMPLETE' if state else 'NO_LOCAL_PROCESS_STATE_INCOMPLETE')
    backup = Path('/workspace/scratch/phase52-backup/STATUS.json')
    return dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                status=status, worker_identity_live=alive, local_process_state_present=bool(state), pid=pid,
                completed_maps=len(maps), expected_maps=32,
                committed_batch_receipts=len(list((w / 'inference').glob('*/batch-*.json'))),
                progress=progress, completed=maps,
                last_recorded_backup=json.loads(backup.read_text()) if backup.exists() else None,
                scope='Process identity and receipt inventory only. Missing local PID state is expected after a file-only restore; never infer a live worker from saved progress. Not a full output/hash verification or platform-availability prediction.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('/workspace/herculaneum-lab/reports/fiftysecond-experiment'))
    print(json.dumps(inspect(parser.parse_args().study), indent=2))


if __name__ == '__main__':
    main()
