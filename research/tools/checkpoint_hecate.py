"""Protect the active PHerc343 Hecate map without changing its process.

Only fully accumulated rows are saved. Resume replays overlapping patches in
their original global batches, then uses the saved rows verbatim. No live
process memory access, signals, new inference, or source edits during capture.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

SOURCE_SHA = 'c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59'
ACTIVE_RUNNER_SHA = '102620e87ece702dcb8caf958731ae18b5e3bca94c0cf05cb4f4d5264529a3cb'
STEM = 'concat_w047-w048_R5B2_z9500-11000_66_padded__hecate__L66zc33.50_reverse.png'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(obj, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def starts(length):
    return list(range(0, max(0, length - 64), 32)) + [max(0, length - 64)]


def geometry(shape):
    return [(y, x) for y in starts(shape[0]) for x in starts(shape[1])]


def cut_after(origins, completed, height):
    if not 0 <= completed <= len(origins):
        raise ValueError('Invalid completed origin count')
    return height if completed == len(origins) else origins[completed][0]


def resume_origin(origins, prefix_rows, height):
    if prefix_rows == height:
        return len(origins)
    first = next(i for i, (y, _) in enumerate(origins) if y + 64 > prefix_rows)
    return first // 4 * 4  # Preserve original batch membership, including blanks.


def load_hecate(lab):
    path = lab / 'reports/fortyfifth-experiment/source-evidence/hecate-hecate.py'
    if sha(path) != SOURCE_SHA:
        raise ValueError('Upstream Hecate source identity changed')
    spec = importlib.util.spec_from_file_location('checkpoint_hecate_upstream', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def resumable_predict(module):
    """Derive only loop bounds and a post-batch callback from pinned source."""
    source = inspect.getsource(module.predict)
    changes = {
        "precision='fp32', three_d=False):":
            "precision='fp32', three_d=False, resume_start=0, prefix_rows=0, checkpoint_callback=None):",
        'for start in range(0, len(origins), batch_size):':
            'for start in range(resume_start, len(origins), batch_size):',
        '        if three_d and model.margin:':
            '            if checkpoint_callback is not None:\n'
            '                checkpoint_callback(min(start+batch_size, len(origins)), numerator, denominator)\n'
            '        if three_d and model.margin:',
        'for y in range(0,h,128):': 'for y in range(prefix_rows,h,128):',
    }
    for old, new in changes.items():
        if source.count(old) != 1:
            raise ValueError('Unexpected upstream loop; refuse to modify it')
        source = source.replace(old, new)
    scope = dict(vars(module))
    exec(compile(source, '<qualified-hecate-resume>', 'exec'), scope)
    return scope['predict']


def binding(lab):
    phase = lab / 'reports/fortysixth-experiment'
    volume = phase / 'work/hecate-padded.zarr'
    qualification = json.loads((phase / 'INFERENCE_QUALIFICATION.json').read_text())
    checkpoint = next(r['checkpoint'] for r in qualification['models'] if r['model'] == 'hecate')
    model = lab / Path(checkpoint['path']).relative_to('/workspace/herculaneum-lab')
    if sha(model) != checkpoint['sha256']:
        raise ValueError('Checkpoint identity mismatch')
    paths = [p for p in volume.rglob('*') if p.is_file()]
    paths += [model, phase / 'PROTOCOL.md', phase / 'FIXED_DISPLAY_STATS.json',
              phase / 'INFERENCE_QUALIFICATION.json', phase / 'compose_and_compare.py',
              lab / 'reports/fortyfifth-experiment/source-evidence/hecate-hecate.py']
    files = {str(p.relative_to(lab)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(paths)}
    shape = json.loads((volume / '0/zarr.json').read_text())['shape']
    if shape != [16, 2340, 6228] or sha(paths[-1]) != SOURCE_SHA:
        raise ValueError('Unsupported input geometry/source')
    return {'schema': 1, 'shape': shape[1:], 'patch': 64, 'stride': 32, 'batch_size': 4,
            'reverse': True, 'precision': 'fp32', 'threads': 4, 'source_sha256': SOURCE_SHA,
            'checkpoint_sha256': checkpoint['sha256'], 'files': files}


def binding_sha(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def check_binding(lab, identity):
    for relative, record in identity['files'].items():
        p = lab / relative
        if p.stat().st_size != record['bytes'] or sha(p) != record['sha256']:
            raise ValueError('Resume identity mismatch: ' + relative)


def publish(root, identity, numerator, denominator, cut, completed, base=None):
    """Publish an immutable generation, then atomically advance LATEST.json."""
    old = json.loads((root / 'LATEST.json').read_text()) if (root / 'LATEST.json').exists() else None
    if cut <= (old['rows'] if old else 0):
        return None
    lower = 0 if base is None else len(base)
    if not lower <= cut <= identity['shape'][0]:
        raise ValueError('Invalid stable row extent')
    result = np.empty((cut, identity['shape'][1]), dtype=np.uint8)
    if base is not None:
        result[:lower] = base
    n = np.asarray(numerator[lower:cut]).copy()
    d = np.asarray(denominator[lower:cut]).copy()
    if not np.isfinite(n).all() or not np.isfinite(d).all() or (d <= 0).any():
        raise ValueError('Invalid completed-row accumulators')
    # The worker has moved to origins whose y >= cut. These rows are immutable.
    if not np.array_equal(n, numerator[lower:cut]) or not np.array_equal(d, denominator[lower:cut]):
        raise ValueError('Claimed completed rows changed during capture')
    result[lower:] = np.rint((n / d).clip(0, 1) * 255).astype(np.uint8)
    directory = Path(tempfile.mkdtemp(prefix='.partial-', dir=root))
    with (directory / 'prefix.npy').open('wb') as f:
        np.save(f, result, allow_pickle=False); f.flush(); os.fsync(f.fileno())
    record = {'schema': 1, 'rows': cut, 'shape': list(result.shape), 'dtype': 'uint8',
              'completed_origins_observed': completed, 'total_origins': len(geometry(identity['shape'])),
              'binding_sha256': binding_sha(identity), 'prefix_sha256': sha(directory / 'prefix.npy'),
              'saved_at_utc': datetime.now(timezone.utc).isoformat(),
              'resume_origin': resume_origin(geometry(identity['shape']), cut, identity['shape'][0])}
    atomic_json(directory / 'MANIFEST.json', record)
    generation = f'rows-{cut:05d}'
    os.rename(directory, root / generation)
    atomic_json(root / 'LATEST.json', {'generation': generation, **record})
    print(json.dumps({'checkpoint_rows': cut, 'height': identity['shape'][0],
                      'completed_origins': completed, 'resume_origin': record['resume_origin']}), flush=True)
    return record


def read_latest(root, identity):
    latest = json.loads((root / 'LATEST.json').read_text())
    name = latest['generation']
    if name != f"rows-{latest['rows']:05d}":
        raise ValueError('Invalid generation path')
    record = json.loads((root / name / 'MANIFEST.json').read_text())
    if latest != {'generation': name, **record} or record['binding_sha256'] != binding_sha(identity):
        raise ValueError('Checkpoint manifest/binding mismatch')
    path = root / name / 'prefix.npy'
    if sha(path) != record['prefix_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    prefix = np.load(path, allow_pickle=False)
    if prefix.dtype != np.uint8 or list(prefix.shape) != [record['rows'], identity['shape'][1]]:
        raise ValueError('Checkpoint array mismatch')
    expected = resume_origin(geometry(identity['shape']), len(prefix), identity['shape'][0])
    if record['resume_origin'] != expected:
        raise ValueError('Checkpoint resume cursor mismatch')
    return record, prefix


def watch(lab, root):
    phase = lab / 'reports/fortysixth-experiment'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = binding(lab)
        if (root / 'BINDING.json').exists():
            if json.loads((root / 'BINDING.json').read_text()) != identity:
                raise ValueError('Existing checkpoint binding differs')
        else:
            atomic_json(root / 'BINDING.json', identity)
        s = json.loads((phase / 'PIPELINE_STATUS.json').read_text())
        runner = phase / 'run_fixed_depth_resume.py'
        if sha(runner) != ACTIVE_RUNNER_SHA or s['stages'][-1]['script'] != runner.name:
            raise ValueError('Watcher only supports the verified active runner')
        pid = s['stages'][-1]['pid']; proc = Path('/proc') / str(pid)
        cmdline = (proc / 'cmdline').read_bytes()
        if str(runner).encode() not in cmdline:
            raise ValueError('Worker PID does not identify the expected runner')
        start_time = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()[19]
        started = datetime.fromisoformat(s['started_at_utc']).timestamp()
        size = int(np.prod(identity['shape'])) * 4
        candidates = [d for d in Path('/tmp').glob('hecate-*') if d.is_dir()
                      and all((d / x).is_file() and not (d / x).is_symlink()
                              and (d / x).stat().st_size == size
                              and (d / x).stat().st_mtime >= started for x in ['sum', 'weight'])]
        if len(candidates) != 1:
            raise ValueError('Cannot uniquely identify this run’s temporary accumulators')
        accumulators = candidates[0]
        n = np.memmap(accumulators / 'sum', mode='r', dtype='float32', shape=tuple(identity['shape']))
        d = np.memmap(accumulators / 'weight', mode='r', dtype='float32', shape=tuple(identity['shape']))
        origins = geometry(identity['shape'])
        atomic_json(root / 'WATCHER.json', {'pid': os.getpid(), 'worker_pid': pid,
                    'worker_start_ticks': start_time, 'worker_run_directory': s['run_directory'],
                    'temporary_arrays': str(accumulators), 'method': 'Read-only completed-row capture; no worker signals or edits.'})
        while proc.exists():
            if (proc / 'stat').read_text().rsplit(') ', 1)[1].split()[19] != start_time:
                break
            progress = re.findall(r'(\d+)/(\d+) patches', Path(s['stages'][-1]['log']).read_text())
            if progress:
                completed, total = map(int, progress[-1])
                if total != len(origins) or (completed != total and completed % 4):
                    raise ValueError('Unexpected global tile progress')
                cut = cut_after(origins, completed, identity['shape'][0])
                publish(root, identity, n, d, cut, completed)
            atomic_json(root / 'WATCHER_STATUS.json', {'status': 'WATCHING', 'updated_at_utc': datetime.now(timezone.utc).isoformat()})
            if (phase / 'work/maps_padded' / (STEM + '.receipt.json')).exists():
                break
            time.sleep(30)
        atomic_json(root / 'WATCHER_STATUS.json', {'status': 'STOPPED_WORKER_ENDED', 'updated_at_utc': datetime.now(timezone.utc).isoformat()})


def resume(lab, root):
    import torch
    phase = lab / 'reports/fortysixth-experiment'
    with (phase / 'work/fixed-pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = json.loads((root / 'BINDING.json').read_text()); check_binding(lab, identity)
        record, prefix = read_latest(root, identity)
        output = phase / 'work/maps_padded' / STEM
        if output.exists() or Path(str(output) + '.receipt.json').exists():
            raise ValueError('Inspect existing final output before resuming')
        torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        hecate = load_hecate(lab)
        qualification = json.loads((phase / 'INFERENCE_QUALIFICATION.json').read_text())
        model_record = next(r for r in qualification['models'] if r['model'] == 'hecate')
        checkpoint = lab / Path(model_record['checkpoint']['path']).relative_to('/workspace/herculaneum-lab')
        model = hecate.load_model(checkpoint, 'cpu')
        volume = hecate.open_volume(phase / 'work/hecate-padded.zarr')
        result = np.zeros(identity['shape'], dtype=np.uint8); result[:len(prefix)] = prefix
        origins = geometry(identity['shape']); started = time.monotonic()

        def callback(completed, numerator, denominator):
            cut = cut_after(origins, completed, identity['shape'][0])
            publish(root, identity, numerator, denominator, cut, completed, base=prefix)
            atomic_json(root / 'RESUME_STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(),
                        'completed_origins': completed, 'total_origins': len(origins),
                        'updated_at_utc': datetime.now(timezone.utc).isoformat()})

        resumable_predict(hecate)(model, volume, result, reverse=True, stride=None, batch_size=4,
                                 precision='fp32', resume_start=record['resume_origin'],
                                 prefix_rows=len(prefix), checkpoint_callback=callback)
        temporary = output.with_suffix('.partial')
        hecate.write_png(result, temporary); os.replace(temporary, output)
        atomic_json(output.with_suffix('.json'), {'sampling_um': 9.6, 'reverse_z': True,
                    'precision': 'fp32', 'stride': 32, 'shape_yx': list(result.shape)})
        atomic_json(Path(str(output) + '.receipt.json'), {
            'status': 'PASS', 'model': 'hecate', 'phase': 'padded', 'checkpoint': model_record['checkpoint'],
            'output': {'path': str(output), 'bytes': output.stat().st_size, 'sha256': sha(output)},
            'decoded_sha256': hashlib.sha256(result.tobytes()).hexdigest(), 'shape': list(result.shape),
            'min': int(result.min()), 'max': int(result.max()), 'tile_forwards': None,
            'elapsed_seconds': time.monotonic() - started, 'device': 'cpu', 'precision': 'fp32', 'compile': False,
            'resumed_from': record, 'elapsed_scope': 'Resume segment only; excludes saved-prefix computation.'})
        atomic_json(root / 'RESUME_STATUS.json', {'status': 'MAP_COMPLETE', 'pid': os.getpid(),
                    'updated_at_utc': datetime.now(timezone.utc).isoformat()})
    # Existing supervisor rechecks all eight receipts and runs the frozen composition.
    subprocess.run([sys.executable, '-B', str(phase / 'execute_resume.py')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['watch', 'resume', 'verify'])
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); lab = args.lab.resolve(); root = args.root.resolve()
    if args.mode == 'watch': watch(lab, root)
    elif args.mode == 'resume': resume(lab, root)
    else:
        identity = json.loads((root / 'BINDING.json').read_text())
        check_binding(lab, identity); record, _ = read_latest(root, identity)
        print(json.dumps({'status': 'PASS', **record}), flush=True)


if __name__ == '__main__':
    main()
