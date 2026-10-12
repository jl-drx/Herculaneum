"""Fixed five-model comparison; original accumulation, committed per-tile logits.

No acquisition, training, geometry selection, calibration fitting or paid compute.
"""
import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pin(path, root=None):
    path = Path(path)
    return dict(path=str(path.relative_to(root) if root else path),
                bytes=path.stat().st_size, sha256=sha(path))


def check(root, row):
    path = Path(root) / row['path']
    assert path.resolve().is_relative_to(Path(root).resolve())
    assert path.is_file() and not path.is_symlink()
    assert pin(path, root) == row, str(path)
    return path


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def array_save(path, value):
    path = Path(path); tmp = path.with_name(path.name + '.partial')
    with tmp.open('wb') as f:
        if isinstance(value, dict):
            np.savez_compressed(f, **value)
        else:
            np.save(f, value, allow_pickle=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def array_pin(value):
    value = np.asarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def runtime(repo):
    sys.path.insert(0, str(repo / 'research/lab/reports/fortysixth-experiment/runtime-source'))
    from vesuvius.ink_detection.inference import infer
    infer.load_checkpoint = lambda p: torch.load(p, map_location='cpu', weights_only=True)
    infer.load_flat_inference_state = lambda m, s: m.load_state_dict(s, strict=True)
    path = repo / 'research/lab/reports/fortyfifth-experiment/source-evidence/hecate-hecate.py'
    spec = importlib.util.spec_from_file_location('hecate_original', path)
    hec = importlib.util.module_from_spec(spec); spec.loader.exec_module(hec)
    return infer, hec


class Packet:
    def __init__(self, root, rows):
        self.root = root
        self.rows = sorted(rows, key=lambda r: r['origin_yx'])
        self.lookup = {tuple(r['origin_yx']): r for r in self.rows}
        assert len(self.lookup) == len(self.rows)

    def get(self, y, x):
        row = self.lookup[(y, x)]; path = self.root / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        a = np.load(path, allow_pickle=False)
        assert a.dtype == np.uint8 and array_pin(a)['sha256'] == row['decoded_sha256']
        assert bool(a.any()) == row['raw_nonempty']
        return a


class PacketReader:
    def __init__(self, packet):
        self.packet = packet

    def read(self, y, x, h, w):
        a = self.packet.get(y, x)
        assert a.shape == (17, h, w) and (h, w) == (128, 128)
        # FlatPatchReader creates a C-contiguous float32 HWZ canvas BEFORE MAD.
        return np.ascontiguousarray(np.moveaxis(a, 0, -1), dtype=np.float32)


class SelectedVolume:
    def __init__(self, packet, shape):
        self.packet = packet; self.shape = tuple(shape); self.dtype = np.dtype('uint8')

    def __getitem__(self, slices):
        zs, ys, xs = slices
        assert (zs.start, zs.stop, zs.step) == (0, 16, None)
        y, x = ys.start, xs.start; h, w = ys.stop-y, xs.stop-x
        assert (h, w) == (64, 64) and ys.step is None and xs.step is None
        if (y, x) not in self.packet.lookup:
            # Only noncontributing tiles are omitted. The original full-grid
            # denominator is still accumulated, including those empty tiles.
            return np.zeros((16, h, w), np.uint8)
        a = self.packet.get(y, x); assert a.shape == (16, h, w)
        return a


class InjectedStop(Exception):
    pass


class CacheModel(torch.nn.Module):
    def __init__(self, model, folder, binding, rows, stop_after=None, readonly=False, progress=None):
        super().__init__()
        self.base = model; self.folder = folder; self.binding = binding
        self.rows = [r for r in rows if r['raw_nonempty']]
        self.count = self.new = self.reused = 0
        self.stop_after = stop_after; self.readonly = readonly; self.progress = progress
        folder.mkdir(parents=True, exist_ok=True)
        for name in ['patch_size', 'divisor', 'margin']:
            if hasattr(model, name):
                setattr(self, name, getattr(model, name))

    def forward(self, *args):
        index = self.count; self.count += 1
        assert index < len(self.rows), 'Unexpected model call'
        source = self.rows[index]
        meta = self.folder / f'tile-{index:04d}.json'
        path = meta.with_suffix('.npy')
        inputs = [array_pin(a.detach().cpu().numpy()) for a in args]
        identity = dict(binding=self.binding, index=index,
                        origin_yx=source['origin_yx'], source_sha256=source['sha256'], inputs=inputs)
        if meta.exists():
            record = read(meta)
            assert all(record[k] == v for k, v in identity.items()), 'Cache identity mismatch'
            check(self.folder, record['logits'])
            a = np.load(path, allow_pickle=False)
            assert array_pin(a) == record['decoded_logits']
            out = torch.from_numpy(a); self.reused += 1
        else:
            assert not self.readonly, 'Read-only replay lacks a committed tile'
            with torch.inference_mode():
                out = self.base(*args)
            assert out.ndim == 4 and out.shape[:2] == (1, 1) and out.dtype == torch.float32
            assert torch.isfinite(out).all(), 'Nonfinite logits'
            a = out.detach().cpu().numpy()
            array_save(path, a)
            atomic(meta, dict(**identity, logits=pin(path, self.folder), decoded_logits=array_pin(a)))
            self.new += 1
            if self.progress:
                self.progress(self.count, len(self.rows), self.new, self.reused)
            if self.stop_after is not None and self.new == self.stop_after:
                raise InjectedStop('Controlled exception after committed tile receipt')
        assert out.ndim == 4 and out.shape[:2] == (1, 1) and out.dtype == torch.float32
        assert torch.isfinite(out).all()
        return out


def hybrid_calculate(infer, reader, blocks, model, shape):
    dataset = infer.FlatBlockDataset(reader=reader, blocks=blocks, patch_size=128,
                                     preprocessing='tifxyz_robust')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    weight = infer.compute_importance_map_2d(patch_size=(128, 128), mode='hann').numpy()
    num = np.zeros(shape, np.float32); den = np.zeros_like(num)
    acc = infer.ChunkAccumulator(shape=shape, chunk_shape=(128, 128), prob_sum_store=num,
            weight_sum_store=den,
            contribution_counts=infer.compute_chunk_contribution_counts(blocks, chunk_shape=(128, 128)))
    with torch.inference_mode():
        infer.run_block_inference(loader=loader, model=model, accumulator=acc, weight_map=weight,
                                 mask=None, device=torch.device('cpu'), amp_dtype=None,
                                 tta_axes=(), tta_batch_size=None)
    acc.flush_remaining()
    return num, den


def calculate(infer, hec, root, case, patches, name, model, folder, binding,
              stop_after=None, readonly=False, progress=None):
    family = 'hecate' if name == 'hecate' else 'hybrid'
    packet = Packet(root, [p for p in patches['patches'] if p['family'] == family])
    cached = CacheModel(model, folder, binding, packet.rows, stop_after, readonly, progress)
    h, w = case['native']['shape_zyx'][1:]
    outputs = {}
    if family == 'hybrid':
        blocks = [infer.Block(y, x, 128, 128) for y, x in sorted(packet.lookup)]
        num, den = hybrid_calculate(infer, PacketReader(packet), blocks, cached, (h, w))
        for core in case['cores']:
            x, y, cw, ch = core['core_xywh']; sl = np.s_[y:y+ch, x:x+cw]
            n, d = num[sl].copy(), den[sl].copy()
            assert (d > 1e-6).all(), 'Incomplete blend support'
            p = np.clip(n/d, 0, 1); raw = (p*255).astype(np.uint8)
            outputs[core['id']] = dict(aligned=raw.astype(np.float32)/255,
                display=raw, raw_grid=raw, unquantized=p, numerator=n, denominator=d)
    else:
        volume = SelectedVolume(packet, (16, *patches['hecate_grid_yx']))
        raw = np.zeros(volume.shape[-2:], np.uint8)
        with (folder/'accumulation.log').open('a') as log, contextlib.redirect_stdout(log), torch.inference_mode():
            hec.predict(cached, volume, raw, reverse=False, stride=32, batch_size=1, precision='fp32')
        aligned = cv2.resize(raw.astype(np.float32)/255, (w, h), interpolation=cv2.INTER_LINEAR)
        for core in case['cores']:
            x, y, cw, ch = core['core_xywh']; a = aligned[y:y+ch, x:x+cw].copy()
            outputs[core['id']] = dict(aligned=a, display=np.rint(a*255).astype(np.uint8))
        # Sparse global prediction: meaningful only on the frozen native cores.
        outputs['_model_grid'] = dict(raw_grid=raw)
    for core in case['cores']:
        a = outputs[core['id']]['aligned']
        assert a.shape == (128, 128) and np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()
    assert cached.count == len(cached.rows)
    return outputs, dict(model_calls=cached.count, new_forwards=cached.new, reused_forwards=cached.reused)


def execute(repo, root, run):
    go = read(run/'GO.json'); go_sha = sha(run/'GO.json')
    for row in go['code_pins']:
        check(run, row)
    for row in go['runtime_pins']:
        check(repo, row)
    for row in go['input_pins']:
        check(root, row)
    for row in go['models']:
        p = Path(row['checkpoint']['path']); assert pin(p) == row['checkpoint']
    assert sys.version.split()[0] == go['environment']['python']
    assert torch.__version__ == go['environment']['torch'] and np.__version__ == go['environment']['numpy']
    assert cv2.__version__ == go['environment']['opencv']
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    infer, hec = runtime(repo)
    selection = read(root/'SELECTION.json'); patches = read(root/'PATCHES_COMPLETE.json')
    started = time.monotonic(); receipts = []
    for model_row in go['models']:
        name = model_row['name']; ck = Path(model_row['checkpoint']['path'])
        if name == 'hecate':
            model = hec.load_model(ck, 'cpu')
            assert tuple(model.patch_size) == (16, 64, 64) and model.divisor == 255
        else:
            conf = infer.configure_model(infer.parse_args(['unused.zarr', str(ck), 'unused.tif', '--no-compile']))
            model = conf.model
            assert conf.input_depth == 17 and conf.patch_size == 128 and conf.preprocessing == 'tifxyz_robust'
        assert not model.training
        for case, pc in zip(selection['cases'], patches['cases']):
            assert case['id'] == pc['id']
            job = name + '--' + case['id']; folder = run/'jobs'/job; folder.mkdir(parents=True, exist_ok=True)
            binding = dict(go_sha256=go_sha, model=name, checkpoint_sha256=model_row['checkpoint']['sha256'], case=case['id'])
            complete = folder/'COMPLETE.json'
            if complete.exists():
                record = read(complete); assert record['binding'] == binding
                for r in record['outputs']:
                    check(folder, r)
                for rp in folder.glob('tile-*.json'):
                    r = read(rp); assert r['binding'] == binding; check(folder, r['logits'])
                receipts.append(record); continue
            begin = time.monotonic()
            def progress(count, total, new, reused):
                atomic(run/'STATUS.json', dict(status='RUNNING', utc=now(), job=job,
                    completed_jobs=len(receipts), expected_jobs=20, committed_job_tiles=count,
                    expected_job_tiles=total, new_forwards_this_attempt=new, reused_forwards=reused,
                    elapsed_seconds=time.monotonic()-started, go_sha256=go_sha))
                if count % 10 == 0 or count == total:
                    print(f'{now()} {job} {count}/{total} committed', flush=True)
            progress(0, pc['counts']['hecate' if name == 'hecate' else 'hybrid']['nonempty'], 0, 0)
            if name == 'reader_v2' and case == selection['cases'][0] and not (run/'INJECTED_STOP.json').exists():
                existing = len(list(folder.glob('tile-*.json')))
                if existing == 0:
                    try:
                        calculate(infer, hec, root, case, pc, name, model, folder, binding, stop_after=2, progress=progress)
                    except InjectedStop:
                        atomic(run/'INJECTED_STOP.json', dict(status='EXPECTED_CONTROLLED_EXCEPTION',
                            go_sha256=go_sha, job=job, committed_tiles=2, completed_map=False,
                            scope='Exception after tile receipt; not process crash or lost-success-response test'))
                    else:
                        raise AssertionError('Injection failed to stop')
                else:
                    raise AssertionError('Unrecorded interruption precedes controlled injection; inspect before resume')
            arrays, counts = calculate(infer, hec, root, case, pc, name, model, folder, binding, progress=progress)
            replay, replay_counts = calculate(infer, hec, root, case, pc, name, model, folder, binding, readonly=True)
            assert arrays.keys() == replay.keys()
            assert all(arrays[c].keys() == replay[c].keys() and all(np.array_equal(v, replay[c][k])
                       for k, v in arrays[c].items()) for c in arrays), 'Logit replay changed outputs'
            assert replay_counts['new_forwards'] == 0 and replay_counts['reused_forwards'] == counts['model_calls']
            output_pins = []
            for core_id, values in arrays.items():
                path = folder/(core_id+'.npz'); array_save(path, values); output_pins.append(pin(path, folder))
            record = dict(status='PASS', binding=binding, counts=counts, read_only_replay=replay_counts,
                          seconds=time.monotonic()-begin, outputs=output_pins, utc=now())
            atomic(complete, record); receipts.append(record)
            print(f'{now()} COMPLETE {job} {record["seconds"]:.1f}s', flush=True)
            del arrays, replay; gc.collect()
        del model
        if name != 'hecate':
            del conf
        gc.collect()
    assert len(receipts) == 20 and sum(r['counts']['model_calls'] for r in receipts) == 1698
    result = dict(status='ALL_MAPS_COMPLETE', go_sha256=go_sha, utc=now(), completed_jobs=20,
                  core_maps=120, unique_model_forwards=1698, exact_read_only_replays=20,
                  total_job_seconds=sum(r['seconds'] for r in receipts), gpu_spend_GBP=0,
                  receipts=[pin(p, run) for p in sorted((run/'jobs').glob('*/COMPLETE.json'))])
    atomic(run/'INFERENCE_COMPLETE.json', result)
    atomic(run/'STATUS.json', result); print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True); p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    a.run.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (a.run/'RUN_LOCK').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    atomic(a.run/'WORKER.json', dict(pid=os.getpid(), start_ticks=fields[19], utc=now()))
    try:
        execute(a.repo, a.inputs, a.run)
    except Exception as e:
        atomic(a.run/'FAILED.json', dict(status='STOPPED_ON_ERROR', utc=now(),
               error_type=type(e).__name__, message=str(e), go_sha256=sha(a.run/'GO.json')))
        raise


if __name__ == '__main__':
    main()
