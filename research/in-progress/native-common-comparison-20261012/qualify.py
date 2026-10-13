"""Synthetic pipeline parity and cache failures; zero trained-model forwards."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import numpy as np
import torch
import cv2
import runner as r


class Fake(torch.nn.Module):
    patch_size = (16, 64, 64)
    divisor = 255
    margin = 0

    def __init__(self):
        super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = 0

    def forward(self, image, valid=None):
        self.calls += 1
        # Nonconstant local-position dependence makes a grid-origin error visible.
        h, w = image.shape[-2:]
        x = torch.linspace(-.8, .8, w)[None, None, None, :]
        y = torch.linspace(-.6, .6, h)[None, None, :, None]
        return image.mean(2)*1.3 + x + y - .7


def source_reader(infer, a):
    reader = infer.FlatPatchReader(input_path='unused', resolution='0', depth_axis_first=True,
        height=a.shape[1], width=a.shape[2], layer_indices=np.arange(17), output_depth=17,
        preprocessing='tifxyz_robust')
    reader._array = a
    return reader


def rows_for(root, a, starts, family):
    rows = []
    p = 128 if family == 'hybrid' else 64
    for y, x in starts:
        v = a[:, y:y+p, x:x+p]; path = root/f'{family}-{y}-{x}.npy'
        r.array_save(path, v)
        rows.append(dict(**r.pin(path, root), family=family, origin_yx=[y, x],
                         decoded_sha256=r.array_pin(v)['sha256'], raw_nonempty=bool(v.any())))
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); infer, hec = r.runtime(a.repo)
    import sys
    sys.path.insert(0, str(a.repo/'research/in-progress/native-common-support-20261012'))
    import sampling
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    checks = []; rng = np.random.default_rng(3430800)
    with tempfile.TemporaryDirectory(prefix='common-qualification-') as t:
        root = Path(t)
        for index, (h, w, spacing) in enumerate([(384, 448, 9.366), (402, 470, 8.64)]):
            sub = root/str(index); sub.mkdir()
            cores = [[128, 128, 128, 128], [w-144, h-144, 128, 128]]
            native = rng.integers(0, 256, (17, h, w), dtype=np.uint8)
            native[:, :128, :128] = 0
            full = [(y, x) for y in sampling.starts(h, 128, 64) for x in sampling.starts(w, 128, 64)]
            subset = sorted({(y, x) for cx, cy, cw, ch in cores
                             for x, y, _, _ in sampling.tiles((h, w), [cx, cy, cx+cw, cy+ch], 128, 64)})
            rows = rows_for(sub, native, subset, 'hybrid'); packet = r.Packet(sub, rows)
            blocks = [infer.Block(y, x, 128, 128) for y, x in full]
            selected = [infer.Block(y, x, 128, 128) for y, x in subset]
            whole = r.hybrid_calculate(infer, source_reader(infer, native), blocks, Fake(), (h, w))
            sparse = r.hybrid_calculate(infer, r.PacketReader(packet), selected, Fake(), (h, w))
            for cx, cy, cw, ch in cores:
                sl = np.s_[cy:cy+ch, cx:cx+cw]
                assert all(np.array_equal(l[sl], rr[sl]) for l, rr in zip(whole, sparse))
            checks.append(f'hybrid full/sparse numerator and denominator exact, spacing={spacing}')
            hh, ww = round(h*spacing/9.6), round(w*spacing/9.6)
            hv = rng.integers(0, 256, (16, hh, ww), dtype=np.uint8); hv[:, :96, :96] = 0
            selected = set()
            for cx, cy, cw, ch in cores:
                x0, x1 = sampling.linear_interval(cx, cx+cw, ww, w)
                y0, y1 = sampling.linear_interval(cy, cy+ch, hh, h)
                selected.update((y, x) for x, y, _, _ in sampling.tiles((hh, ww), [x0, y0, x1, y1], 64, 32))
            hp = r.Packet(sub, rows_for(sub, hv, sorted(selected), 'hecate'))
            whole = np.zeros((hh, ww), np.uint8); sparse = np.zeros_like(whole)
            with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
                hec.predict(Fake(), hv, whole, stride=32)
                hec.predict(Fake(), r.SelectedVolume(hp, hv.shape), sparse, stride=32)
            whole = cv2.resize(whole.astype(np.float32)/255, (w, h), interpolation=cv2.INTER_LINEAR)
            sparse = cv2.resize(sparse.astype(np.float32)/255, (w, h), interpolation=cv2.INTER_LINEAR)
            for cx, cy, cw, ch in cores:
                assert np.array_equal(whole[cy:cy+ch, cx:cx+cw], sparse[cy:cy+ch, cx:cx+cw])
            checks.append(f'Hecate full/sparse native output exact, spacing={spacing}, terminal edge tiles included')
        # Every actual hybrid patch follows the same memory layout and normalization.
        count = 0
        for case in r.read(a.inputs/'PATCHES_COMPLETE.json')['cases']:
            packet = r.Packet(a.inputs, [x for x in case['patches'] if x['family'] == 'hybrid'])
            for row in packet.rows:
                y, x = row['origin_yx']; raw = packet.get(y, x)
                original = source_reader(infer, raw)
                ds1 = infer.FlatBlockDataset(reader=original, blocks=[infer.Block(0, 0, 128, 128)],
                    patch_size=128, preprocessing='tifxyz_robust')
                ds2 = infer.FlatBlockDataset(reader=r.PacketReader(packet), blocks=[infer.Block(y, x, 128, 128)],
                    patch_size=128, preprocessing='tifxyz_robust')
                assert torch.equal(ds1[0][0], ds2[0][0]); count += 1
        assert count == 216; checks.append('All 216 actual hybrid normalized input tensors match original reader exactly')
        folder = root/'cache'; identity = dict(go='synthetic', model='fake')
        rows = [dict(origin_yx=[i*32, 0], sha256=str(i), raw_nonempty=True) for i in range(3)]
        image = torch.arange(16*64*64, dtype=torch.float32).reshape(1, 1, 16, 64, 64)/65536
        fake = Fake(); cached = r.CacheModel(fake, folder, identity, rows, stop_after=2)
        try:
            cached(image); cached(image+1)
        except r.InjectedStop:
            assert cached.new == 2
        else:
            raise AssertionError('Expected injection')
        resumed = r.CacheModel(fake, folder, identity, rows)
        out = [resumed(image+i) for i in range(3)]
        assert resumed.new == 1 and resumed.reused == 2 and fake.calls == 3
        replay = r.CacheModel(fake, folder, identity, rows, readonly=True)
        assert all(torch.equal(out[i], replay(image+i)) for i in range(3))
        assert fake.calls == 3; checks.append('Controlled interruption/resume and read-only exact replay, no duplicate forward')
        def rejected(label, operation):
            try:
                operation()
            except (AssertionError, ValueError):
                checks.append(label)
            else:
                raise AssertionError('Failed to reject: '+label)
        rejected('Reject changed binding', lambda: r.CacheModel(fake, folder, {'go':'wrong'}, rows)(image))
        rejected('Reject changed normalized input', lambda: r.CacheModel(fake, folder, identity, rows)(image+2))
        changed = [dict(rows[0], sha256='changed'), *rows[1:]]
        rejected('Reject changed raw input hash', lambda: r.CacheModel(fake, folder, identity, changed)(image))
        file = folder/'tile-0000.npy'; original = file.read_bytes(); file.write_bytes(original[:-1]+bytes([original[-1]^1]))
        rejected('Reject corrupted logits', lambda: r.CacheModel(fake, folder, identity, rows)(image))
        file.write_bytes(original)
        meta = folder/'tile-0000.json'; saved = meta.read_bytes(); meta.unlink()
        rejected('Reject uncommitted logits during read-only replay', lambda: r.CacheModel(fake, folder, identity, rows, readonly=True)(image))
        assert fake.calls == 3
        new = r.CacheModel(fake, folder, identity, rows); new(image)
        assert new.new == 1 and new.reused == 0 and fake.calls == 4
        checks.append('Uncommitted logits recomputed rather than trusted during writable resume')
    result = dict(status='PASS', utc=r.now(), checks=checks, trained_model_forwards=0,
                  source='Frozen original inference implementations', code=[r.pin(Path(__file__)), r.pin(Path(r.__file__))])
    r.atomic(a.output, result); print(__import__('json').dumps(result))


if __name__ == '__main__':
    main()
