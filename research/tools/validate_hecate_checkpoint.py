"""Check interruption/resume equivalence and reject incomplete/corrupt saves."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

import checkpoint_hecate as checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Choose a fresh validation report')
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    original = checkpoint.load_hecate(args.lab)
    resumed = checkpoint.resumable_predict(original)
    results = []

    class Toy(torch.nn.Module):
        patch_size = (16, 64, 64)
        divisor = 255.
        margin = 0
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.batches = []
        def forward(self, image, valid):
            self.batches.append(hashlib.sha256(image.numpy().tobytes()).hexdigest())
            return image.mean(2) * 3.25 - 1.12 + valid.float().mean(2) * .07

    class Interrupted(Exception): pass

    def exercise(label, model, volume, stops):
        shape = volume.shape[-2:]; origins = checkpoint.geometry(shape)
        expected = np.zeros(shape, np.uint8)
        with contextlib.redirect_stdout(io.StringIO()):
            original.predict(model, volume, expected, reverse=True, batch_size=4)
        for stop in stops:
            with tempfile.TemporaryDirectory(prefix='validate-checkpoint-') as temp:
                root = Path(temp)
                identity = {'schema': 1, 'shape': list(shape), 'test': label}
                incomplete = np.zeros(shape, np.uint8)
                def save_and_interrupt(completed, numerator, denominator):
                    if completed == stop:
                        cut = checkpoint.cut_after(origins, completed, shape[0])
                        checkpoint.publish(root, identity, numerator, denominator, cut, completed)
                        raise Interrupted()
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        resumed(model, volume, incomplete, reverse=True, batch_size=4,
                                checkpoint_callback=save_and_interrupt)
                except Interrupted: pass
                else: raise AssertionError('Expected interruption did not occur')
                record, prefix = checkpoint.read_latest(root, identity)
                assert np.array_equal(prefix, expected[:len(prefix)]), 'Captured rows differ from full reference'
                actual = np.zeros(shape, np.uint8); actual[:len(prefix)] = prefix
                with contextlib.redirect_stdout(io.StringIO()):
                    resumed(model, volume, actual, reverse=True, batch_size=4,
                            resume_start=record['resume_origin'], prefix_rows=len(prefix))
                assert np.array_equal(actual, expected), 'Resumed map differs from original full map'
                # No manifest / altered binding / corrupt array must never be accepted.
                manifest = root / f"rows-{record['rows']:05d}" / 'MANIFEST.json'
                content = manifest.read_bytes(); manifest.unlink()
                try: checkpoint.read_latest(root, identity)
                except FileNotFoundError: pass
                else: raise AssertionError('Missing manifest accepted')
                manifest.write_bytes(content)
                try: checkpoint.read_latest(root, {**identity, 'test': 'different'})
                except ValueError: pass
                else: raise AssertionError('Changed binding accepted')
                array = manifest.parent / 'prefix.npy'
                data = bytearray(array.read_bytes()); data[-1] ^= 1; array.write_bytes(data)
                try: checkpoint.read_latest(root, identity)
                except ValueError: pass
                else: raise AssertionError('Corrupt checkpoint accepted')
                results.append({'case': label, 'interrupted_after_origins': stop,
                                'saved_rows': record['rows'], 'resume_origin': record['resume_origin'],
                                'decoded_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                                'full_vs_resumed_bitwise_equal': True,
                                'missing_manifest_changed_binding_corrupt_array_rejected': True})

    rng = np.random.default_rng(20261010)
    for shape in [(16, 194, 211), (16, 192, 160)]:
        volume = rng.integers(1, 256, shape, dtype=np.uint8)
        volume[:, :64, :64] = 0  # Blank selection and irregular edge tiles.
        total = len(checkpoint.geometry(shape[-2:]))
        cuts = [k for k in range(4, total, 4)
                if checkpoint.cut_after(checkpoint.geometry(shape[-2:]), k, shape[-2]) > 0]
        exercise('toy-' + str(shape), Toy(), volume, [cuts[0], cuts[len(cuts)//2], total])
    if args.real:
        phase = args.lab / 'reports/fortysixth-experiment'
        model = original.load_model(phase / 'work/hecate_9.6um.pth', 'cpu')
        volume = original.open_volume(phase / 'work/hecate-padded.zarr')
        # Fixed input-coordinate runtime fixture, not a candidate-selected region.
        crop = np.asarray(volume[:, 928:1024, 2752:2880])
        exercise('real-hecate-fixed-96x128', model, crop, [4])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'PASS', 'source_sha256': checkpoint.SOURCE_SHA,
              'checkpoint_tool_sha256': checkpoint.sha(Path(checkpoint.__file__)),
              'validation_code_sha256': checkpoint.sha(Path(__file__)),
              'cases': results, 'real_fixture_run': args.real,
              'scope': 'Numerical recovery equivalence on bounded fixtures; not a full-map replay, platform shutdown test, or scientific accuracy test.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
