"""Portable CPU entry point for a fixed PHerc343 smoke check or full map.

Uses restored inputs and upstream code, never edits archived evidence. This is
the published fixed d=+1 reading-side configuration, not a training experiment.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'map'])
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['seed42', 'seed43', 'dnative', 'hecate'], default='seed42')
    parser.add_argument('--phase', choices=['plain', 'padded'], default='plain')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 9:
        parser.error('--threads must be between 1 and 9')
    lab = args.lab.resolve(); output = args.output.resolve()
    if output.is_relative_to(lab):
        parser.error('Use an output directory outside the restored evidence tree')
    if args.mode == 'smoke' and args.phase != 'plain':
        parser.error('The frozen qualification tile uses the plain input phase')
    phase = lab / 'reports/fortysixth-experiment'; work = phase / 'work'
    qualification = json.loads((phase / 'INFERENCE_QUALIFICATION.json').read_text())
    assert qualification['status'] == 'PASS'
    label = {'seed42': 'hybrid_3d2d-seed42', 'seed43': 'hybrid_3d2d-seed43'}.get(args.model, args.model)
    record = next(r for r in qualification['models'] if r['model'] == label)
    checkpoint = lab / Path(record['checkpoint']['path']).relative_to('/workspace/herculaneum-lab')
    assert checkpoint.stat().st_size == record['checkpoint']['bytes'] and digest(checkpoint) == record['checkpoint']['sha256']
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(phase / 'runtime-source'))
    import numpy as np
    import torch
    import zarr
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    if args.model == 'hecate':
        code = lab / 'reports/fortyfifth-experiment/source-evidence/hecate-hecate.py'
        spec = importlib.util.spec_from_file_location('hecate', code)
        hecate = importlib.util.module_from_spec(spec); spec.loader.exec_module(hecate)
        model = hecate.load_model(checkpoint, 'cpu')
        volume = hecate.open_volume(work / f'hecate-{args.phase}.zarr')
        if args.mode == 'smoke':
            raw = np.asarray(volume[:, 928:992, 2752:2816])[::-1].copy()
            assert hashlib.sha256(raw.tobytes()).hexdigest() == record['raw_sha256']
            x = torch.from_numpy((raw.astype(np.float32) / model.divisor)[None, None])
            support = torch.from_numpy(raw.any(0))[None, None, None].expand(1, 1, 16, 64, 64)
            with torch.inference_mode():
                result = model(x, support).sigmoid().numpy()
        else:
            result = np.zeros(volume.shape[-2:], np.uint8)
            hecate.predict(model, volume, result, reverse=True, stride=None, batch_size=4, precision='fp32')
            hecate.write_png(result, output / 'map.png')
    else:
        from vesuvius.ink_detection.inference import infer
        infer.load_checkpoint = lambda p: torch.load(p, map_location='cpu', weights_only=True)
        infer.load_flat_inference_state = lambda m, s: m.load_state_dict(s, strict=True)
        name = 'concat_w047-w048_R5B2_z9500-11000_66' + ('_padded' if args.phase == 'padded' else '')
        source = work / (name + '.zarr')
        view = output / 'level0.zarr'; view.mkdir(); (view / '.zgroup').write_text('{"zarr_format":2}\n')
        (view / '0').symlink_to(source / '0', target_is_directory=True)
        settings = infer.parse_args([str(view), str(checkpoint), str(output / 'map.tif'),
                                    '--overlap', '0.5', '--blend-mode', 'hann', '--no-compile',
                                    '--layer-start', '26', '--layer-end', '43', '--direction', 'reverse',
                                    '--batch-size', '4', '--num-workers', '0'])
        with contextlib.redirect_stdout(io.StringIO()):
            configured = infer.configure_model(settings)
        if args.mode == 'smoke':
            h, w = zarr.open_group(str(source), mode='r')['0'].shape[-2:]
            layers = infer.select_layer_indices(66, layer_start=26, layer_end=43, output_depth=17, direction='reverse')
            reader = infer.FlatPatchReader(input_path=view, resolution='0', depth_axis_first=True,
                                          height=h, width=w, layer_indices=layers, output_depth=17,
                                          preprocessing=configured.preprocessing)
            raw = reader.read(1024, 3072, 128, 128).transpose(2, 0, 1)
            assert hashlib.sha256(raw.tobytes()).hexdigest() == record['raw_sha256']
            x = torch.from_numpy(infer.normalize_flat_patch(raw, configured.preprocessing)[None, None])
            with torch.inference_mode():
                result = configured.model(x).sigmoid().numpy()
        else:
            infer.infer_single_zarr(args=settings, input_zarr=view, configured_model=configured,
                                   device=torch.device('cpu'), output_tiff=output / 'map.tif', layer_direction='reverse')
            import tifffile
            result = tifffile.imread(output / 'map.tif')
    assert np.isfinite(result).all()
    report = {'status': 'PASS', 'mode': args.mode, 'model': args.model, 'phase': args.phase,
              'threads': args.threads, 'checkpoint_sha256': digest(checkpoint),
              'shape': list(result.shape), 'decoded_sha256': hashlib.sha256(result.tobytes()).hexdigest(),
              'elapsed_seconds': time.monotonic() - started, 'torch': torch.__version__,
              'device': 'cpu', 'precision': 'fp32', 'depth': '+1', 'direction': 'reverse'}
    if args.mode == 'smoke':
        expected = np.load(work / f'qualify-{label}-single.npy', allow_pickle=False)
        assert result.shape == expected.shape
        report['max_probability_error_vs_cloud_tile'] = float(np.max(np.abs(result - expected)))
        report['status'] = 'PASS' if report['max_probability_error_vs_cloud_tile'] < 1e-4 else 'FAIL'
        report['scope'] = 'One fixed real tile; validates relocated runtime/input compatibility, not full-map reproduction or ink accuracy.'
        np.save(output / 'probability.npy', result)
    else:
        report['scope'] = 'One full fixed-depth map; completion alone does not establish the four-model reproduction or a reading.'
    (output / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if report['status'] != 'PASS':
        raise SystemExit('Numerical qualification failed; inspect before full inference')


if __name__ == '__main__':
    main()
