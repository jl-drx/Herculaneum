"""Strict CPU checkpoint/runtime qualification using synthetic input only."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time


EXPECTED_SHA256 = '654ec5acec2b6c4788d9cb326e2f0b8c2c730584949a934ef775db7f6ab5d3a6'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert digest(args.checkpoint) == EXPECTED_SHA256
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(args.runtime.resolve()))
    from vesuvius.ink_detection.inference import infer
    assert Path(infer.__file__).resolve().is_relative_to(args.runtime.resolve())
    start = time.monotonic()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    # The released 40,000-update artifact serializes the zero-based index.
    assert checkpoint['step'] == 39999
    state = checkpoint['model']
    assert all(torch.isfinite(t).all() for t in state.values())
    # Enforce strict state loading in this isolated process, never in the worker.
    infer.load_checkpoint = lambda path: checkpoint
    strict_loads = []
    def strict_load(model, weights):
        result = model.load_state_dict(weights, strict=True)
        strict_loads.append({'missing': result.missing_keys, 'unexpected': result.unexpected_keys})
        return result
    infer.load_flat_inference_state = strict_load
    config = infer.configure_model(infer.parse_args(['unused.zarr', str(args.checkpoint), 'unused.tif', '--no-compile']))
    assert config.input_depth == 17 and config.patch_size == 128
    assert config.preprocessing == 'tifxyz_robust'
    # An asymmetric bounded ramp exercises layout and deterministic output.
    z, y, x = np.indices((17, 128, 128), dtype=np.int32)
    raw = ((11 * z + 3 * y + 7 * x + (x * y) % 19) % 256).astype(np.uint8)
    normalized = infer.normalize_flat_patch(raw, config.preprocessing)
    tensor = torch.from_numpy(normalized)[None, None]
    with torch.inference_mode():
        a = config.model(tensor)
        b = config.model(tensor)
    assert tuple(a.shape) == (1, 1, 128, 128)
    assert torch.isfinite(a).all() and torch.equal(a, b)
    output = a.numpy()
    modules = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and name.startswith('vesuvius') and Path(path).is_file():
            path = Path(path).resolve()
            assert path.is_relative_to(args.runtime.resolve())
            modules[str(path.relative_to(args.runtime.resolve()))] = digest(path)
    receipt = {
        'status': 'PASS', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'checkpoint_sha256': EXPECTED_SHA256, 'checkpoint_bytes': args.checkpoint.stat().st_size,
        'serialized_step': checkpoint['step'], 'publisher_update_count': 40000,
        'torch': torch.__version__,
        'parameters': sum(p.numel() for p in config.model.parameters()),
        'state_tensors': len(state), 'all_state_values_finite': True,
        'strict_loads': strict_loads, 'input_depth': config.input_depth,
        'patch_size': config.patch_size, 'preprocessing': config.preprocessing,
        'synthetic_forward_calls': 2, 'repeat_bitwise_equal': True,
        'synthetic_output_shape': list(output.shape),
        'synthetic_output_sha256': hashlib.sha256(output.tobytes()).hexdigest(),
        'runtime_module_sha256': modules, 'seconds': time.monotonic() - start,
        'real_ct_read': False, 'held_out_outputs_read': False,
        'claim': 'Strict load, finite state, synthetic CPU forward and repeatability only; no inference parity with author, accuracy or GPU qualification.'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'runtime_module_sha256'}, indent=2))


if __name__ == '__main__':
    main()
