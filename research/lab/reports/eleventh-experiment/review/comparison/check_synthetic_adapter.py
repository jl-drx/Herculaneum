"""Synthetic callable/adapter capture parity; no actual model or checkpoint."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LAB = Path('/workspace/herculaneum-lab')
COMPARISON = HERE.parents[1] / 'comparison'
sys.path.insert(0, str(COMPARISON))
from run_comparison import Capture, require


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(path):
    path = Path(path)
    return {'path': str(path), 'size_bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class ArithmeticCallable:
    def __init__(self):
        self.calls = 0
        self.last = None

    def __call__(self, tensor):
        self.calls += 1
        self.last = {'ink': tensor[:, :, 0]+.5*tensor[:, :, -1]+.125*tensor[:, :, 8]}
        return self.last


def main():
    torch.set_num_threads(4)
    reference = load('synthetic_reference', LAB / 'run_ink_reference.py')
    normalization_path = LAB / 'vendor/villa-ink-baseline/vesuvius/image_proc/intensity/normalization.py'
    normalizer = load('synthetic_normalization', normalization_path)
    z, y, x = np.ogrid[:17, :384, :384]
    surface = (z*z + .005*y*y + .025*x + .001*y*z).astype(np.float32)
    initial_sha = hashlib.sha256(surface.tobytes()).hexdigest()
    results = []
    total_calls = 0
    for direction in ['forward', 'reverse']:
        arithmetic = ArithmeticCallable()
        capture = Capture(arithmetic)
        scores, metadata, evaluated = reference.infer_surface(capture, surface, 0,
            direction, normalizer.normalize_robust, None)
        assert arithmetic.calls == len(capture.logits) == 25 and evaluated.all()
        assert metadata['scored_tile_count'] == metadata['tile_count'] == 25
        assert [entry['bbox_roi_xywh'] for entry in metadata['tiles']] == [
            [xx, yy, 128, 128] for yy in range(0, 257, 64) for xx in range(0, 257, 64)]
        weights = reference.hann_blend_map()
        summed = np.zeros((384, 384), dtype=np.float32)
        weight_sum = np.zeros_like(summed)
        layers = surface if direction == 'forward' else surface[::-1]
        for index, tile in enumerate(metadata['tiles']):
            xx, yy, width, height = tile['bbox_roi_xywh']
            raw = layers[:, yy:yy+height, xx:xx+width]
            normalized = np.ascontiguousarray(normalizer.normalize_robust(raw), dtype=np.float32)
            assert not np.shares_memory(raw, normalized)
            assert hashlib.sha256(normalized.tobytes()).hexdigest() == capture.input_hashes[index]
            tensor = torch.from_numpy(normalized)
            expected = tensor[0]+.5*tensor[-1]+.125*tensor[8]
            assert np.array_equal(expected.numpy(), capture.logits[index])
            summed[yy:yy+height, xx:xx+width] += torch.sigmoid(expected).numpy()*weights
            weight_sum[yy:yy+height, xx:xx+width] += weights
        assert np.array_equal(summed/weight_sum, scores)
        assert hashlib.sha256(surface.tobytes()).hexdigest() == initial_sha
        results.append({'direction': direction, 'tiles': 25, 'exact_logit_and_blend_parity': True,
                        'input_source_unchanged': True, 'normalization_does_not_alias_source': True})
        total_calls += arithmetic.calls
    # Capture must return the exact unchanged dictionary, not a rebuilt output.
    arithmetic = ArithmeticCallable()
    capture = Capture(arithmetic)
    output = capture(torch.ones((1, 1, 17, 128, 128), dtype=torch.float32))
    assert output is arithmetic.last
    total_calls += arithmetic.calls
    arithmetic = ArithmeticCallable()
    zero_capture = Capture(arithmetic)
    _, meta, evaluated = reference.infer_surface(zero_capture, np.zeros_like(surface), 0,
        'forward', normalizer.normalize_robust, None)
    assert arithmetic.calls == 0 and meta['skipped_tile_count'] == 25 and not evaluated.any()
    primary = np.zeros((384, 384), dtype=bool)
    primary[32:-32, 32:-32] = True
    try:
        require(evaluated[primary].all(), 'Fixed primary loses coverage')
    except ValueError:
        rejected_missing_coverage = True
    else:
        rejected_missing_coverage = False
    assert rejected_missing_coverage
    receipt = {'status': 'PASS', 'directions': results, 'exact_output_dictionary_identity': True,
        'zero_patch_skips_rejected_for_fixed_primary': True, 'synthetic_callable_calls': total_calls,
        'actual_model_constructions': 0, 'actual_model_forward_calls': 0, 'checkpoint_loads': 0,
        'network_calls': 0, 'retained_candidate_arrays_read': 0,
        'sources': [record(COMPARISON/'code-freeze.json'), record(COMPARISON/'run_comparison.py'),
                    record(LAB/'run_ink_reference.py'), record(normalization_path), record(Path(__file__))]}
    target = HERE / 'synthetic-adapter-verification.json'
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'receipt': record(target)}))


if __name__ == '__main__':
    main()
