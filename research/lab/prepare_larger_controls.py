#!/usr/bin/env python3
"""Freeze the predefined central control crop without loading detector maps."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np


ROOT = Path('/workspace/herculaneum-lab')
PARENT = ROOT / 'data/experiments/pherc0800-larger1536'
DEST = ROOT / 'data/experiments/pherc0800-larger-controls1024'
PLAN = ROOT / 'reports/third-experiment/fixed-controls-plan.json'
RUNNER = ROOT / 'run_ink_baseline.py'
PYTHON = Path('/workspace/.venvs/herculaneum/bin/python')
EXPECTED_RUNNER_SHA = '2cbdb280777e19794fc78857969f9c1d43965da17772f6a1659264e6f5a48a59'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    return {'path': str(path), 'sha256': sha256(path), 'size_bytes': path.stat().st_size}


def main():
    if DEST.exists() or PLAN.exists():
        raise ValueError('Frozen control inputs/plan already exist; never overwrite them.')
    parent_provenance_path = PARENT / 'provenance.json'
    parent_provenance = json.loads(parent_provenance_path.read_text())
    parent_inputs = {name: record(PARENT / name) for name in ('surface.npy', 'geometry-mask.npy', 'provenance.json')}
    if parent_inputs['surface.npy']['sha256'] != parent_provenance['array_sha256']:
        raise ValueError('Parent surface hash differs from acquisition provenance.')
    if parent_inputs['geometry-mask.npy']['sha256'] != parent_provenance['derived_outputs']['geometry-mask.npy']['sha256']:
        raise ValueError('Parent geometry mask hash differs from acquisition provenance.')
    if sha256(RUNNER) != EXPECTED_RUNNER_SHA:
        raise ValueError('Original baseline source differs from its frozen hash.')
    surface = np.load(PARENT / 'surface.npy', mmap_mode='r', allow_pickle=False)
    mask = np.load(PARENT / 'geometry-mask.npy', mmap_mode='r', allow_pickle=False)
    if surface.shape != (31, 1536, 1536) or surface.dtype != np.uint8:
        raise ValueError('Unexpected parent surface contract.')
    if mask.shape != (1536, 1536) or mask.dtype != np.bool_:
        raise ValueError('Unexpected parent geometry mask contract.')
    if parent_provenance['bbox_source_xywh'] != [512, 512, 1536, 1536]:
        raise ValueError('Unexpected parent native origin.')
    crop = np.ascontiguousarray(surface[:, 256:1280, 256:1280])
    crop_mask = np.ascontiguousarray(mask[256:1280, 256:1280])
    DEST.mkdir(parents=True, exist_ok=False)
    np.save(DEST / 'surface.npy', crop, allow_pickle=False)
    np.save(DEST / 'geometry-mask.npy', crop_mask, allow_pickle=False)
    shutil.copyfile(parent_provenance_path, DEST / 'parent-provenance.json')
    created = datetime.now(timezone.utc).isoformat()
    provenance = {
        'created_at_utc': created,
        'sample_id': parent_provenance['sample_id'],
        'target_volume_id': parent_provenance['target_volume_id'],
        'segment_id': parent_provenance['segment_id'],
        'surface_url': parent_provenance['surface_url'],
        'pyramid_level': parent_provenance['pyramid_level'],
        'bbox_source_xywh': [768, 768, 1024, 1024],
        'bbox_parent_xywh': [256, 256, 1024, 1024],
        'shape_zyx': list(crop.shape),
        'dtype': str(crop.dtype),
        'voxel_spacing_zyx_um': parent_provenance['voxel_spacing_zyx_um'],
        'nominal_canvas_size_mm_xy': [8.84736, 8.84736],
        'array_sha256': sha256(DEST / 'surface.npy'),
        'parent_inputs': parent_inputs,
        'retained_parent_provenance_sha256': sha256(DEST / 'parent-provenance.json'),
        'geometry_mask': record(DEST / 'geometry-mask.npy'),
        'geometry_valid_pixels': int(crop_mask.sum()),
        'geometry_valid_fraction': float(crop_mask.mean()),
        'source_layer_indices': list(range(31)),
        'selection': 'Fixed central crop at relative origin (256,256), specified before control scores; no detector maps loaded during preparation.',
        'derivation': 'Lossless array slice only; no contrast, resampling, mask multiplication or depth selection applied to retained uint8 source.',
        'preparation_code': record(Path(__file__)),
        'status': 'Predefined depth/orientation diagnostic inputs, not independent validation data or an ink label.',
    }
    (DEST / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    checkpoint = ROOT / 'data/models/ink9um-seed42-step075000.pth'
    checkpoint_record = record(checkpoint)
    if checkpoint_record['sha256'] != 'e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab':
        raise ValueError('Seed 42 checkpoint is not the verified official final checkpoint.')
    conditions = [('center-forward', 7, 'forward'), ('minus2-forward', 5, 'forward'), ('plus2-forward', 9, 'forward'), ('center-reverse', 7, 'reverse')]
    runs = []
    for label, start, direction in conditions:
        out = ROOT / f'reports/third-experiment/ink9um-controls-{label}'
        if out.exists():
            raise ValueError(f'Control output already exists: {out}')
        argv = [str(PYTHON), str(RUNNER), str(DEST / 'surface.npy'), '--provenance', str(DEST / 'provenance.json'), '--mask', str(DEST / 'geometry-mask.npy'), '--output-dir', str(out), '--seed42-checkpoint', str(checkpoint), '--layer-start', str(start), '--direction', direction, '--threads', '4']
        runs.append({'condition': label, 'seed': 42, 'layer_start_stop': [start, start + 17], 'source_layer_indices': list(range(start, start + 17))[::(-1 if direction == 'reverse' else 1)], 'direction': direction, 'output_dir': str(out), 'scheduled_tiles': 225, 'argv': argv})
    plan = {
        'frozen_at_utc': created,
        'scope': 'Four predefined sequential seed-42 controls on the fixed central 1024 crop of the geometry-selected larger context.',
        'launch_gate': 'Do not launch until parent confirms the full-surface two-seed process has completed.',
        'selection_does_not_use_detector_maps': True,
        'source_native_bbox_xywh': [768, 768, 1024, 1024],
        'source_parent_bbox_xywh': [256, 256, 1024, 1024],
        'source_inputs': {name: record(DEST / name) for name in ('surface.npy', 'geometry-mask.npy', 'provenance.json', 'parent-provenance.json')},
        'runner': record(RUNNER),
        'checkpoint': checkpoint_record,
        'python_executable': str(PYTHON),
        'runs': runs,
        'total_scheduled_checkpoint_tiles': 900,
        'processing': {'cpu_threads': 4, 'batch_size': 1, 'dtype': 'float32', 'tile_shape_zyx': [17, 128, 128], 'stride_yx': [64, 64], 'normalization': 'Unmodified upstream robust-MAD per full ZYX patch'},
        'planned_context_consistency_check': 'Exact seed-42 full/crop equality on geometry-valid, evaluated, observed common pixels at least 64 pixels from the crop edge; full origin and crop offset are aligned to stride 64.',
        'planned_common_domain': 'Intersection of geometry-valid, evaluated in all compared runs, and data-present support for all selected depth windows, with 64-pixel crop-edge and original-global-geometry margins. Also report stricter support excluding any contributing scored tile with invalid geometry or missing selected-depth support.',
        'limitations': ['Depth/orientation controls are sensitivity diagnostics, not independent observations or accuracy validation.', 'The fixed central crop shares source data with the full-context run.', 'Margins do not guarantee all contributing tiles contain valid observed pixels.', 'Model scores and seed agreement do not authenticate ink, letters or text.'],
    }
    PLAN.write_text(json.dumps(plan, indent=2) + '\n')
    for name, original in parent_inputs.items():
        if sha256(PARENT / name) != original['sha256']:
            raise ValueError('Parent input changed during preparation.')
    for path in DEST.iterdir():
        path.chmod(0o444)
    PLAN.chmod(0o444)
    print(json.dumps({'input_ready': str(DEST), 'frozen_plan': record(PLAN), 'shape_zyx': list(crop.shape), 'geometry_valid_fraction': float(crop_mask.mean()), 'source_sha256': provenance['array_sha256'], 'mask_sha256': provenance['geometry_mask']['sha256'], 'control_count': len(runs), 'scheduled_checkpoint_tiles': 900}, indent=2))


if __name__ == '__main__':
    main()
