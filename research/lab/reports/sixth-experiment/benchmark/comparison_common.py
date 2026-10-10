"""File and annotation gates for one fixed, descriptive w029 comparison."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ['russo_ex01675k', 'russo_control75k', 'seed42', 'seed43']
INPUT_NAMES = ['surface', 'inklabels', 'supervision_mask', 'validation_mask',
               'acquisition_observed_mask', 'source_observed_nonzero_mask']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha(path))


def verify_record(item):
    require(isinstance(item, dict) and isinstance(item.get('path'), str), 'Missing file record/path')
    path = Path(item['path']).resolve(strict=True)
    require(path.is_file() and path.stat().st_size == item['size_bytes']
            and sha(path) == item['sha256'], f'Input size/SHA mismatch: {path}')
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def load_plan(path, *, execute=False):
    path = Path(path).resolve(strict=True)
    require(path.stat().st_size <= 1024 * 1024, 'Plan exceeds 1 MiB bound')
    plan = json.loads(path.read_text())
    require(plan['schema_version'] == 1 and plan['kind'] == 'fixed-w029-cpu-comparison', 'Unknown study contract')
    if execute:
        require(plan['ready_for_execution'] is True, 'Study is not frozen/ready for execution')
    require(plan['model_order'] == ARMS and [x['label'] for x in plan['models']] == ARMS,
            'Exactly four fixed arms in declared order are required')
    require(plan['inference'] == dict(layer_start=2, layer_stop=19, direction='forward',
        tile_size=128, stride=64, threads=4, interop_threads=1, batch_size=1, precision='float32',
        device='cpu', valid_mask='none_canvas_only'), 'Inference settings changed')
    require(plan['analysis'] == dict(interior_margin_pixels=64, minimum_positive_pixels=256,
        minimum_negative_pixels=256, diagnostic_threshold=0.5, threshold_operator='>=',
        training_overlap_exclusion='none_established', zero_profiles='retain_and_stratify'),
        'Analysis settings changed; a separate protocol is required')
    require(0 < plan['limits']['per_model_timeout_seconds'] <= 120
            and 0 < plan['limits']['maximum_output_bytes'] <= 32 * 1024 * 1024
            and plan['limits']['automatic_retries'] == 0, 'Resource bounds changed')
    code = plan['code']
    for name in ['run_comparison.py', 'comparison_common.py', 'analyze_comparison.py', 'binary_metrics.py']:
        require(verify_record(code[name]) == HERE / name, 'Execution source path differs')
    for item in plan['verified_inputs']:
        verify_record(item)
    verify_record(plan['model_contract_helper'])
    verify_record(plan['data_contract'])
    model_audit = json.loads(verify_record(plan['model_contract_audit']).read_text())
    require(model_audit['status'] == 'passed'
            and model_audit['tensor_schema_identical_across_four_arms'] is True
            and model_audit['matched_pair']['matched_settings'] is True,
            'Four-arm model contract must pass before any prediction')
    require(set(model_audit['models']) == set(ARMS), 'Model audit does not cover the four fixed arms')
    for model in plan['models']:
        audited = model_audit['models'][model['label']]
        require(audited['sha256'] == model['sha256'] and audited['strict_state_match'] is True,
                'Checkpoint differs from passed strict model audit')
    require(plan['data']['segment'] == 'pherc1667-w029'
            and plan['data']['pooled_layer_count'] == 21
            and plan['data']['source_depth_half_open'] == [13, 97]
            and plan['data']['pooling'] == 'float32_mean_groups4_then_np.rint_uint8'
            and plan['data']['label_plane'] == 10, 'Data preprocessing contract differs')
    bbox = plan['data']['bbox_source_xywh']
    require(isinstance(bbox, list) and len(bbox) == 4
            and all(isinstance(x, int) and not isinstance(x, bool) for x in bbox)
            and bbox[0] >= 0 and bbox[1] >= 0
            and bbox[2:] == [plan['data']['shape'][2], plan['data']['shape'][1]],
            'Source-coordinate crop context is missing or inconsistent')
    spacing = plan['data']['pixel_spacing_um']
    require(spacing is None or (isinstance(spacing, (int, float)) and not isinstance(spacing, bool)
            and np.isfinite(spacing) and spacing > 0), 'Invalid source spacing record')
    return plan, path


def import_helper(plan):
    path = verify_record(plan['model_contract_helper'])
    spec = importlib.util.spec_from_file_location('sixth_verified_model_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_counts(labels, mask):
    positive = int(np.count_nonzero(labels & mask))
    return dict(pixel_count=int(mask.sum()), supplied_positive_pixels=positive,
                supplied_negative_pixels=int(mask.sum()) - positive)


def sufficient(counts):
    return counts['supplied_positive_pixels'] >= 256 and counts['supplied_negative_pixels'] >= 256


def load_data(plan, *, enforce_minimum=True):
    inputs = plan['inputs']
    require(set(inputs) == set(INPUT_NAMES), 'Unexpected or missing input array roles')
    arrays = {name: np.load(verify_record(inputs[name]), mmap_mode='r', allow_pickle=False)
              for name in INPUT_NAMES}
    surface = arrays['surface']
    require(surface.dtype == np.uint8 and surface.ndim == 3 and surface.shape[0] == 21
            and all(128 <= dim <= 256 for dim in surface.shape[1:]), 'Expected bounded uint8 [21,H,W] surface')
    shape = surface.shape[1:]
    require(list(surface.shape) == plan['data']['shape'], 'Input shape differs from frozen source contract')
    encoded = {}
    for key in ['inklabels', 'supervision_mask', 'validation_mask']:
        array = arrays[key]
        foreground = plan['data']['binary_foreground'][key]
        require(foreground in [1, 255] and array.dtype == np.uint8 and array.shape == shape,
                f'Invalid binary annotation contract: {key}')
        require(set(np.unique(array).tolist()) <= {0, foreground}, f'Unexpected grayscale annotation values: {key}')
        encoded[key] = array == foreground
    for key in ['acquisition_observed_mask', 'source_observed_nonzero_mask']:
        require(arrays[key].dtype == np.bool_ and arrays[key].shape == shape,
                f'Expected distinct Boolean source-support mask: {key}')
    require(np.array_equal(arrays['source_observed_nonzero_mask'], np.any(surface[2:19] != 0, axis=0)),
            'Nonzero occupancy mask differs from fixed selected-depth definition')
    labels, validation, supervision = encoded['inklabels'], encoded['validation_mask'], encoded['supervision_mask']
    observed = np.asarray(arrays['acquisition_observed_mask'])
    require(observed.all(), 'STOP: bounded crop requires every source pixel to have actually acquired support; no fill input')
    secondary = validation & observed
    margin = np.zeros(shape, dtype=bool)
    margin[64:shape[0]-64, 64:shape[1]-64] = True
    primary = secondary & margin
    counts = class_counts(labels, primary)
    if enforce_minimum:
        require(sufficient(counts), 'STOP: fixed 64-pixel interior lacks >=256 supplied pixels of each class')
    summary = dict(primary_before_predictions=counts, secondary_before_predictions=class_counts(labels, secondary),
        raw_supervision_validation_overlap_pixels=int((supervision & validation).sum()),
        conditional_current_loader_effective_supervision_pixels=int((supervision & ~validation).sum()),
        acquisition_observed_pixels=int(observed.sum()),
        selected_depth_nonzero_occupancy_pixels=int(arrays['source_observed_nonzero_mask'].sum()),
        primary_observed_zero_profile_pixels=int((primary & ~arrays['source_observed_nonzero_mask']).sum()),
        domain='Declared validation AND actual acquisition-observed support; primary adds fixed 64-pixel margin.',
        training_exposure='Raw supervision overlap is not excluded. Current pinned loader subtracts validation; historical training/teacher/physical independence remains unknown.',
        geometry='No authenticated mesh-validity mask. Acquisition support and nonzero occupancy are not geometric validity.')
    return arrays, labels, primary, secondary, summary


def output_bytes(directory):
    return sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())
