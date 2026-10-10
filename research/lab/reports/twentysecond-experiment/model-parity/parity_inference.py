"""Instrumented synthetic-only comparison of historical and current inference."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from parity_sources import require


def tensor_bytes(value):
    return value.detach().cpu().contiguous().numpy().tobytes()


def array_record(value):
    value = np.asarray(value)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}


def synthetic_volume(depth, height, width, zero=False):
    """Root-prespecified uint8 synthetic pattern; no fitting to model responses."""
    if zero:
        return np.zeros((depth, height, width), dtype=np.uint8)
    z, y, x = np.indices((depth, height, width), dtype=np.uint32)
    return (1 + ((17*z + 3*y + 5*x + (y*x)%29 + (z*y)%23) % 251)).astype(np.uint8)


def selected_indices(case):
    a, b = case['interval']
    ids = np.arange(a, b, dtype=np.int64)
    if ids.size > 17:
        start = ids.size // 2 - 8
        ids = ids[start:start + 17]
    if case.get('reverse', False):
        ids = ids[::-1].copy()
    return ids


class RecordingModel(torch.nn.Module):
    def __init__(self, model, records, tensor_output=False):
        super().__init__()
        self.model = model
        self.records = records
        self.tensor_output = tensor_output

    def forward(self, images):
        require(not torch.is_grad_enabled(), 'Autograd enabled during inference')
        require(images.dtype == torch.float32 and tuple(images.shape) == (1, 1, 17, 128, 128), 'Unexpected input shape or type')
        require(bool(torch.isfinite(images).all()), 'Nonfinite normalized tensor')
        outputs = self.model(images)
        require(isinstance(outputs, dict) and set(outputs) == {'ink'}, 'Unexpected output mapping')
        logits = outputs['ink']
        if isinstance(logits, (list, tuple)):
            require(len(logits) == 1, 'Unexpected deep supervision')
            logits = logits[0]
        require(tuple(logits.shape) == (1, 1, 128, 128) and logits.dtype == torch.float32
                and bool(torch.isfinite(logits).all()), 'Unexpected logits')
        # Historical logits_to_probabilities mutates FP32 logits with sigmoid_.
        self.records.append({'normalized': images.detach().cpu().numpy().copy(),
            'logits': logits.detach().cpu().numpy().copy(),
            'probabilities': logits.detach().float().sigmoid().cpu().numpy().copy(),
            'input_strides': list(images.stride()), 'input_contiguous': images.is_contiguous()})
        return logits if self.tensor_output else outputs


def historical_case(model, normalizer, inference, case, volume):
    del normalizer  # Exact historical dataset imports the pinned historical module.
    ids = inference.center_crop_layer_indices(np.arange(*case['interval']), output_depth=17)
    if case.get('reverse', False):
        ids = ids[::-1].copy()
    require(np.array_equal(ids, selected_indices(case)), 'Historical selected indices differ')
    h, w = volume.shape[1:]
    reader = inference.OmeZarrPatchReader(input_path='<synthetic-memory>', resolution='0',
        depth_axis_first=True, height=h, width=w, layer_indices=ids,
        output_depth=17, preprocessing='tifxyz_robust')
    reader._array = volume  # No Zarr loader can be called; it is fail-closed.
    blocks = inference.iter_blocks((h, w), 128, 64, None, (1, 1))
    dataset = inference.OmeZarrBlockDataset(reader=reader, blocks=blocks, patch_size=128,
        preprocessing='tifxyz_robust')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    weights = inference.compute_importance_map_2d(patch_size=(128,128), mode='hann', sigma_scale=.125).numpy()
    prob_sum = np.zeros((h,w), dtype=np.float32)
    weight_sum = np.zeros_like(prob_sum)
    counts = inference.compute_chunk_contribution_counts(blocks, chunk_shape=(128,128))
    accumulator = inference.ChunkAccumulator(shape=(h,w), chunk_shape=(128,128),
        prob_sum_store=prob_sum, weight_sum_store=weight_sum, contribution_counts=counts)
    records = []
    recording = RecordingModel(model, records, tensor_output=True).eval()
    inference.run_block_inference(loader=loader, model=recording, accumulator=accumulator,
        weight_map=weights, mask=None, device=torch.device('cpu'), amp=False, tta_axes=())
    accumulator.flush_remaining()
    raw = np.zeros_like(prob_sum)
    np.divide(prob_sum, weight_sum, out=raw, where=weight_sum > 0)
    # The retained historical writer receives a whole-canvas synthetic output tile.
    # Copy prevents the generator's in-place np.asarray path mutating raw sums.
    serialized = list(inference.iter_probability_tiles(prob_sum.copy(), weight_sum.copy(), (h,w)))
    require(len(serialized) == 1, 'Unexpected serialization tile count')
    return package_case(case, volume, ids, records, weights, prob_sum, weight_sum, raw,
        serialized[0], [[b.y0,b.x0,b.valid_h,b.valid_w] for b in blocks])


def current_case(model, normalizer, baseline, case, volume):
    ids = selected_indices(case)
    require(len(ids) == 17 and volume.shape[1] >= 128 and volume.shape[2] >= 128,
            'Actual current comparison excludes padding; padding covered by pure fixtures')
    records = []
    recording = RecordingModel(model, records).eval()
    direction = 'reverse' if case.get('reverse', False) else 'forward'
    layer_start = int(min(ids))
    raw, metadata, evaluated = baseline.infer_surface(recording, volume, layer_start,
        direction, normalizer, valid_mask=None)
    weights = baseline.hann_blend_map()
    prob_sum = np.zeros_like(raw)
    weight_sum = np.zeros_like(raw)
    index = 0
    blocks = []
    for tile in metadata['tiles']:
        x,y,w,h = tile['bbox_roi_xywh']
        blocks.append([y,x,h,w])
        if tile['status'] != 'scored':
            continue
        prob_sum[y:y+h,x:x+w] += records[index]['probabilities'][0,0] * weights
        weight_sum[y:y+h,x:x+w] += weights
        index += 1
    require(index == len(records), 'Current reconstructed accumulation count differs')
    reconstructed = np.zeros_like(raw)
    np.divide(prob_sum, weight_sum, out=reconstructed, where=weight_sum > 0)
    require(np.array_equal(raw, reconstructed) and np.array_equal(evaluated, weight_sum > 0),
            'Current retained baseline raw score differs from recorded tile reconstruction')
    serialized = (np.clip(raw, 0, 1) * np.float32(255)).astype(np.uint8)
    return package_case(case, volume, ids, records, weights, prob_sum, weight_sum, raw, serialized, blocks)


def package_case(case, volume, ids, records, weights, prob_sum, weight_sum, raw, serialized, blocks):
    require(len(records) == case['forwards'], 'Case forward count differs')
    arrays = dict(selected_indices=ids, hann=weights, probability_sum=prob_sum,
        weight_sum=weight_sum, raw_score=raw, uint8=serialized,
        author_clipped_score=np.clip((serialized.astype(np.float32)-64)/128, 0, 1),
        evaluated=(weight_sum > 0), blocks=np.array(blocks, dtype=np.int64))
    for key, shape in [('normalized',(0,1,1,17,128,128)), ('logits',(0,1,1,128,128)),
                       ('probabilities',(0,1,1,128,128))]:
        arrays[key] = np.stack([r[key] for r in records]) if records else np.empty(shape, dtype=np.float32)
    require(all(np.isfinite(a).all() for a in arrays.values()), 'Nonfinite retained array')
    metadata = dict(case=case, input=array_record(volume), forwards=len(records),
        arrays={key:array_record(a) for key,a in arrays.items()},
        evaluated_pixels=int(arrays['evaluated'].sum()), source_uint8_unchanged=True,
        input_layout=[{key:r[key] for key in ('input_strides','input_contiguous')} for r in records])
    return arrays, metadata


def save_case(directory, name, arrays, metadata):
    directory = Path(directory) / name
    directory.mkdir()
    for key, array in arrays.items():
        np.save(directory / (key + '.npy'), array, allow_pickle=False)
    return metadata
