"""Independent NumPy calibration-map reconstruction; no array I/O or model calls.

The caller owns file pins and completed-run authorization. Functions accept
already loaded arrays only. The source/synthetic preparation imports no author
inference, normalization, selector or scorer. Score equality uses the frozen
2e-6 absolute tolerance; evaluated masks must match exactly.
"""

import hashlib
import math
import re

import numpy as np


SCORE_TOLERANCE = 2e-6
TILE_SIZE = 128
STRIDE = 64
SHAPES = ((256, 256), (384, 384))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _shape(shape):
    require(isinstance(shape, (tuple, list)) and len(shape) == 2,
            "Shape must be a literal pair")
    require(all(type(v) is int for v in shape), "Shape dimensions must be integers")
    result = tuple(shape)
    require(result in SHAPES, "Only the frozen 256-square and 384-square maps are allowed")
    return result


def expected_grid(shape):
    """Independent fixed row-major XYWH grid, with 128-pixel complete windows."""
    height, width = _shape(shape)
    return [[x, y, TILE_SIZE, TILE_SIZE]
            for y in range(0, height - TILE_SIZE + 1, STRIDE)
            for x in range(0, width - TILE_SIZE + 1, STRIDE)]


def hann_weights():
    """Mathematical nonperiodic Hann in float64, floored then rounded to FP32."""
    angles = np.arange(TILE_SIZE, dtype=np.float64) * (2. * np.pi / (TILE_SIZE - 1))
    vector = .5 * (1. - np.cos(angles))
    weights = vector[:, None] * vector[None, :]
    weights /= max(float(np.max(weights)), float(np.finfo(np.float32).eps))
    return np.maximum(weights, .001).astype(np.float32)


def sigmoid_probabilities(logits):
    """Stable mathematical sigmoid in float64, rounded once to FP32."""
    values = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -values)).astype(np.float32)


def reconstruct(logits, shape):
    """Blend saved raw logits in fixed order with FP32 arithmetic throughout."""
    height, width = _shape(shape)
    grid = expected_grid(shape)
    require(isinstance(logits, np.ndarray) and logits.dtype == np.dtype(np.float32),
            "Raw logits must be a native FP32 array")
    require(logits.shape == (len(grid), TILE_SIZE, TILE_SIZE) and np.isfinite(logits).all(),
            "Raw tile-logit shape, count or finite-value contract differs")
    numerator = np.zeros((height, width), dtype=np.float32)
    denominator = np.zeros_like(numerator)
    weights = hann_weights()
    probabilities = sigmoid_probabilities(logits)
    for (left, top, w, h), probability in zip(grid, probabilities):
        region = np.s_[top:top + h, left:left + w]
        numerator[region] += probability * weights
        denominator[region] += weights
    scores = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=scores, where=denominator > 0)
    return scores, denominator > 0


def check_metadata(shape, metadata):
    """Validate the retained Capture schema without following any file paths."""
    grid = expected_grid(shape)
    require(isinstance(metadata, dict), "Per-case metadata must be a dictionary")
    require(metadata.get("adapter_direction") == "forward" and
            metadata.get("adapter_layer_start") == 2, "Forward layers [2:19] required")
    inference = metadata.get("inference")
    require(isinstance(inference, dict), "Inference metadata is missing")
    tiles = inference.get("tiles")
    captured = metadata.get("captured_tiles")
    require(isinstance(tiles, list) and isinstance(captured, list) and
            len(tiles) == len(captured) == len(grid), "Tile record counts differ")
    require(all(isinstance(t, dict) for t in tiles + captured), "Tile record type differs")
    require(all(isinstance(t.get("bbox_roi_xywh"), list) and len(t["bbox_roi_xywh"]) == 4 and
                all(type(v) is int for v in t["bbox_roi_xywh"]) for t in tiles + captured),
            "Tile bboxes must be literal integer XYWH lists")
    require([t.get("bbox_roi_xywh") for t in tiles] == grid and
            [t.get("bbox_roi_xywh") for t in captured] == grid,
            "Tile placement or row-major order differs")
    require(all(t.get("status") == "scored" for t in tiles), "Every frozen tile must be scored")
    for field in ("tile_count", "scored_tile_count"):
        require(type(inference.get(field)) is int and inference[field] == len(grid),
                "Inference tile accounting differs: " + field)
    require(type(inference.get("skipped_tile_count")) is int and
            inference["skipped_tile_count"] == 0, "Skipped tiles are outside frozen scope")
    pixels = math.prod(_shape(shape))
    for field in ("valid_pixel_count", "evaluated_valid_pixel_count"):
        require(type(inference.get(field)) is int and inference[field] == pixels,
                "Full-support pixel accounting differs: " + field)
    require(type(inference.get("valid_but_unscored_pixel_count")) is int and
            inference["valid_but_unscored_pixel_count"] == 0,
            "Unscored support is outside frozen scope")
    require(all(isinstance(t.get("input_sha256"), str) and
                re.fullmatch(r"[0-9a-f]{64}", t["input_sha256"]) is not None for t in captured),
            "Normalized input SHA256 record differs")
    return grid


def verify_map(logits, scores, evaluated, shape, metadata):
    """Return a numerical receipt for one loaded calibration map; no file I/O."""
    dimensions = _shape(shape)
    grid = check_metadata(shape, metadata)
    require(isinstance(scores, np.ndarray) and scores.dtype == np.dtype(np.float32) and
            scores.shape == dimensions and np.isfinite(scores).all() and
            np.all((scores >= 0.) & (scores <= 1.)), "Saved FP32 score contract differs")
    require(isinstance(evaluated, np.ndarray) and evaluated.dtype == np.dtype(np.bool_) and
            evaluated.shape == dimensions, "Saved boolean evaluated-mask contract differs")
    rebuilt, coverage = reconstruct(logits, shape)
    require(np.array_equal(evaluated, coverage), "Evaluated-mask equality failed")
    errors = np.abs(rebuilt.astype(np.float64) - scores.astype(np.float64))
    maximum = float(np.max(errors))
    require(maximum <= SCORE_TOLERANCE, "Score reconstruction exceeds the frozen 2e-6 limit")
    return {
        "status": "PASS", "shape_yx": list(dimensions), "tiles_verified": len(grid),
        "pixels": int(errors.size), "evaluated_pixels": int(coverage.sum()),
        "evaluated_mask_exact": True, "maximum_absolute_score_error": maximum,
        "mean_absolute_score_error": float(errors.mean()),
        "root_mean_square_score_error": float(np.sqrt(np.mean(errors * errors))),
        "maximum_absolute_score_error_limit": SCORE_TOLERANCE,
        "score_comparison": "absolute numerical tolerance; no bitwise score claim",
    }


def normalize_patch(image):
    """Independent full-patch FP32 robust-MAD specification, with stock fallbacks.

    Uses no mask, clips linear percentiles 1/99, then uses median and MAD*1.4826.
    A tiny/nonfinite MAD falls back to preclip FP32 standard deviation, half
    percentile span, then 1. Arithmetic on the copied array is in-place FP32.
    """
    array = np.array(image, dtype=np.float32, copy=True, order="C")
    require(array.size > 0 and np.isfinite(array).all(), "Finite nonempty normalization input required")
    flat = array.ravel()
    original_std = float(np.std(flat))
    lower = float(np.percentile(flat, 1., method="linear"))
    upper = float(np.percentile(flat, 99., method="linear"))
    np.clip(array, lower, upper, out=array)
    median = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - median)))
    if not np.isfinite(scale) or scale < 1e-6:
        candidates = (abs(original_std), abs(upper - lower) / 2.)
        scale = next((v for v in candidates if np.isfinite(v) and v >= 1e-6), 1.)
    array -= median
    array /= scale
    np.nan_to_num(array, copy=False, nan=0., posinf=0., neginf=0.)
    return array


def normalized_input_hash(image):
    """SHA256 of C-order normalized FP32 tensor bytes, excluding shape headers."""
    normalized = normalize_patch(image)
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def verify_normalized_inputs(surface, shape, metadata):
    """Check all captured normalized hashes from an already loaded uint8 surface.

    This optional helper must be called only when the caller has authorization
    to read actual saved surface arrays. Preparation invokes it on synthetic
    data only. Singleton batch/channel axes do not alter the tensor byte stream.
    """
    dimensions = _shape(shape)
    grid = check_metadata(shape, metadata)
    require(isinstance(surface, np.ndarray) and surface.dtype == np.dtype(np.uint8) and
            surface.shape == (21,) + dimensions, "Original uint8 pooled 21-plane surface required")
    verified = []
    for bbox, captured in zip(grid, metadata["captured_tiles"]):
        x, y, width, height = bbox
        patch = surface[2:19, y:y + height, x:x + width]
        require(np.any(patch), "Entirely zero selected-layer patches are outside scored scope")
        digest = normalized_input_hash(patch)
        require(digest == captured["input_sha256"], "Normalized input bytes differ")
        verified.append({"bbox_roi_xywh": bbox, "input_sha256": digest})
    return {"status": "PASS", "normalized_input_hashes_exact": True,
            "tiles_verified": len(verified), "patch_shape_zyx": [17, 128, 128],
            "tensor_shape": [1, 1, 17, 128, 128], "records": verified}
