"""Source and synthetic fixtures only for independent calibration blending."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from blend_verifier import (SCORE_TOLERANCE, check_metadata, expected_grid,
                            hann_weights, normalize_patch, normalized_input_hash,
                            reconstruct, require, sigmoid_probabilities,
                            verify_map, verify_normalized_inputs)


def metadata(shape):
    grid = expected_grid(shape)
    pixels = math.prod(shape)
    return {
        "adapter_direction": "forward", "adapter_layer_start": 2,
        "inference": {
            "tiles": [{"bbox_roi_xywh": b, "status": "scored", "forward_seconds": .001}
                      for b in grid],
            "tile_count": len(grid), "scored_tile_count": len(grid), "skipped_tile_count": 0,
            "valid_pixel_count": pixels, "evaluated_valid_pixel_count": pixels,
            "valid_but_unscored_pixel_count": 0,
        },
        "captured_tiles": [{"bbox_roi_xywh": b, "input_sha256": "0" * 64} for b in grid],
    }


def expected_point(logits, grid, y, x):
    """Independent scalar math kernels and explicit FP32 scalar arithmetic."""
    peak = (.5 * (1. - math.cos(2. * math.pi * 63. / 127.))) ** 2
    numerator, denominator = np.float32(0.), np.float32(0.)
    for i, (left, top, width, height) in enumerate(grid):
        if left <= x < left + width and top <= y < top + height:
            z = float(logits[i, y - top, x - left])
            p = 1. / (1. + math.exp(-z)) if z >= 0. else math.exp(z) / (1. + math.exp(z))
            wy = .5 * (1. - math.cos(2. * math.pi * (y - top) / 127.))
            wx = .5 * (1. - math.cos(2. * math.pi * (x - left) / 127.))
            weight = np.float32(max(wy * wx / peak, .001))
            numerator = np.float32(numerator + np.float32(np.float32(p) * weight))
            denominator = np.float32(denominator + weight)
    return np.float32(numerator / denominator)


def run_fixtures():
    results = []

    def passed(name, **details):
        results.append({"case": name, "status": "PASS", **details})

    def reject(name, action):
        try:
            action()
        except ValueError as error:
            passed(name, rejected_as_expected=str(error))
        else:
            raise AssertionError("Expected rejection did not occur: " + name)

    literal256 = [[x, y, 128, 128] for y in (0, 64, 128) for x in (0, 64, 128)]
    literal384 = [[x, y, 128, 128] for y in (0, 64, 128, 192, 256)
                  for x in (0, 64, 128, 192, 256)]
    require(expected_grid((256, 256)) == literal256 and
            expected_grid((384, 384)) == literal384, "Literal grid fixtures differ")
    passed("literal_9_and_25_tile_grids")

    weights = hann_weights()
    require(weights.dtype == np.float32 and weights.shape == (128, 128), "Weight dtype or shape differs")
    require(np.all(weights[[0, -1], :] == np.float32(.001)) and
            np.all(weights[:, [0, -1]] == np.float32(.001)) and
            np.all(weights[63:65, 63:65] == np.float32(1.)), "Hann edge floor or normalized peak differs")
    require(np.allclose(weights, weights[::-1, ::-1], rtol=0., atol=1e-7), "Hann symmetry differs")
    passed("hann_floor_peak_and_symmetry")

    extremes = sigmoid_probabilities(np.array([-1000., -20., 0., 20., 1000.], dtype=np.float32))
    require(extremes.dtype == np.float32 and np.isfinite(extremes).all() and
            np.array_equal(extremes[[0, 2, 4]], np.array([0., .5, 1.], dtype=np.float32)) and
            np.all(np.diff(extremes) >= 0.), "Stable sigmoid extremes differ")
    passed("stable_sigmoid_extremes")

    for shape, grid in [((256, 256), literal256), ((384, 384), literal384)]:
        zeros = np.zeros((len(grid), 128, 128), dtype=np.float32)
        score, mask = reconstruct(zeros, shape)
        require(mask.all() and np.array_equal(score, np.full(shape, .5, dtype=np.float32)),
                "Exact zero-logit map differs")
        receipt = verify_map(zeros, score, mask, shape, metadata(shape))
        require(receipt["maximum_absolute_score_error"] == 0., "Self-consistent map fixture differs")
        passed("zero_logit_full_coverage_" + str(shape[0]), tiles=len(grid))
        for logit, probability in ((-1000., 0.), (1000., 1.)):
            extreme_map, extreme_mask = reconstruct(np.full_like(zeros, logit), shape)
            require(extreme_mask.all() and np.array_equal(extreme_map, np.full(shape, probability, dtype=np.float32)),
                    "Extreme-logit exact full map differs")
            passed("extreme_logit_" + str(int(logit)) + "_full_map_" + str(shape[0]))

        yy, xx = np.indices((128, 128), dtype=np.float64)
        varied = np.stack([(i - len(grid) // 2) / 7. + xx / 83. - yy / 57.
                          for i in range(len(grid))]).astype(np.float32)
        mixed, valid = reconstruct(varied, shape)
        points = [(0, 0), (63, 64), (127, 127), (128, 128),
                  (191, 255), (shape[0] - 1, shape[1] - 1)]
        for y, x in points:
            scalar = expected_point(varied, grid, y, x)
            error = abs(float(mixed[y, x]) - float(scalar))
            require(error <= 3e-7, "Independent scalar point reconstruction differs")
            passed("scalar_point_" + str(shape[0]) + "_" + str(y) + "_" + str(x),
                   maximum_absolute_score_error=error)
        require(valid.all(), "Varied logit coverage differs")

    shape = (256, 256)
    logits = np.zeros((9, 128, 128), dtype=np.float32)
    score = np.full(shape, .5, dtype=np.float32)
    evaluated = np.ones(shape, dtype=np.bool_)
    meta = metadata(shape)
    inside = score.copy()
    boundary = np.float32(.5 + SCORE_TOLERANCE)
    inside[0, 0] = np.nextafter(boundary, np.float32(.5))
    outside = score.copy()
    outside[0, 0] = boundary
    require(abs(float(inside[0, 0]) - .5) < SCORE_TOLERANCE <
            abs(float(outside[0, 0]) - .5), "Float32 tolerance straddle fixture differs")
    accepted = verify_map(logits, inside, evaluated, shape, meta)
    passed("score_error_just_inside_frozen_tolerance",
           actual_error=accepted["maximum_absolute_score_error"])
    reject("score_error_just_outside_frozen_tolerance",
           lambda: verify_map(logits, outside, evaluated, shape, meta))
    zero_logits = np.full_like(logits, -1000.)
    near_zero_inside = np.zeros(shape, dtype=np.float32)
    near_zero_inside[0, 0] = np.float32(SCORE_TOLERANCE)
    require(float(near_zero_inside[0, 0]) < SCORE_TOLERANCE, "Near-zero inside tolerance fixture differs")
    verify_map(zero_logits, near_zero_inside, evaluated, shape, meta)
    passed("score_error_near_zero_inside_frozen_tolerance", actual_error=float(near_zero_inside[0, 0]))
    near_zero_outside = near_zero_inside.copy()
    near_zero_outside[0, 0] = np.nextafter(near_zero_inside[0, 0], np.float32(np.inf))
    reject("score_error_near_zero_outside_frozen_tolerance",
           lambda: verify_map(zero_logits, near_zero_outside, evaluated, shape, meta))
    missing = evaluated.copy()
    missing[0, 0] = False
    reject("evaluated_mask_requires_exact_equality",
           lambda: verify_map(logits, score, missing, shape, meta))
    reject("score_float64_rejected", lambda: verify_map(logits, score.astype(np.float64), evaluated, shape, meta))
    reject("logit_float64_rejected", lambda: reconstruct(logits.astype(np.float64), shape))
    reject("short_logit_stack_rejected", lambda: reconstruct(logits[:-1], shape))
    nonfinite = logits.copy()
    nonfinite[0, 0, 0] = np.nan
    reject("nonfinite_logits_rejected", lambda: reconstruct(nonfinite, shape))
    reject("nonfrozen_shape_rejected", lambda: expected_grid((257, 257)))
    reversed_meta = copy.deepcopy(meta)
    reversed_meta["captured_tiles"].reverse()
    reject("captured_tile_order_rejected", lambda: check_metadata(shape, reversed_meta))
    float_bbox_meta = copy.deepcopy(meta)
    float_bbox_meta["captured_tiles"][0]["bbox_roi_xywh"][0] = 0.
    reject("floating_point_bbox_rejected", lambda: check_metadata(shape, float_bbox_meta))
    skipped_meta = copy.deepcopy(meta)
    skipped_meta["inference"]["tiles"][0]["status"] = "skipped"
    reject("skipped_tile_rejected", lambda: check_metadata(shape, skipped_meta))
    layer_meta = copy.deepcopy(meta)
    layer_meta["adapter_layer_start"] = 3
    reject("changed_layer_start_rejected", lambda: check_metadata(shape, layer_meta))
    hash_meta = copy.deepcopy(meta)
    hash_meta["captured_tiles"][0]["input_sha256"] = "not-a-hash"
    reject("malformed_input_hash_rejected", lambda: check_metadata(shape, hash_meta))

    constant = normalize_patch(np.full((3, 4, 5), 17, dtype=np.uint8))
    require(constant.dtype == np.float32 and np.array_equal(constant, np.zeros_like(constant)),
            "Constant normalization fallback differs")
    passed("constant_normalization_unit_scale_fallback")
    hand_input = np.arange(5, dtype=np.uint8)
    hand = np.array([.04, 1., 2., 3., 3.96], dtype=np.float32)
    hand -= 2.
    hand /= 1.4826
    normalized = normalize_patch(hand_input)
    require(np.array_equal(normalized.view(np.uint32), hand.view(np.uint32)) and
            np.array_equal(hand_input, np.arange(5, dtype=np.uint8)), "Hand MAD or input immutability differs")
    passed("hand_linear_percentiles_mad_and_input_immutability")
    sparse = np.zeros(100, dtype=np.uint8)
    sparse[-1] = 100
    expected_sparse = np.zeros(100, dtype=np.float32)
    expected_sparse[-1] = np.float32(1.) / np.float32(math.sqrt(99.))
    sparse_normalized = normalize_patch(sparse)
    require(np.array_equal(sparse_normalized.view(np.uint32), expected_sparse.view(np.uint32)),
            "Zero-MAD preclip standard-deviation fallback differs")
    passed("zero_mad_preclip_std_fallback")
    sparse255 = np.zeros(100, dtype=np.uint8)
    sparse255[-1] = 255
    require(normalized_input_hash(sparse255) ==
            "83818721ea558ec645a37ed282fa0cef235c85000c36c1f20d88fd1bee6bd682",
            "Independent preclip-standard-deviation golden hash differs")
    passed("independent_sparse255_preclip_std_hash_golden")
    require(normalized_input_hash(np.arange(4, dtype=np.uint8)) ==
            "9f8e4a66b761faee49792d2f3aa0d2cb46d063d30d2bd4410831dc751d8a2f01",
            "Independent even-median/MAD clipping golden hash differs")
    passed("independent_four_value_even_median_hash_golden")
    require(normalized_input_hash(np.full(8, 37, dtype=np.uint8)) ==
            "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925",
            "Independent constant-zero-byte golden hash differs")
    passed("independent_constant8_hash_golden")
    small_span = np.zeros(100, dtype=np.float32)
    small_span[-2:] = 5e-6
    expected_span = np.zeros(100, dtype=np.float32)
    expected_span[-2:] = 2.
    require(np.array_equal(normalize_patch(small_span).view(np.uint32), expected_span.view(np.uint32)),
            "Tiny-preclip-standard-deviation half-span fallback differs")
    passed("general_fp32_helper_half_percentile_span_fallback")
    expected_hash = hashlib.sha256(hand.tobytes(order="C")).hexdigest()
    require(normalized_input_hash(hand_input) == expected_hash, "Normalized byte hash differs")
    require(hashlib.sha256(hand[None, None].tobytes(order="C")).hexdigest() == expected_hash,
            "Singleton tensor axes changed byte hash")
    passed("normalized_c_order_tensor_byte_hash")

    synthetic_surface = np.full((21, 256, 256), 17, dtype=np.uint8)
    input_meta = metadata(shape)
    zero_bytes_hash = hashlib.sha256(bytes(17 * 128 * 128 * 4)).hexdigest()
    for tile in input_meta["captured_tiles"]:
        tile["input_sha256"] = zero_bytes_hash
    inputs_receipt = verify_normalized_inputs(synthetic_surface, shape, input_meta)
    require(inputs_receipt["tiles_verified"] == 9, "Synthetic surface hash accounting differs")
    passed("synthetic_21_plane_surface_all_9_hashes")
    bad_hash_meta = copy.deepcopy(input_meta)
    bad_hash_meta["captured_tiles"][0]["input_sha256"] = "0" * 64
    reject("normalized_input_byte_mismatch_rejected",
           lambda: verify_normalized_inputs(synthetic_surface, shape, bad_hash_meta))
    reject("wrong_surface_dtype_rejected",
           lambda: verify_normalized_inputs(synthetic_surface.astype(np.float32), shape, input_meta))
    reject("zero_raw_patch_rejected",
           lambda: verify_normalized_inputs(np.zeros_like(synthetic_surface), shape, input_meta))
    require("torch" not in sys.modules, "Synthetic preparation loaded Torch")
    return {"status": "PASS", "cases": len(results), "checks": results,
            "actual_retained_array_reads": 0, "model_calls": 0, "Torch_imports": 0,
            "checkpoint_reads": 0, "network_requests": 0}


def source_record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Synthetic output must be fresh")
    result = run_fixtures()
    result["sources"] = [source_record(Path(__file__).with_name("blend_verifier.py")), source_record(__file__)]
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "cases": result["cases"],
                      "receipt": source_record(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
