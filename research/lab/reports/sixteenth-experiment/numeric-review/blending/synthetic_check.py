"""Synthetic-only fixtures for the proposed FP32 17-plane blend contract."""

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


SHAPE = (384, 384)
GRID = [[x, y, 128, 128] for y in (0, 64, 128, 192, 256)
        for x in (0, 64, 128, 192, 256)]
BALANCED_WORDS = [3212935314, 3198987117, 1051503469, 1065451666]
BALANCED_HASHES = [
    "e5617e0a5c3ded1c241aed593ae56380c150062cf50149fde420aea7e4ac0af8",
    "b71becdd274239325b76513a89ed8c89ab0a46e19c328de2a55b74ca8f8d46f5",
    "fb3bf70e0ffcfd6600f0b62bb847f6659157b113cdd0561e55d3dde70dd94b52",
    "e56cc3f1fe90cb68200aab7105bbac690c81f82a2e11af9e0ae1bcf64ab091d2",
]
LITERAL_TILE_PHASES = [0, 1, 2, 3, 0, 2, 3, 0, 1, 2, 0, 1, 2, 3, 0,
                      2, 3, 0, 1, 2, 0, 1, 2, 3, 0]


def metadata():
    return {
        "adapter_direction": "forward", "adapter_layer_start": 0,
        "inference": {
            "tiles": [{"bbox_roi_xywh": b, "status": "scored", "forward_seconds": .001}
                      for b in GRID],
            "tile_count": 25, "scored_tile_count": 25, "skipped_tile_count": 0,
            "valid_pixel_count": 384 * 384, "evaluated_valid_pixel_count": 384 * 384,
            "valid_but_unscored_pixel_count": 0,
        },
        "captured_tiles": [{"bbox_roi_xywh": b, "input_sha256": BALANCED_HASHES[phase]}
                           for b, phase in zip(GRID, LITERAL_TILE_PHASES)],
    }


def scalar_point(logits, y, x):
    peak = (.5 * (1. - math.cos(2. * math.pi * 63. / 127.))) ** 2
    numerator, denominator = np.float32(0.), np.float32(0.)
    for i, (left, top, width, height) in enumerate(GRID):
        if left <= x < left + width and top <= y < top + height:
            value = float(logits[i, y - top, x - left])
            p = 1. / (1. + math.exp(-value)) if value >= 0. else math.exp(value) / (1. + math.exp(value))
            wy = .5 * (1. - math.cos(2. * math.pi * (y - top) / 127.))
            wx = .5 * (1. - math.cos(2. * math.pi * (x - left) / 127.))
            weight = np.float32(max(wy * wx / peak, .001))
            numerator = np.float32(numerator + np.float32(np.float32(p) * weight))
            denominator = np.float32(denominator + weight)
    return np.float32(numerator / denominator)


def fixtures():
    results = []

    def passed(case, **details):
        results.append({"case": case, "status": "PASS", **details})

    def reject(case, action):
        try:
            action()
        except ValueError as error:
            passed(case, rejected_as_expected=str(error))
        else:
            raise AssertionError("Expected rejection did not occur: " + case)

    require(expected_grid(SHAPE) == GRID and len(GRID) == 25, "Literal row-major tile grid differs")
    passed("literal_all_25_row_major_grid_positions")
    weights = hann_weights()
    require(weights.dtype == np.float32 and np.all(weights[[0, -1], :] == np.float32(.001)) and
            np.all(weights[:, [0, -1]] == np.float32(.001)) and
            np.all(weights[63:65, 63:65] == np.float32(1.)), "Hann floor/peak or dtype differs")
    passed("float64_hann_then_fp32_floor_and_peak")
    require(np.array_equal(sigmoid_probabilities(np.array([-1000., 0., 1000.], dtype=np.float32)),
                           np.array([0., .5, 1.], dtype=np.float32)), "Stable sigmoid extremes differ")
    passed("float64_stable_sigmoid_then_fp32_extremes")
    meta = metadata()
    for logit, p in ((-1000., 0.), (0., .5), (1000., 1.)):
        tile_logits = np.full((25, 128, 128), logit, dtype=np.float32)
        score, mask = reconstruct(tile_logits, SHAPE)
        require(mask.all() and np.array_equal(score, np.full(SHAPE, p, dtype=np.float32)),
                "Exact synthetic full map differs")
        verify_map(tile_logits, score, mask, SHAPE, meta)
        passed("exact_synthetic_full_map_" + str(p))

    yy, xx = np.indices((128, 128), dtype=np.float64)
    varied = np.stack([(i - 12.) / 7. + xx / 83. - yy / 57. for i in range(25)]).astype(np.float32)
    score, mask = reconstruct(varied, SHAPE)
    points = [(0, 0), (63, 64), (127, 127), (128, 128), (191, 255), (255, 191), (320, 320), (383, 383)]
    for y, x in points:
        error = abs(float(score[y, x]) - float(scalar_point(varied, y, x)))
        require(error <= 3e-7, "Independent scalar blending differs")
        passed("independent_asymmetric_scalar_point_" + str(y) + "_" + str(x), absolute_error=error)

    zeros = np.full((25, 128, 128), -1000., dtype=np.float32)
    inside = np.zeros(SHAPE, dtype=np.float32)
    inside[0, 0] = np.float32(SCORE_TOLERANCE)
    require(float(inside[0, 0]) < SCORE_TOLERANCE, "FP32 tolerance straddle differs")
    verify_map(zeros, inside, mask, SHAPE, meta)
    passed("score_error_inside_declared_tolerance", actual_error=float(inside[0, 0]))
    outside = inside.copy()
    outside[0, 0] = np.nextafter(inside[0, 0], np.float32(np.inf))
    reject("score_error_outside_declared_tolerance", lambda: verify_map(zeros, outside, mask, SHAPE, meta))
    false_mask = mask.copy()
    false_mask[0, 0] = False
    reject("evaluated_mask_requires_exact_equality", lambda: verify_map(varied, score, false_mask, SHAPE, meta))
    reject("evaluated_mask_uint8_rejected", lambda: verify_map(varied, score, mask.astype(np.uint8), SHAPE, meta))
    reject("score_float64_rejected", lambda: verify_map(varied, score.astype(np.float64), mask, SHAPE, meta))
    reject("logits_float64_rejected", lambda: reconstruct(varied.astype(np.float64), SHAPE))
    reject("partial_logit_stack_rejected", lambda: reconstruct(varied[:-1], SHAPE))
    reject("old_256_map_shape_rejected", lambda: expected_grid((256, 256)))
    bad_logits = varied.copy()
    bad_logits[0, 0, 0] = np.nan
    reject("nonfinite_logit_rejected", lambda: reconstruct(bad_logits, SHAPE))
    reordered = copy.deepcopy(meta)
    reordered["captured_tiles"].reverse()
    reject("changed_capture_order_rejected", lambda: check_metadata(SHAPE, reordered))
    skipped = copy.deepcopy(meta)
    skipped["inference"]["tiles"][0]["status"] = "skipped"
    reject("skipped_tile_rejected", lambda: check_metadata(SHAPE, skipped))
    old_layers = copy.deepcopy(meta)
    old_layers["adapter_layer_start"] = 2
    reject("old_layers_2_to_19_rejected", lambda: check_metadata(SHAPE, old_layers))
    bool_layer = copy.deepcopy(meta)
    bool_layer["adapter_layer_start"] = False
    reject("boolean_layer_start_rejected", lambda: check_metadata(SHAPE, bool_layer))
    float_layer = copy.deepcopy(meta)
    float_layer["adapter_layer_start"] = 0.
    reject("floating_point_layer_start_rejected", lambda: check_metadata(SHAPE, float_layer))
    reverse = copy.deepcopy(meta)
    reverse["adapter_direction"] = "reverse"
    reject("reverse_depth_metadata_rejected", lambda: check_metadata(SHAPE, reverse))
    shrunken = copy.deepcopy(meta)
    shrunken["inference"]["valid_pixel_count"] -= 1
    reject("shrunken_mask_support_rejected", lambda: check_metadata(SHAPE, shrunken))

    hand = np.array([.125, .375, .625, .875], dtype=np.float32)
    original = hand.copy()
    require(normalized_input_hash(hand) == "9f8e4a66b761faee49792d2f3aa0d2cb46d063d30d2bd4410831dc751d8a2f01",
            "Independent fractional clipping golden differs")
    require(np.array_equal(hand.view(np.uint32), original.view(np.uint32)), "Normalization mutated input")
    passed("fractional_fp32_clipping_golden_and_source_immutability")
    require(normalized_input_hash(hand) != normalized_input_hash(hand.astype(np.uint8)),
            "Fixture does not discriminate uint8 quantization")
    passed("fractional_fp32_discriminates_uint8_quantization")
    constant = np.full((17, 128, 128), .125, dtype=np.float32)
    require(np.array_equal(normalize_patch(constant), np.zeros_like(constant)), "Unit fallback differs")
    passed("constant_nonzero_fp32_unit_scale_fallback")
    first_fallback = np.zeros(100, dtype=np.float32)
    first_fallback[-4:] = .125
    require(normalized_input_hash(first_fallback) ==
            "00da0338da0ca8e606b9e55ce9de8206c7fc934bef28204c4e9c3f387aaa924f",
            "First qualifying preclip std fallback golden differs")
    passed("mad_zero_first_preclip_std_fallback_golden")
    span = np.zeros(100, dtype=np.float32)
    span[-2:] = 5e-6
    expected_span = np.zeros_like(span)
    expected_span[-2:] = 2.
    require(np.array_equal(normalize_patch(span).view(np.uint32), expected_span.view(np.uint32)),
            "Tiny preclip std half-span fallback differs")
    passed("tiny_std_half_percentile_span_fallback")
    pooling_units = (np.array([1., 2., 13., 1024.], dtype=np.float64) / 6939.).astype(np.float32)
    require(normalized_input_hash(pooling_units) ==
            "2c17d2b4afe25cc3a5212269857e895a8d39d730a89e84d5b6b53c0f3820decf",
            "Independent pooling-unit fractional golden differs")
    passed("fractional_pooling_units_normalization_golden")

    zz, full_y, full_x = np.indices((17, 384, 384), dtype=np.int32)
    phases = (zz + full_x // 64 + 2 * (full_y // 64) + full_x) % 4
    surface = np.array([.125, .375, .625, .875], dtype=np.float32)[phases]
    input_before = hashlib.sha256(surface.tobytes(order="C")).hexdigest()
    expected_lookup = np.array(BALANCED_WORDS, dtype=np.uint32).view(np.float32)
    for i, (x, y, w, h) in enumerate(GRID):
        expected = expected_lookup[phases[:, y:y + h, x:x + w]]
        digest = hashlib.sha256(expected.tobytes(order="C")).hexdigest()
        require(digest == BALANCED_HASHES[LITERAL_TILE_PHASES[i]], "Independent full-tile golden assembly differs")
        passed("independent_fractional_byte_hash_tile_" + str(i), bbox_roi_xywh=GRID[i], input_sha256=digest)
    receipt = verify_normalized_inputs(surface, SHAPE, meta)
    require(receipt["tiles_verified"] == 25 and receipt["tensor_shape"] == [1, 1, 17, 128, 128],
            "Full FP32 surface hash verification differs")
    require(hashlib.sha256(surface.tobytes(order="C")).hexdigest() == input_before, "Verifier mutated surface")
    passed("native_fp32_17_plane_all_25_input_hashes_and_immutability")
    patch = surface[:, :128, :128]
    require(np.array_equal(normalize_patch(patch).view(np.uint32), expected_lookup[phases[:, :128, :128]].view(np.uint32)),
            "Fractional normalization word golden differs")
    passed("balanced_fractional_patch_golden_words")
    require(normalized_input_hash(patch[::-1]) ==
            "38e7066b1b10cfdde45eef7b5c2e19ed1282432a2a6b2e8b2bb231cc785c5592",
            "Independent reversed-depth golden differs")
    require(normalized_input_hash(patch[2:17]) ==
            "3e14f7ffba426b77a763c95b73382ce5306bf9c18add70c538f03fff0ce4deb9",
            "Independent old 15-plane slice golden differs")
    passed("byte_hashes_discriminate_reversed_depth_and_old_15_plane_slice")
    reject("quantized_uint8_surface_rejected", lambda: verify_normalized_inputs(surface.astype(np.uint8), SHAPE, meta))
    reject("fp64_surface_rejected", lambda: verify_normalized_inputs(surface.astype(np.float64), SHAPE, meta))
    reject("old_21_plane_surface_rejected", lambda: verify_normalized_inputs(np.ones((21, 384, 384), dtype=np.float32), SHAPE, meta))
    reject("old_15_plane_slice_rejected", lambda: verify_normalized_inputs(surface[2:17], SHAPE, meta))
    reject("reversed_depth_input_bytes_rejected", lambda: verify_normalized_inputs(surface[::-1], SHAPE, meta))
    reject("zero_raw_scored_surface_rejected", lambda: verify_normalized_inputs(np.zeros_like(surface), SHAPE, meta))
    nonfinite = surface.copy()
    nonfinite[0, 0, 0] = np.nan
    reject("nonfinite_fp32_surface_rejected", lambda: verify_normalized_inputs(nonfinite, SHAPE, meta))
    changed_hash = copy.deepcopy(meta)
    changed_hash["captured_tiles"][0]["input_sha256"] = "0" * 64
    reject("normalized_input_byte_mismatch_rejected", lambda: verify_normalized_inputs(surface, SHAPE, changed_hash))
    require("torch" not in sys.modules, "Synthetic preparation imported Torch")
    return {"status": "PASS", "cases": len(results), "checks": results,
            "golden_expectations": "Independent scalar FP32 words and stdlib byte assembly supplied by normalization_review; no author normalization import",
            "actual_retained_array_reads": 0, "model_calls": 0, "Torch_imports": 0,
            "checkpoint_reads": 0, "network_requests": 0}


def source_record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Synthetic receipt must be fresh")
    result = fixtures()
    result["sources"] = [source_record(Path(__file__).with_name("blend_verifier.py")), source_record(__file__)]
    result["numpy_version"] = np.__version__
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "cases": result["cases"], "receipt": source_record(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
