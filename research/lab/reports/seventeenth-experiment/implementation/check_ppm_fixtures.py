"""Synthetic-only, literal expected-value fixtures for conditional PPM decoding."""
import json
import math

import numpy as np

from regional_ppm import compare_geometry, decode_records, geometry, validate_candidates


def main():
    checks = []

    def checked(name, assertion):
        if not assertion:
            raise AssertionError(name)
        checks.append(name)

    def must_raise(name, call):
        try:
            call()
        except (ValueError, TypeError):
            checks.append(name)
        else:
            raise AssertionError(name)

    xy = np.array([[3, 4], [6, 7]], dtype=np.int64)
    result = np.array([[30, 40, 50, 0, 0, 1], [60, 70, 80, 0, 1, 0]], dtype=np.float64)
    flat = np.array([[3, 4, 32, 0, 0, -1], [6, 7, 32, 0, 0, -1]], dtype=np.float64)
    for order, dtype in (("little", "<f8"), ("big", ">f8")):
        for variant, values in (("result", result), ("surface_volume", flat)):
            raw = values.astype(dtype).tobytes()
            validated = validate_candidates(raw, xy, variant)
            checked(variant + "-" + order + "-unique", validated["passing_candidates"] == [order])
            checked(variant + "-" + order + "-decode", np.array_equal(decode_records(raw, validated), values))
            checked(variant + "-" + order + "-json", bool(json.dumps(validated, allow_nan=False)))
            must_raise(variant + "-" + order + "-changed-bytes-rejected", lambda: decode_records(bytes(len(raw)), validated))

    tolerated = result.copy()
    tolerated[1, 3:] = [0, 0, 1 + 0.5e-5]
    checked("fixed-tolerance-pass", validate_candidates(tolerated.astype("<f8").tobytes(), xy, "result")["selected_byte_order"] == "little")
    tolerated[1, 5] = 1 + 2e-5
    failed = validate_candidates(tolerated.astype("<f8").tobytes(), xy, "result")
    checked("any-record-tolerance-rejects", failed["status"] == "unresolved" and failed["candidates"]["little"]["nonunit_nonsentinel_count"] == 1)
    must_raise("neither-candidate-cannot-decode", lambda: decode_records(tolerated.astype("<f8").tobytes(), failed))

    with_zero = result.copy()
    with_zero[1] = 0
    validated = validate_candidates(with_zero.astype("<f8").tobytes(), xy, "result")
    checked("zero-sentinel-counted-and-permitted", validated["selected_byte_order"] == "little" and validated["candidates"]["little"]["zero_sentinel_count"] == 1)
    only_zero = validate_candidates(bytes(96), xy, "result")
    checked("all-zero-neither-pass", only_zero["passing_candidates"] == [] and only_zero["candidates"]["big"]["zero_sentinel_count"] == 2)
    nonfinite = result.copy()
    nonfinite[1, 0] = np.nan
    validated = validate_candidates(nonfinite.astype("<f8").tobytes(), xy, "result")
    checked("nonfinite-position-rejected", not validated["candidates"]["little"]["passed"] and validated["candidates"]["little"]["nonfinite_nonsentinel_count"] == 1)
    nonfinite[1, 0], nonfinite[1, 4] = 60, np.inf
    validated = validate_candidates(nonfinite.astype("<f8").tobytes(), xy, "result")
    checked("nonfinite-normal-rejected", not validated["candidates"]["little"]["passed"])

    # This palindromic byte pattern has the same near-unit value in both orders.
    palindrome = bytes.fromhex("3f f0 00 00 00 00 f0 3f")
    ambiguous_raw = bytes(40) + palindrome
    ambiguous = validate_candidates(ambiguous_raw, np.array([[0, 0]]), "result")
    checked("both-pass-remains-unresolved", ambiguous["passing_candidates"] == ["little", "big"] and ambiguous["selected_byte_order"] is None)
    must_raise("both-candidates-cannot-decode", lambda: decode_records(ambiguous_raw, ambiguous))

    for index, value, name in ((0, 3.5, "xy"), (2, 31, "depth"), (3, 1e-100, "nx"), (4, 1, "ny"), (5, 0, "sign")):
        wrong = flat.copy()
        wrong[0, index] = value
        checked("flat-exact-" + name, not validate_candidates(wrong.astype("<f8").tobytes(), xy, "surface_volume")["candidates"]["little"]["passed"])
    changed_sign = flat.copy()
    changed_sign[1, 5] = 1
    validated = validate_candidates(changed_sign.astype("<f8").tobytes(), xy, "surface_volume")
    checked("flat-sign-consistency-all-records", not validated["candidates"]["little"]["normal_sign_consistent"] and not validated["candidates"]["little"]["passed"])
    flat_positive = flat.copy()
    flat_positive[:, 5] = 1
    checked("flat-positive-sign-permitted", validate_candidates(flat_positive.astype("<f8").tobytes(), xy, "surface_volume")["selected_byte_order"] == "little")
    checked("flat-all-sentinels-neither-pass", validate_candidates(bytes(96), xy, "surface_volume")["passing_candidates"] == [])
    flat_with_zero = flat.copy()
    flat_with_zero[1] = 0
    for order, dtype in (("little", "<f8"), ("big", ">f8")):
        validated = validate_candidates(flat_with_zero.astype(dtype).tobytes(), xy, "surface_volume")
        checked("flat-" + order + "-sentinel-counted-and-permitted", validated["selected_byte_order"] == order and validated["candidates"][order]["zero_sentinel_count"] == 1 and validated["candidates"][order]["nonsentinel_count"] == 1)
    rows = np.frombuffer(result.astype("<f8").tobytes(), dtype=np.uint8).reshape(2, 48)
    checked("uint8-row-input", validate_candidates(rows, xy, "result.ppm")["selected_byte_order"] == "little")
    must_raise("partial-record-rejected", lambda: validate_candidates(bytes(47), xy, "result"))
    must_raise("wrong-schedule-size-rejected", lambda: validate_candidates(bytes(48), xy, "result"))

    neighborhood = np.zeros((3, 3, 6), dtype=np.float64)
    for y in range(3):
        for x in range(3):
            neighborhood[y, x] = [10 + 2*x, 20 + 3*y, 32, 0, 0, -2]
    g = geometry(neighborhood)
    checked("geometry-axis-central-differences", g["du"] == [2, 0, 0] and g["dv"] == [0, 3, 0])
    checked("geometry-tangent-lengths-area", g["du_norm"] == 2 and g["dv_norm"] == 3 and g["cross_product_area"] == 6)
    checked("geometry-signed-normal-normalized", g["signed_normal_dot"] == -1)
    checked("geometry-orthogonal-tangents", g["tangent_dot"] == 0)
    checked("geometry-all-nine-records", len(g["sampled_records"]) == 3 and all(len(row) == 3 for row in g["sampled_records"]))
    skewed = neighborhood.copy()
    for y in range(3):
        skewed[y, :, 0] += 2*y
    skew = geometry(skewed)
    checked("geometry-skew-dot", math.isclose(skew["tangent_dot"], 2/math.sqrt(13), rel_tol=0, abs_tol=1e-15))
    checked("geometry-skew-area", skew["cross_product_area"] == 6)
    reversed_y = geometry(neighborhood[::-1])
    checked("geometry-raster-y-sign", reversed_y["signed_normal_dot"] == 1)
    degenerate = neighborhood.copy()
    degenerate[:, :, :3] = [1, 2, 3]
    d = geometry(degenerate)
    checked("geometry-zero-denominators-null", d["du_norm"] == 0 and d["cross_product_area"] == 0 and d["signed_normal_dot"] is None and d["tangent_dot"] is None)
    zero_normal = neighborhood.copy()
    zero_normal[1, 1, 3:] = 0
    checked("geometry-zero-normal-only-normal-agreement-null", geometry(zero_normal)["signed_normal_dot"] is None and geometry(zero_normal)["tangent_dot"] == 0)
    sentinel = neighborhood.copy()
    sentinel[1, 1] = 0
    checked("geometry-center-sentinel-no-derivatives", geometry(sentinel)["du"] is None and geometry(sentinel)["center"]["zero_sentinel"])
    checked("geometry-sentinel-is-not-coordinate", geometry(sentinel)["center"]["xyz"] is None)
    sentinel = neighborhood.copy()
    sentinel[1, 0] = 0
    checked("geometry-neighbor-sentinel-invalidates-relevant-derivative", geometry(sentinel)["du"] is None and geometry(sentinel)["dv"] == [0, 3, 0])
    bad = neighborhood.copy()
    bad[1, 2, 0] = np.nan
    invalid = geometry(bad)
    checked("geometry-nonfinite-neighbor-null", invalid["du"] is None and invalid["signed_normal_dot"] is None)
    checked("geometry-json-has-no-nonfinite-numbers", bool(json.dumps(invalid, allow_nan=False)))
    bad = neighborhood.copy()
    bad[1, 2, 3] = np.inf
    checked("geometry-nonfinite-neighbor-normal-invalidates-record", geometry(bad)["du"] is None)
    shifted = neighborhood.copy()
    shifted[:, :, :3] += [3, 4, 0]
    comparison = compare_geometry(g, geometry(shifted))
    checked("same-point-position-comparison", comparison["center_xyz_delta"] == [3, 4, 0] and comparison["center_xyz_distance"] == 5)
    checked("same-point-metric-comparison", comparison["cross_product_area_delta"] == 0 and comparison["signed_normal_dot_delta"] == 0)
    print(json.dumps(dict(status="PASS", synthetic_only=True, actual_input_arrays_decoded=0,
                          count=len(checks), checks=checks), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
