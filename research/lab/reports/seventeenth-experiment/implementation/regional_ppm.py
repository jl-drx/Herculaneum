"""Pure, conditional PPM decoding and geometry under the frozen phase-17 rules.

These helpers perform no file or network reads. Call validation once per object,
with every sampled record from both cases and their complete 3x3 neighborhoods.
A passing invariant is conditional serialization evidence, not historical or
anatomical provenance. Neither candidate is chosen using images or labels.
"""
import hashlib

import numpy as np


NORMAL_TOLERANCE = 1e-5
BYTE_ORDERS = {"little": "<f8", "big": ">f8"}
RECORD_BYTES = 48


def _bytes(records_bytes):
    if isinstance(records_bytes, (bytes, bytearray, memoryview)):
        raw = bytes(records_bytes)
    elif isinstance(records_bytes, np.ndarray) and records_bytes.dtype == np.uint8:
        if records_bytes.ndim != 2 or records_bytes.shape[1] != RECORD_BYTES:
            raise ValueError("Byte arrays must have shape (N, 48)")
        raw = records_bytes.tobytes(order="C")
    else:
        raise TypeError("Records must be bytes or an (N, 48) uint8 array")
    if not raw or len(raw) % RECORD_BYTES:
        raise ValueError("Require a nonempty whole number of 48-byte records")
    return raw


def _number(value):
    return float(value) if np.isfinite(value) else None


def _vector(value):
    return [_number(item) for item in value]


def _norm(values, axis=None):
    # Nonfinite/overflowing norms must be rejected, never silently clipped.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return np.linalg.norm(values, axis=axis)


def _result_candidate(records):
    zero = np.all(records == 0, axis=1)
    finite = np.all(np.isfinite(records), axis=1)
    norms = _norm(records[:, 3:], axis=1)
    with np.errstate(invalid="ignore", over="ignore"):
        errors = np.abs(norms - 1.0)
    unit = np.isfinite(norms) & (errors <= NORMAL_TOLERANCE)
    nonsentinel = ~zero
    nonfinite = nonsentinel & ~finite
    nonunit = nonsentinel & ~unit
    finite_errors = errors[nonsentinel & np.isfinite(errors)]
    passed = bool(nonsentinel.any() and not nonfinite.any() and not nonunit.any())
    return dict(passed=passed, invariant="finite_nonzero_records_with_unit_normals",
                record_count=int(len(records)), zero_sentinel_count=int(zero.sum()),
                nonsentinel_count=int(nonsentinel.sum()),
                nonfinite_nonsentinel_count=int(nonfinite.sum()),
                nonunit_nonsentinel_count=int(nonunit.sum()),
                nonfinite_normal_norm_count=int((nonsentinel & ~np.isfinite(norms)).sum()),
                finite_normal_norm_error_max=float(finite_errors.max()) if finite_errors.size else None,
                normal_norm_absolute_tolerance=NORMAL_TOLERANCE)


def _flat_candidate(records, expected_xy):
    zero = np.all(records == 0, axis=1)
    nonsentinel = ~zero
    finite = np.all(np.isfinite(records), axis=1)
    xy = np.all(records[:, :2] == expected_xy, axis=1)
    constants = ((records[:, 2] == 32) & (records[:, 3] == 0) &
                 (records[:, 4] == 0))
    sign_valid = (records[:, 5] == 1) | (records[:, 5] == -1)
    all_positive = bool(nonsentinel.any() and np.all(records[nonsentinel, 5] == 1))
    all_negative = bool(nonsentinel.any() and np.all(records[nonsentinel, 5] == -1))
    sign_consistent = all_positive or all_negative
    passed = bool(nonsentinel.any() and finite[nonsentinel].all() and xy[nonsentinel].all()
                  and constants[nonsentinel].all() and sign_valid[nonsentinel].all()
                  and sign_consistent)
    return dict(passed=passed, invariant="exact_scheduled_flat_map_x_y_32_0_0_constant_sign",
                record_count=int(len(records)), zero_sentinel_count=int(zero.sum()),
                nonsentinel_count=int(nonsentinel.sum()),
                nonfinite_count=int((nonsentinel & ~finite).sum()),
                xy_mismatch_count=int((nonsentinel & ~xy).sum()),
                constant_component_mismatch_count=int((nonsentinel & ~constants).sum()),
                invalid_sign_count=int((nonsentinel & ~sign_valid).sum()),
                normal_sign_consistent=sign_consistent,
                normal_sign=1 if all_positive else (-1 if all_negative else None))


def validate_candidates(records_bytes, expected_xy, variant):
    """Test exactly little/big float64 over ALL supplied records of one object.

    ``expected_xy`` is independently scheduled native coordinates, shape (N, 2).
    It is checked structurally for both variants and numerically for flat maps.
    No geometry is permitted unless exactly one complete candidate passes.
    The returned record hash binds downstream decoding to these same bytes.
    """
    raw = _bytes(records_bytes)
    count = len(raw) // RECORD_BYTES
    xy = np.asarray(expected_xy)
    if xy.shape != (count, 2) or xy.dtype.kind not in "iuf":
        raise ValueError("Expected XY must be a numeric (N, 2) array")
    if not np.isfinite(xy).all() or not np.equal(xy, np.floor(xy)).all():
        raise ValueError("Expected native XY must be finite integer coordinates")
    if np.any(np.abs(xy) > 2**53):
        raise ValueError("Expected native XY must be exactly representable as float64")
    canonical = str(variant).removesuffix(".ppm")
    if canonical not in ("result", "surface_volume"):
        raise ValueError("Variant must be result or surface_volume")
    candidates = {}
    for order, dtype in BYTE_ORDERS.items():
        records = np.frombuffer(raw, dtype=dtype).reshape(count, 6)
        summary = (_result_candidate(records) if canonical == "result" else
                   _flat_candidate(records, xy))
        candidates[order] = dict(byte_order=order, dtype=dtype, **summary)
    passing = [order for order, summary in candidates.items() if summary["passed"]]
    selected = passing[0] if len(passing) == 1 else None
    return dict(variant=canonical, record_count=count, byte_count=len(raw),
                records_sha256=hashlib.sha256(raw).hexdigest(), candidates=candidates,
                passing_candidates=passing,
                status="conditional" if selected is not None else "unresolved",
                selected_byte_order=selected,
                interpretation="invariant_consistent_serialization_only" if selected else None)


def decode_records(records_bytes, validation):
    """Decode the same bytes only after exactly one object-level invariant passed."""
    raw = _bytes(records_bytes)
    selected = validation.get("selected_byte_order")
    if (validation.get("status") != "conditional" or selected not in BYTE_ORDERS or
            validation.get("passing_candidates") != [selected]):
        raise ValueError("Unresolved serialization: numerical geometry is forbidden")
    if (hashlib.sha256(raw).hexdigest() != validation.get("records_sha256") or
            len(raw) != validation.get("byte_count")):
        raise ValueError("Decoding bytes differ from the validated object samples")
    return np.frombuffer(raw, dtype=BYTE_ORDERS[selected]).reshape(-1, 6).astype(np.float64)


def _record_summary(record):
    xyz, normal = record[:3], record[3:]
    norm = _norm(normal)
    sentinel = bool(np.all(record == 0))
    return dict(xyz=None if sentinel else _vector(xyz), normal=_vector(normal),
                xyz_finite=bool(np.isfinite(xyz).all()),
                normal_finite=bool(np.isfinite(normal).all()),
                all_components_finite=bool(np.isfinite(record).all()),
                zero_sentinel=sentinel, mapped=not sentinel,
                zero_xyz=bool(np.all(xyz == 0)), zero_normal=bool(np.all(normal == 0)),
                normal_norm=_number(norm), normal_norm_zero=bool(norm == 0))


def geometry(neighborhood):
    """Describe one decoded 3x3x6 neighborhood, with raster y increasing down.

    Exact six-zero sentinels invalidate the center or the relevant derivative.
    Nonfinite records likewise invalidate a relevant derivative. Zero/nonfinite
    denominator norms produce null agreement values. Every sampled record has
    a separate summary; diagonals do not enter central finite differences.
    """
    records = np.asarray(neighborhood, dtype=np.float64)
    if records.shape != (3, 3, 6):
        raise ValueError("Geometry requires exactly one (3, 3, 6) neighborhood")
    summaries = [[_record_summary(records[y, x]) for x in range(3)] for y in range(3)]
    center = summaries[1][1]
    center_valid = center["all_components_finite"] and not center["zero_sentinel"]
    usable = [[r["all_components_finite"] and not r["zero_sentinel"] for r in row] for row in summaries]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        du = (records[1, 2, :3] - records[1, 0, :3]) / 2 if center_valid and usable[1][0] and usable[1][2] else None
        dv = (records[2, 1, :3] - records[0, 1, :3]) / 2 if center_valid and usable[0][1] and usable[2][1] else None
    du = du if du is not None and np.isfinite(du).all() else None
    dv = dv if dv is not None and np.isfinite(dv).all() else None
    u_norm = _norm(du) if du is not None else np.nan
    v_norm = _norm(dv) if dv is not None else np.nan
    normal = records[1, 1, 3:]
    n_norm = _norm(normal)
    cross = None
    area = np.nan
    normal_dot = tangent_dot = None
    if du is not None and dv is not None:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            cross = np.cross(du, dv)
            area = _norm(cross)
            if np.isfinite(u_norm) and u_norm > 0 and np.isfinite(v_norm) and v_norm > 0:
                tangent_dot = _number(np.dot(du / u_norm, dv / v_norm))
            if np.isfinite(area) and area > 0 and np.isfinite(n_norm) and n_norm > 0:
                normal_dot = _number(np.dot(cross / area, normal / n_norm))
    return dict(center=center, sampled_records=summaries, valid_center=bool(center_valid),
                du=_vector(du) if du is not None else None,
                dv=_vector(dv) if dv is not None else None,
                du_norm=_number(u_norm), dv_norm=_number(v_norm),
                cross_product=_vector(cross) if cross is not None else None,
                cross_product_area=_number(area), signed_normal_dot=normal_dot,
                tangent_dot=tangent_dot,
                degenerate=dict(zero_du=bool(u_norm == 0), zero_dv=bool(v_norm == 0),
                                zero_cross_area=bool(area == 0), zero_normal=bool(n_norm == 0),
                                du_undefined=du is None, dv_undefined=dv is None,
                                normal_agreement_undefined=normal_dot is None,
                                tangent_agreement_undefined=tangent_dot is None))


def compare_geometry(first, second):
    """Report same-point second-minus-first comparisons without selecting a map."""
    first_xyz, second_xyz = first["center"]["xyz"], second["center"]["xyz"]
    delta = None
    if first["valid_center"] and second["valid_center"]:
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.asarray(second_xyz) - np.asarray(first_xyz)
        if np.isfinite(difference).all():
            delta = difference
    out = dict(comparison="second_minus_first_at_identical_scheduled_native_xy",
               center_xyz_delta=_vector(delta) if delta is not None else None,
               center_xyz_distance=_number(_norm(delta)) if delta is not None else None)
    for key in ("du_norm", "dv_norm", "cross_product_area", "signed_normal_dot", "tangent_dot"):
        a, b = first[key], second[key]
        with np.errstate(over="ignore", invalid="ignore"):
            out[key + "_delta"] = _number(np.float64(b) - np.float64(a)) if a is not None and b is not None else None
    return out
