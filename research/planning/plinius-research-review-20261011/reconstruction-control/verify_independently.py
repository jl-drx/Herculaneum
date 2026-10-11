"""Read-only independent replay of the retained reconstruction-control metrics.

Requirements: Python 3.12+, NumPy, SciPy. Tested versions are recorded in output.
Uses only saved files; no network, model inference, or imports from analyse.py.
Run: python verify_independently.py > INDEPENDENT_VERIFICATION.json
Optional: --root /path/to/reconstruction-control (metadata must be its sibling).
The script writes only JSON to stdout. All numerical libraries use one thread.
"""
import os

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.fft import fftn, ifftn


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histogram_correlation(histogram, x, y):
    """Weighted uncentered moments, independently of demeaned vector products."""
    h = histogram.astype(np.float64)
    n = h.sum()
    ha, hb = h.sum(1), h.sum(0)
    sx, sy = np.dot(ha, x), np.dot(hb, y)
    sxx, syy = np.dot(ha, x * x), np.dot(hb, y * y)
    sxy = np.einsum("i,ij,j->", x, h, y)
    return float((n * sxy - sx * sy)
                 / math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy)))


def joint_histogram(a, b, levels):
    codes = (a.astype(np.int32) * levels + b.astype(np.int32)).ravel()
    return np.bincount(codes, minlength=levels ** 2).reshape(levels, levels)


def intensity_metrics(a, b):
    h = joint_histogram(a, b, 256)
    v = np.arange(256, dtype=np.float64)
    ar, br = h.sum(1), h.sum(0)
    # A constant rank offset cancels from Pearson correlation.
    xr, yr = np.cumsum(ar) - ar / 2, np.cumsum(br) - br / 2
    return (histogram_correlation(h, v, v),
            histogram_correlation(h, xr, yr),
            float(np.sum(h * np.abs(v[:, None] - v[None, :])) / h.sum()))


def verify(root):
    freeze = json.loads((root / "SOURCE_FREEZE.json").read_text())
    plan = json.loads((root / "REQUEST_PLAN.json").read_text())
    ledger = json.loads((root / "REQUEST_LEDGER.json").read_text())
    result = json.loads((root / "RESULTS.json").read_text())
    for filename, expected in freeze["sha256"].items():
        assert sha256(root / filename) == expected, filename
    assert freeze["created_utc"] < min(q["started_utc"] for q in ledger)
    assert len(ledger) == len(plan["requests"]) == 9

    catalog_path = root.parent / "metadata" / "CATALOG_EXTRACT.json"
    assert sha256(catalog_path) == plan["catalog_extract_sha256"]
    catalog = json.loads(catalog_path.read_text())
    transform = next(t for t in catalog["volumes"]["20260102150214"]
                     ["properties"]["transforms"]
                     if t["to_volume_id"] == "20250728140407")
    for i, site in enumerate(plan["sites"]):
        assert site["landmark_xyz"] == transform["to_landmarks"][i]
        index = [int(v // 128) for v in site["landmark_xyz"][::-1]]
        assert index == site["chunk_index_zyx"]
        assert site["bounds_zyx_half_open"] == [[128*v, 128*(v+1)] for v in index]
        z0, z1 = site["bounds_zyx_half_open"][0]
        assert (z1 <= 1144 or z0 > 1310) and z1 <= 20448

    cubes = {}
    for planned, retained in zip(plan["requests"], ledger):
        assert all(planned[k] == retained[k] for k in planned)
        body = (root / "chunks" / planned["name"]).read_bytes()
        assert len(body) == retained["bytes"] == 128 ** 3
        assert hashlib.sha256(body).hexdigest() == retained["sha256"]
        assert retained["status"] == 200
        cubes[planned["site"], planned["volume"]] = np.frombuffer(
            body, np.uint8).reshape(128, 128, 128)
    assert sum(row["bytes"] for row in ledger) == 18874368

    errors = dict.fromkeys(["pearson", "spearman", "mae", "gradient",
                           "phase_peak", "phase_far"], 0.0)
    phase_rows = []
    for retained in result["comparisons"]:
        a = cubes[retained["site"], retained["volume_a"]]
        b = cubes[retained["site"], retained["volume_b"]]
        mask = (a != 0) & (b != 0)
        assert int(mask.sum()) == retained["common_nonzero_count"]
        assert float(np.mean((a != 0) == (b != 0))) == retained["nonzero_mask_agreement"]
        for key, selected in [("all", np.ones(a.shape, dtype=bool)),
                              ("common_nonzero", mask)]:
            values = intensity_metrics(a[selected], b[selected])
            for name, value in zip(["pearson", "spearman", "mae"], values):
                field = "mae_uint8" if name == "mae" else name
                error = abs(value - retained[key][field])
                errors[name] = max(errors[name], error)
                assert error < 1e-11, (key, name, error)

        gh = np.zeros((511, 511), np.int64)
        for axis in range(3):
            good = (np.take(mask, range(127), axis=axis)
                    & np.take(mask, range(1, 128), axis=axis))
            da = np.diff(a.astype(np.int16), axis=axis)[good] + 255
            db = np.diff(b.astype(np.int16), axis=axis)[good] + 255
            gh += joint_histogram(da, db, 511)
        assert int(gh.sum()) == retained["gradient_common_edges"]
        gradient = histogram_correlation(gh, np.arange(-255, 256, dtype=float),
                                         np.arange(-255, 256, dtype=float))
        error = abs(gradient - retained["gradient_pearson_common_nonzero"])
        errors["gradient"] = max(errors["gradient"], error)
        assert error < 1e-11

        w = np.hanning(128)
        taper = w[:, None, None] * w[None, :, None] * w[None, None, :]
        fa = fftn((a - a.mean()) * taper, workers=1)
        fb = fftn((b - b.mean()) * taper, workers=1)
        cross = fa * fb.conjugate()
        spectrum = np.where(np.abs(cross) > 0, np.exp(1j * np.angle(cross)), 0)
        correlation = ifftn(spectrum, workers=1).real
        index = np.unravel_index(correlation.argmax(), correlation.shape)
        shift = [int((i + 64) % 128 - 64) for i in index]
        expected_phase = retained["phase_correlation"]
        assert shift == expected_phase["shift_to_apply_to_b_zyx"]
        peak = float(correlation[index])
        errors["phase_peak"] = max(errors["phase_peak"],
                                   abs(peak - expected_phase["peak"]))
        # Independent modular L-infinity distance, not indexed cube assignment.
        grids = np.indices(correlation.shape)
        distances = np.maximum.reduce([
            np.abs((grids[k] - index[k] + 64) % 128 - 64) for k in range(3)])
        far = float(correlation[distances > 2].max())
        errors["phase_far"] = max(errors["phase_far"],
                                  abs(far - expected_phase["second_peak_outside_radius2"]))
        phase_rows.append({"site": retained["site"], "a": retained["volume_a"],
                           "b": retained["volume_b"], "shift_zyx": shift,
                           "corr_zero": float(correlation[0, 0, 0]), "peak": peak,
                           "peak_minus_zero": peak - float(correlation[0, 0, 0]),
                           "zero_to_peak_ratio": float(correlation[0, 0, 0]) / peak,
                           "gradient": gradient})
    assert errors["phase_peak"] < 1e-11 and errors["phase_far"] < 1e-11

    maxcode_rows = []
    for retained in result["cubes"]:
        a = cubes[retained["site"], retained["volume"]].ravel()
        h = np.bincount(a, minlength=256)
        v = np.arange(256, dtype=float)
        n = len(a)
        mean = float(np.dot(h, v) / n)
        std = float(np.sqrt(np.dot(h, v*v) / n - mean*mean))
        assert abs(mean - retained["mean"]) < 1e-12
        assert abs(std - retained["std"]) < 1e-11
        assert 1 - int(h[0]) / n == retained["nonzero_fraction"]
        assert int(h[255]) / n == retained["saturated255_fraction"]
        cumulative, quantiles = np.cumsum(h), []
        for percentile in [0, 1, 10, 50, 90, 99, 100]:
            z = (n - 1) * percentile / 100
            lo, hi = int(np.floor(z)), int(np.ceil(z))
            vl = int(np.searchsorted(cumulative, lo + 1))
            vh = int(np.searchsorted(cumulative, hi + 1))
            quantiles.append(vl + (vh - vl) * (z - lo))
        assert quantiles == retained["quantiles_0_1_10_50_90_99_100"]
        maximum = int(np.flatnonzero(h)[-1])
        maxcode_rows.append({"site": retained["site"], "volume": retained["volume"],
                             "maxcode": maximum, "maxcode_fraction": float(h[maximum] / n)})

    metadata_sources = {}
    for volume in ["20250728140407", "20251107132835", "20251107135911"]:
        path = (root.parent / "metadata" / "sampling-metadata" / "responses"
                / (volume + "-0-zarray.json"))
        z = json.loads(path.read_text())
        assert z["dtype"] == "|u1" and z["order"] == "C"
        assert z["chunks"] == [128, 128, 128]
        assert z["compressor"] is None and z["filters"] is None
        assert z["dimension_separator"] == "/"
        metadata_sources[str(path.relative_to(root.parent))] = sha256(path)

    return {
        "status": "PASS",
        "scope": "Independent replay of existing numerical comparisons; no new acquisition/inference.",
        "reviewer_script_sha256": sha256(Path(__file__).resolve()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "numerical_threads": 1},
        "inputs": {"RESULTS.json": sha256(root / "RESULTS.json"),
                   "SOURCE_FREEZE.json": sha256(root / "SOURCE_FREEZE.json"),
                   "REQUEST_LEDGER.json": sha256(root / "REQUEST_LEDGER.json"),
                   "catalog_excerpt": sha256(catalog_path),
                   "array_metadata": metadata_sources},
        "all9_chunks_and_requests_verified": True,
        "all5_freeze_hashes_verified": True,
        "all3_landmark_selections_verified": True,
        "all9_cube_summary_metrics": "PASS",
        "array_metadata_encoding": "PASS",
        "comparisons": len(phase_rows),
        "distinct_formula_max_abs_errors": errors,
        "phase_audit_rows": phase_rows,
        "supplemental_literal_maxcode_occupancy": maxcode_rows,
        "limits": ["Does not independently certify acquisition geometry or ink truth.",
                   "Common nonzero masks are intensity supports, not tissue truth.",
                   "Code255 and maximum-code occupancy are not calibrated clipping measures.",
                   "Image review is recorded in REVIEW.md, not automated by this checker."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2, allow_nan=False))
