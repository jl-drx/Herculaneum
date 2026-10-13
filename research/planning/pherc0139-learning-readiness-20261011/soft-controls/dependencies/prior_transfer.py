#!/usr/bin/env python3
"""Offline, frozen-CT finite sensitivity of a two-dimensional fine teacher.

`plan` never opens target data. `analyse` requires a separately supplied uint8
crop and metadata. These are conditional XY projections of a 2D teacher; neither
operator verifies an ink-bearing face, target accuracy, or historical overrides.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HALO_XY = (2296, 1784)
NATIVE_J0 = 4
PLANES = tuple(range(10, 18))
PRIMARY_J = 14
FINE_WH = (31040, 24880)
NATIVE_WH = (7980, 6400)
THRESHOLD = 48.0
FIT_ZYX = (0, -1, 0)
OPERATORS = ("mean16_bilinear", "point_bilinear")
LIMITATION = ("Conditional numerical sensitivity of a 2D teacher projected by "
              "CT-derived XY correspondences; not ink accuracy, ink-face/depth "
              "verification, manual-override reconstruction, or admission of "
              "the site. Prior structural screen remains FAIL.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_geometry(packet, candidates_path):
    packet = Path(packet)
    folder = packet / "geometry-review/pullback/site1"
    result = json.loads((folder / "RESULTS.json").read_text())
    expected = {x["name"]: x["sha256"] for x in result["output_files"]}
    for name in ("queries_uvk.npy", "support.npy"):
        if sha256(folder / name) != expected[name]:
            raise ValueError(f"Geometry checksum mismatch: {name}")
    q = np.load(folder / "queries_uvk.npy", allow_pickle=False, mmap_mode="r")
    support = np.load(folder / "support.npy", allow_pickle=False, mmap_mode="r")
    if q.shape != (20, 272, 272, 3) or support.shape != q.shape[:-1]:
        raise ValueError("Unexpected retained site 1 geometry shape")
    if result["native_xywh"] != [2296, 1784, 272, 272]:
        raise ValueError("Unexpected retained site 1 origin")
    candidates = json.loads(Path(candidates_path).read_text())
    if tuple(candidates["fit_translation_zyx"]) != FIT_ZYX:
        raise ValueError("Changed CT fit")
    if [len(x["candidates"]) for x in candidates["supports"]] != [11, 6, 22, 4]:
        raise ValueError("Changed finite candidate counts")
    for entry in candidates["input_files"]:
        if sha256(packet / entry["relative"]) != entry["sha256"]:
            raise ValueError("Candidate-source checksum mismatch")
    return q, support, candidates


def shifted_queries(q, geometry_support, bounds, j, shift):
    """Select query at native (j+dz, y+dy, x+dx), without modifying UV."""
    x0, y0, x1, y1 = bounds
    dz, dy, dx = shift
    iz = j + dz - NATIVE_J0
    ix0, ix1 = x0 + dx - HALO_XY[0], x1 + dx - HALO_XY[0]
    iy0, iy1 = y0 + dy - HALO_XY[1], y1 + dy - HALO_XY[1]
    if not (0 <= iz < q.shape[0] and 0 <= ix0 < ix1 <= q.shape[2]
            and 0 <= iy0 < iy1 <= q.shape[1]):
        raise ValueError("Requested shifted CT query outside retained geometry")
    uv = np.asarray(q[iz, iy0:iy1, ix0:ix1, :2], dtype=np.float64)
    valid = np.asarray(geometry_support[iz, iy0:iy1, ix0:ix1], dtype=bool)
    return uv, valid & np.isfinite(uv).all(axis=-1)


def naive_queries(bounds):
    """Unshifted whole-canvas pixel-centre resize, expressed in group-2 UV."""
    x0, y0, x1, y1 = bounds
    yy, xx = np.mgrid[y0:y1, x0:x1]
    n = np.stack((xx, yy), axis=-1)
    fullfine = (n + 0.5) * (np.array(FINE_WH) / NATIVE_WH) - 0.5
    uv = (fullfine - 1.5) / 4.0
    return uv, np.ones(uv.shape[:-1], dtype=bool)


def query_records(q, geometry_support, candidates):
    for entry in candidates["supports"]:
        sid = entry["support"]
        bounds = entry["native_xy_half_open"]
        variants = [("fit", FIT_ZYX, None), ("geometry_zero", (0, 0, 0), None),
                    ("naive_canvas", None, None)]
        variants += [(f"candidate_{i:02d}", tuple(c["shift_zyx"]), i)
                     for i, c in enumerate(entry["candidates"])]
        for name, shift, index in variants:
            for j in PLANES:
                uv, valid = (naive_queries(bounds) if shift is None else
                             shifted_queries(q, geometry_support, bounds, j, shift))
                record = dict(key=f"s{sid}_{name}_j{j:02d}", support=sid,
                              variant=name, candidate_index=index,
                              native_plane=j, primary=(j == PRIMARY_J),
                              shift_zyx=list(shift) if shift is not None else None,
                              native_xy_half_open=bounds)
                yield record, uv, valid


def source_bounds(uv, valid, operator):
    coords = np.asarray(uv, dtype=np.float64)[valid]
    if not len(coords):
        return None
    if operator == "mean16_bilinear":
        # Each bilinear corner is the exact mean of its own 4 x 4 full-fine pixels.
        low = np.floor(coords).astype(np.int64).min(axis=0) * 4
        high = (np.floor(coords).astype(np.int64).max(axis=0) + 2) * 4
    elif operator == "point_bilinear":
        coords = 4.0 * coords + 1.5
        low = np.floor(coords).astype(np.int64).min(axis=0)
        high = np.floor(coords).astype(np.int64).max(axis=0) + 2
    else:
        raise ValueError("Unknown operator")
    return [int(low[0]), int(low[1]), int(high[0]), int(high[1])]


def footprint_plan(q, support, candidates, packet, candidate_path):
    records = []
    all_bounds = []
    for meta, uv, valid in query_records(q, support, candidates):
        meta["geometry_complete_pixels"] = int(valid.sum())
        meta["source_bounds_fullfine_xy_half_open"] = {}
        for operator in OPERATORS:
            bounds = source_bounds(uv, valid, operator)
            meta["source_bounds_fullfine_xy_half_open"][operator] = bounds
            if bounds is not None:
                all_bounds.append(bounds)
        records.append(meta)
    bounds = np.asarray(all_bounds)
    total = [int(bounds[:, 0].min()), int(bounds[:, 1].min()),
             int(bounds[:, 2].max()), int(bounds[:, 3].max())]
    folder = Path(packet) / "geometry-review/pullback/site1"
    return dict(stage="Target-pixel-free source-footprint plan", records=records,
                source_bbox_fullfine_xy_half_open=total,
                source_shape_yx=[total[3] - total[1], total[2] - total[0]],
                fullfine_wh=list(FINE_WH), native_wh=list(NATIVE_WH),
                candidates_sha256=sha256(candidate_path),
                queries_sha256=sha256(folder / "queries_uvk.npy"),
                geometry_support_sha256=sha256(folder / "support.npy"),
                primary_native_plane=PRIMARY_J, secondary_native_planes=list(PLANES),
                operators=list(OPERATORS), threshold_uint8=THRESHOLD,
                interpolation="Explicit bilinear; no boundary padding or zero fill",
                limitation=LIMITATION)


def pooled_crop(crop, origin_xy):
    """Mean exact aligned blocks in full-fine coordinates, including arbitrary origins."""
    ox, oy = map(int, origin_xy)
    h, w = crop.shape
    gx0, gy0 = -(-ox // 4), -(-oy // 4)
    gx1, gy1 = (ox + w) // 4, (oy + h) // 4
    if gx1 <= gx0 or gy1 <= gy0:
        raise ValueError("Crop contains no complete aligned 4 x 4 blocks")
    aligned = crop[gy0 * 4 - oy:gy1 * 4 - oy, gx0 * 4 - ox:gx1 * 4 - ox]
    pooled = aligned.reshape(gy1 - gy0, 4, gx1 - gx0, 4).mean(axis=(1, 3), dtype=np.float64)
    return pooled, (gx0, gy0)


def bilinear(image, uv, origin_xy, geometry_valid):
    local = np.asarray(uv, dtype=np.float64) - np.asarray(origin_xy)
    finite = np.isfinite(local).all(axis=-1)
    safe = np.where(finite[..., None], local, 0.0)
    base = np.floor(safe).astype(np.int64)
    frac = safe - base
    x, y = base[..., 0], base[..., 1]
    valid = (np.asarray(geometry_valid, dtype=bool) & finite & (x >= 0) & (y >= 0)
             & (x + 1 < image.shape[1]) & (y + 1 < image.shape[0]))
    values = np.full(uv.shape[:-1], np.nan, dtype=np.float64)
    xv, yv = x[valid], y[valid]
    dx, dy = frac[..., 0][valid], frac[..., 1][valid]
    samples = (image[yv, xv] * (1 - dx) * (1 - dy)
               + image[yv, xv + 1] * dx * (1 - dy)
               + image[yv + 1, xv] * (1 - dx) * dy
               + image[yv + 1, xv + 1] * dx * dy)
    if not np.isfinite(samples).all():
        raise ValueError("Nonfinite teacher source data")
    values[valid] = samples
    return values, valid


def sample_teacher(crop, origin_xy, uv, geometry_valid, operator, cached_pool=None):
    if operator == "mean16_bilinear":
        image, origin = cached_pool if cached_pool is not None else pooled_crop(crop, origin_xy)
        return bilinear(image, uv, origin, geometry_valid)
    if operator == "point_bilinear":
        return bilinear(crop, 4.0 * uv + 1.5, origin_xy, geometry_valid)
    raise ValueError("Unknown operator")


def load_native_eligibility(packet):
    """Reassemble only retained native chunks, verifying receipts; never fetch."""
    folder = Path(packet) / "ct-study"
    plan = json.loads((folder / "REQUEST_PLAN.json").read_text())
    ledger = json.loads((folder / "REQUEST_LEDGER.json").read_text())
    if not ledger["complete"] or ledger["plan_sha256"] != sha256(folder / "REQUEST_PLAN.json"):
        raise ValueError("Native CT receipt mismatch")
    records = {r["name"]: r for r in ledger["requests"]}
    site = next(s for s in plan["sites"] if s["site"] == "site1")
    x0, y0, x1, y1 = site["source_xy_half_open"]["native"]
    if [x0, y0, x1, y1] != [2296, 1784, 2568, 2056]:
        raise ValueError("Unexpected native eligibility bounds")
    shape = tuple(plan["volumes"]["native"]["array"]["chunks"])
    if shape[0] != 28:
        raise ValueError("Native eligibility requires all 28 source planes")
    covered = np.zeros((y1-y0, x1-x0), dtype=bool)
    eligible = np.zeros_like(covered)
    used = []
    for req in plan["requests"]:
        if req["site"] != "site1" or req["acquisition"] != "native":
            continue
        p = folder / "chunks" / req["name"]
        rec = records[req["name"]]
        if (rec["status"] != 200 or rec["bytes"] != req["exact_bytes"]
                or p.stat().st_size != req["exact_bytes"] or sha256(p) != rec["sha256"]):
            raise ValueError("Retained native chunk integrity failure")
        chunk = np.fromfile(p, dtype=np.uint8).reshape(shape)
        zz, yy, xx = req["chunk_zyx"]
        if zz != 0:
            raise ValueError("Unexpected native depth chunk")
        ox, oy = xx * shape[2], yy * shape[1]
        xa, xb, ya, yb = max(x0, ox), min(x1, ox+shape[2]), max(y0, oy), min(y1, oy+shape[1])
        eligible[ya-y0:yb-y0, xa-x0:xb-x0] = chunk[:, ya-oy:yb-oy, xa-ox:xb-ox].max(axis=0) > 0
        covered[ya-y0:yb-y0, xa-x0:xb-x0] = True
        used.append(dict(name=req["name"], sha256=rec["sha256"]))
    if not covered.all():
        raise ValueError("Missing native CT source support")
    return eligible, dict(origin_native_xy=[x0, y0], shape_yx=list(eligible.shape),
                          rule="max over all 28 retained native CT planes > 0", chunks=used,
                          request_plan_sha256=sha256(folder / "REQUEST_PLAN.json"),
                          request_ledger_sha256=sha256(folder / "REQUEST_LEDGER.json"))


def compare_targets(base, base_valid, candidate, candidate_valid, eligible=None,
                    metric_domain=None, fixed_positive_union=None):
    complete = base_valid & candidate_valid & np.isfinite(base) & np.isfinite(candidate)
    if eligible is not None:
        complete &= eligible
    a0, b0 = base[complete], candidate[complete]
    sa, sb = a0 > 0, b0 > 0
    nu = int(np.count_nonzero(sa | sb))
    nc = int(np.count_nonzero(sa & sb))
    supervision_common = complete & (base > 0) & (candidate > 0)
    domain = supervision_common if metric_domain is None else supervision_common & metric_domain
    a, b = base[domain], candidate[domain]
    pos_a, pos_b = a >= THRESHOLD, b >= THRESHOLD
    positive_union, positive_common = pos_a | pos_b, pos_a & pos_b
    positive_xor = pos_a ^ pos_b
    common_negative = ~pos_a & ~pos_b
    pu = int(positive_union.sum())
    diff = np.abs(a - b)
    support_change = float(np.count_nonzero(sa ^ sb) / nu) if nu else None
    threshold_change = float(positive_xor.sum() / pu) if pu else None
    active = positive_union if fixed_positive_union is None else fixed_positive_union[domain]
    active_count = int(active.sum())
    mad_positive = float(diff[active].mean()) if active_count else None
    adequate = pu >= 100 and int(common_negative.sum()) >= 100
    return dict(complete_pixels=int(complete.sum()), requested_pixels=int(base.size),
                geometry_or_source_support_xor_pixels=int(np.count_nonzero(base_valid ^ candidate_valid)),
                metric_common_pseudo_supervision_pixels=int(domain.sum()),
                complete_teacher_nonzero_baseline=int(sa.sum()),
                complete_teacher_nonzero_candidate=int(sb.sum()),
                teacher_nonzero_common_pixels=nc,
                teacher_nonzero_union_pixels=nu,
                teacher_nonzero_support_xor_fraction_of_union=support_change,
                positive_baseline_pixels=int(pos_a.sum()), positive_candidate_pixels=int(pos_b.sum()),
                positive_common_pixels=int(positive_common.sum()), positive_union_pixels=pu,
                common_negative_pixels=int(common_negative.sum()),
                positive_xor_pixels=int(positive_xor.sum()),
                gained_positive_pixels=int(np.count_nonzero(~pos_a & pos_b)),
                lost_positive_pixels=int(np.count_nonzero(pos_a & ~pos_b)),
                threshold_xor_fraction_of_positive_union=threshold_change,
                threshold_iou=float(positive_common.sum() / pu) if pu else None,
                mean_absolute_difference_all_uint8=float(diff.mean()) if len(diff) else None,
                mean_absolute_difference_all_is_on_common_pseudo_supervision=True,
                mean_absolute_difference_geometrically_complete_supplement_uint8=float(np.abs(a0-b0).mean()) if len(a0) else None,
                mean_absolute_difference_positive_union_uint8=mad_positive,
                mean_absolute_difference_positive_union_normalized=mad_positive / 255.0 if active_count else None,
                positive_mae_domain_pixels=active_count,
                positive_mae_domain="whole finite-set positive union" if fixed_positive_union is not None else "pair positive union",
                positive_evidence="present" if pu else "no-positive-evidence",
                adequate_positive_union_and_negative_common_counts=adequate,
                operational_material=(support_change is not None and support_change > .01)
                    or (threshold_change is not None and threshold_change > .05)
                    or (mad_positive is not None and mad_positive > .05 * 255),
                routing="conditional_numeric_only" if adequate else "inconclusive_counts",
                no_accuracy_or_site_admission_claim=True)


def finite_set_summary(values, validity, eligible=None):
    values, validity = np.asarray(values), np.asarray(validity)
    complete = validity.all(axis=0) & np.isfinite(values).all(axis=0)
    if eligible is not None:
        complete &= eligible
    s = (values[:, complete] > 0)
    nu, nc = int(s.any(axis=0).sum()), int(s.all(axis=0).sum())
    common = complete & (values > 0).all(axis=0)
    v = values[:, common]
    positive = v >= THRESHOLD
    positive_union = positive.any(axis=0)
    positive_common = positive.all(axis=0)
    negative_common = (~positive).all(axis=0)
    variable = positive_union & ~positive_common
    pu = int(positive_union.sum())
    variable_fraction = float(variable.sum() / pu) if pu else None
    support_change = float((nu-nc) / nu) if nu else None
    score_range = np.ptp(v, axis=0)
    active_range = score_range[positive_union]
    return dict(complete_across_finite_set_pixels=int(complete.sum()),
                common_pseudo_supervision_pixels=int(common.sum()),
                positive_union_pixels=pu, positive_common_pixels=int(positive_common.sum()),
                negative_common_pixels=int(negative_common.sum()),
                threshold_variable_envelope_pixels=int(variable.sum()),
                threshold_variable_envelope_fraction_of_positive_union=variable_fraction,
                teacher_nonzero_union_pixels=nu,
                teacher_nonzero_common_pixels=nc,
                teacher_nonzero_variable_envelope_fraction_of_union=support_change,
                score_range_mean_uint8=float(score_range.mean()) if len(score_range) else None,
                score_range_p95_uint8=float(np.quantile(score_range, .95)) if len(score_range) else None,
                score_range_positive_union_mean_uint8=float(active_range.mean()) if len(active_range) else None,
                score_range_positive_union_p95_uint8=float(np.quantile(active_range, .95)) if len(active_range) else None,
                adequate_positive_union_and_negative_common_counts=pu >= 100
                    and int(negative_common.sum()) >= 100,
                operational_material_envelope=(variable_fraction is not None and variable_fraction > .05)
                    or (support_change is not None and support_change > .01),
                positive_evidence="present" if pu else "no-positive-evidence")


def analyse(q, geometry_support, candidates, teacher_path, metadata_path, out, native_eligible, eligibility_meta):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(Path(metadata_path).read_text())
    origin = metadata["origin_fullfine_xy"]
    if len(origin) != 2 or any(type(x) is not int for x in origin):
        raise ValueError("Crop origin must be two exact full-fine integer coordinates")
    if "crop_sha256" not in metadata or sha256(teacher_path) != metadata["crop_sha256"]:
        raise ValueError("Target crop checksum is absent or mismatched")
    crop = np.load(teacher_path, allow_pickle=False, mmap_mode="r")
    if crop.ndim != 2 or crop.dtype != np.uint8:
        raise ValueError("Expected decoded 2D uint8 teacher crop")
    if list(crop.shape) != metadata["crop_shape_yx"]:
        raise ValueError("Target crop shape differs from metadata")
    cached_pool = pooled_crop(crop, origin)
    records, samples, validity, supervision, per_operator = [], {}, {}, {}, {}
    for record, uv, valid in query_records(q, geometry_support, candidates):
        records.append(record)
        for operator in OPERATORS:
            key = f"{operator}_{record['key']}"
            samples[key], validity[key] = sample_teacher(crop, origin, uv, valid, operator, cached_pool)
            if not validity[key].all():
                raise ValueError(f"Incomplete frozen query footprint: {key}")
            x0, y0, x1, y1 = record["native_xy_half_open"]
            eligible = native_eligible[y0-HALO_XY[1]:y1-HALO_XY[1], x0-HALO_XY[0]:x1-HALO_XY[0]]
            supervision[key] = validity[key] & eligible & (samples[key] > 0)
    for operator in OPERATORS:
        comparisons, envelopes, domains = [], [], {}
        for support in candidates["supports"]:
            sid = support["support"]
            x0, y0, x1, y1 = support["native_xy_half_open"]
            eligible = native_eligible[y0-HALO_XY[1]:y1-HALO_XY[1], x0-HALO_XY[0]:x1-HALO_XY[0]]
            for j in PLANES:
                keys = [f"{operator}_s{sid}_fit_j{j:02d}"] + [
                    f"{operator}_s{sid}_candidate_{i:02d}_j{j:02d}" for i in range(len(support["candidates"]))]
                values = np.asarray([samples[k] for k in keys])
                valid = np.asarray([validity[k] for k in keys])
                common = valid.all(axis=0) & eligible & (values > 0).all(axis=0)
                active = common & (values >= THRESHOLD).any(axis=0)
                domains[(sid, j)] = (eligible, common, active, values, valid)
        for rec in records:
            if rec["variant"] == "fit":
                continue
            key = f"{operator}_{rec['key']}"
            base_variant = "geometry_zero" if rec["variant"] == "naive_canvas" else "fit"
            bk = f"{operator}_s{rec['support']}_{base_variant}_j{rec['native_plane']:02d}"
            comparison = dict(rec)
            comparison["contrast"] = ("geometry_zero_to_naive_canvas" if rec["variant"] == "naive_canvas"
                else "fit_to_geometry_zero" if rec["variant"] == "geometry_zero" else "fit_to_CT_alternative")
            eligible, domain, positive_domain, _, _ = domains[(rec["support"], rec["native_plane"])]
            candidate_contrast = rec["candidate_index"] is not None
            comparison["metrics"] = compare_targets(samples[bk], validity[bk], samples[key], validity[key],
                eligible, domain if candidate_contrast else None, positive_domain if candidate_contrast else None)
            comparisons.append(comparison)
        for support in candidates["supports"]:
            sid = support["support"]
            for j in PLANES:
                eligible, common, active, values, valid = domains[(sid, j)]
                summary = finite_set_summary(values, valid, eligible)
                pairwise = [c["metrics"] for c in comparisons if c["support"] == sid
                            and c["native_plane"] == j and c["candidate_index"] is not None]
                mad = [m["mean_absolute_difference_positive_union_uint8"] for m in pairwise
                       if m["mean_absolute_difference_positive_union_uint8"] is not None]
                summary["worst_candidate_mean_absolute_difference_positive_union_uint8"] = max(mad) if mad else None
                summary["operational_material"] = summary["operational_material_envelope"] or (bool(mad) and max(mad) > .05*255)
                summary.update(support=sid, native_plane=j, primary=j == PRIMARY_J)
                summary["routing"] = ("inconclusive_counts" if not summary["adequate_positive_union_and_negative_common_counts"]
                    else "conditional_material" if summary["operational_material"] else "within_declared_conditional_tolerance")
                envelopes.append(summary)
        per_operator[operator] = dict(comparisons=comparisons, finite_set_summaries=envelopes)
    np.savez_compressed(out / "transferred_teacher_float64_uint8_scale.npz", **samples)
    np.savez_compressed(out / "transferred_teacher_complete_support.npz", **validity)
    np.savez_compressed(out / "transferred_teacher_pseudo_supervision_support.npz", **supervision)
    np.save(out / "native_eligible_max28_gt0.npy", native_eligible)
    result = dict(teacher_crop_sha256=sha256(teacher_path), teacher_metadata_sha256=sha256(metadata_path),
                  teacher_metadata=metadata, records=records, operators=per_operator,
                  native_eligibility_provenance=eligibility_meta,
                  threshold_uint8=THRESHOLD, primary_native_plane=PRIMARY_J,
                  secondary_native_planes=list(PLANES), prior_structural_screen="FAIL; unchanged",
                  operational_thresholds=dict(nonzero_support_change_fraction=.01,
                      threshold_positive_union_change_fraction=.05,
                      positive_union_mean_absolute_difference_normalized=.05,
                      minimum_positive_union_pixels=100, minimum_common_negative_pixels=100),
                  limitation=LIMITATION)
    write_json(out / "RESULTS.json", result)
    write_json(out / "CHECKSUMS.json", {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "eligibility", "analyse"))
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--teacher-metadata", type=Path)
    args = parser.parse_args()
    q, support, candidates = load_geometry(args.packet, args.candidates)
    if args.mode == "plan":
        if args.teacher is not None or args.teacher_metadata is not None:
            parser.error("Planning cannot accept target paths")
        write_json(args.output, footprint_plan(q, support, candidates, args.packet, args.candidates))
    elif args.mode == "eligibility":
        if args.teacher is not None or args.teacher_metadata is not None:
            parser.error("Eligibility preparation cannot accept target paths")
        mask, meta = load_native_eligibility(args.packet)
        np.save(args.output, mask)
        meta["mask_sha256"] = sha256(args.output)
        meta["eligible_pixels"] = int(mask.sum())
        write_json(args.output.with_name("NATIVE_ELIGIBILITY.json"), meta)
    else:
        if args.teacher is None or args.teacher_metadata is None:
            parser.error("Analysis requires an explicit offline crop and metadata")
        native_eligible, eligibility_meta = load_native_eligibility(args.packet)
        analyse(q, support, candidates, args.teacher, args.teacher_metadata, args.output, native_eligible, eligibility_meta)


if __name__ == "__main__":
    main()
