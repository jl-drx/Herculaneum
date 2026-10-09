"""Pinned sparse PPM invariant validation and conditional geometry; no network."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import io
from pathlib import Path
import resource
import signal
import socket
import time
import json


def offline(*args, **kwargs):
    raise RuntimeError("Network disabled for retained sparse PPM diagnostic")


socket.create_connection = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline

import numpy as np
from regional_common import checked_bytes, checked_json, record, require, verify, write_json
from regional_ppm import validate_candidates, decode_records, geometry, compare_geometry


AXIS = [32, 72, 112, 152, 192, 232, 272, 312, 351]
PROTOCOLS = dict(protocol="08bc48f203d6722216c70d54b961d0d29ce59204359da49d2b6110661a9a9cc1",
                clarifications="22ed3e0749d56bd61113b6a2cd74f6ae7373082a34a1c090352ccba99eb2c643",
                ppm_amendment="ed3096856274505990fb9b4069ce6fdbb657eaeb93ae52b6bfae2144de6d9f9e",
                ppm_zero_clarification="114e3791f2a7549b2e1d82192ef454865224f3ea9506550a0513f4eaec591d1a")


def assemble_request(body, receipt, request, contract, case):
    """Reconstruct opaque neighborhoods from independent byte/coordinate ranges."""
    require(len(body) <= 222640, "Encoded multipart cap exceeded")
    x0, y0, width, height = case["bbox_source_xywh"]
    require(case["native_bbox_yxhw"] == [y0, x0, height, width] and (width, height) == (1152, 1152), "Frozen crop differs")
    require(contract["dim"] == 6 and contract["type"] == "double" and contract["ordered"] is True
            and contract["record_bytes"] == 48, "Ordered six-double PPM required")
    require(contract["object_total_bytes"] == contract["data_offset"] + contract["width"] * contract["height"] * 48,
            "PPM extent inconsistent with header")
    for key in ("request_id", "object_id", "case_id", "variant", "url"):
        require(receipt[key] == request[key], f"Sample route differs: {key}")
    require(request["object_id"] == contract["object_id"] and request["url"] == contract["url"]
            and receipt["etag"] == contract["etag"] == request["pin"]["etag"]
            and receipt["object_total_bytes"] == request["pin"]["object_total_bytes"] == contract["object_total_bytes"],
            "Pinned object identity differs")
    require(request["case_id"] == case["id"], "Case route differs")
    expected_points, expected_ranges, expected_xy = [], [], []
    for cy in AXIS:
        for cx in AXIS:
            y, x = y0 + 3 * cy + 1, x0 + 3 * cx + 1
            ranges = []
            for dy in (-1, 0, 1):
                require(0 <= y + dy < contract["height"] and 0 <= x - 1 < x + 1 < contract["width"], "Sample neighborhood escapes PPM canvas")
                start = int(contract["data_offset"]) + 48 * ((y + dy) * int(contract["width"]) + x - 1)
                ranges.append([start, start + 143])
                expected_xy.extend([[x + dx, y + dy] for dx in (-1, 0, 1)])
            expected_points.append(dict(coarse_yx=[cy, cx], native_center_yx=[y, x], row_ranges=ranges))
            expected_ranges.extend(ranges)
    require(request["points"] == expected_points and request["ranges"] == expected_ranges,
            "Sparse point/range schedule differs from frozen integer mapping")
    require(len(receipt["parts"]) == len(expected_ranges) == 243, "Exactly 243 physical row parts required")
    parts = {tuple(part["range_inclusive"]): part for part in receipt["parts"]}
    require(len(parts) == 243 and set(parts) == {tuple(span) for span in expected_ranges}, "Part range coverage differs")
    intervals = []
    chunks = []
    for span in expected_ranges:
        part = parts[tuple(span)]
        start, size = part["payload_offset"], part["payload_bytes"]
        require(type(start) is int and type(size) is int and start >= 0 and size == 144 and start + size <= len(body),
                "Physical payload interval differs")
        intervals.append((start, start + size))
        require(part["headers"].get("content-range") == f"bytes {span[0]}-{span[1]}/{contract['object_total_bytes']}",
                "Physical part source header differs")
        payload = body[start:start + size]
        require(hashlib.sha256(payload).hexdigest() == part["payload_sha256"], "Opaque row payload SHA differs")
        chunks.append(payload)
    intervals.sort()
    require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), "Overlapping multipart payloads")
    raw = b"".join(chunks)
    require(len(raw) == 34992 and len(expected_xy) == 729, "Sparse useful-byte/record count differs")
    return raw, np.asarray(expected_xy, dtype=np.int64), expected_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    signal.alarm(600)
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1200))
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2, 256 * 1024**2))
    plan_pin = record(args.plan)
    plan = checked_json(plan_pin)
    require(plan["kind"] == "phase17-frozen-sparse-ppm-diagnostic", "Unexpected PPM plan")
    pins = {plan_pin["path"]: plan_pin}
    for key, expected in PROTOCOLS.items():
        require(plan[key]["sha256"] == expected, "Protocol/amendment SHA differs")
    for pin in [plan[k] for k in PROTOCOLS] + plan["code"] + [plan["descriptor"], plan["acquisition_result"], plan["acquisition_plan"], plan["source_audit"]]:
        verify(pin)
        pins[pin["path"]] = pin
    acquisition = checked_json(plan["acquisition_result"])
    sampling = checked_json(plan["acquisition_plan"])
    descriptor = checked_json(plan["descriptor"])
    require(acquisition["status"] == "PASS" and acquisition["stage"] == "samples", "Sparse acquisition is incomplete")
    require(acquisition["plan"]["sha256"] == plan["acquisition_plan"]["sha256"], "Sample plan/result linkage differs")
    cases = {case["id"]: case for case in descriptor["cases"]}
    expected_cases = [f"frag{fragment}-context{context}" for fragment in (1, 2, 3) for context in (1, 2)]
    require(list(cases) == expected_cases, "Frozen case inventory differs")
    contracts = {c["object_id"]: c for c in sampling["ppm_contracts"]}
    object_ids = [f"frag{fragment}-{variant}" for fragment in (1, 2, 3) for variant in ("result", "surface_volume")]
    require(set(contracts) == set(object_ids), "PPM object inventory differs")
    requests = {request["request_id"]: request for request in sampling["requests"]}
    receipts = {receipt["request_id"]: receipt for receipt in acquisition["objects"]}
    require(len(requests) == len(receipts) == 12 and set(requests) == set(receipts), "Sparse request inventory differs")
    # Freeze and verify every raw multipart object before any float64 interpretation.
    for receipt in receipts.values():
        verify(receipt["file"])
        pins[receipt["file"]["path"]] = receipt["file"]
    for contract in contracts.values():
        for key in ("prefix", "format_source"):
            verify(contract[key])
            pins[contract[key]["path"]] = contract[key]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs, geometry_index = [], {}
    useful_bytes = 0
    for object_id in object_ids:
        contract = contracts[object_id]
        fragment, variant = object_id.split("-", 1)
        case_ids = [f"{fragment}-context{number}" for number in (1, 2)]
        packages = []
        for case_id in case_ids:
            matching = [request for request in requests.values() if request["case_id"] == case_id and request["object_id"] == object_id]
            require(len(matching) == 1, "Missing or duplicate case/object request")
            request = matching[0]
            receipt = receipts[request["request_id"]]
            body = checked_bytes(receipt["file"], 222640)
            raw, expected_xy, points = assemble_request(body, receipt, request, contract, cases[case_id])
            useful_bytes += len(raw)
            packages.append(dict(case_id=case_id, raw=raw, expected_xy=expected_xy, points=points))
        object_raw = b"".join(package["raw"] for package in packages)
        object_xy = np.concatenate([package["expected_xy"] for package in packages])
        validation = validate_candidates(object_raw, object_xy, variant)
        require(validation["record_count"] == 1458, "All two-case 3x3 neighborhoods required for object validation")
        result = dict(object_id=object_id, variant=variant, contract=contract, validation=validation,
                      source_case_ids=case_ids, all_sampled_records_validated=True,
                      geometry_status="computed_conditionally" if validation["status"] == "conditional" else "omitted_unresolved_serialization",
                      cases=[])
        if validation["status"] == "conditional":
            decoded = decode_records(object_raw, validation).reshape(2, 81, 3, 3, 6)
            for case_number, package in enumerate(packages):
                points = []
                for point_number, point in enumerate(package["points"]):
                    output = dict(coarse_yx=point["coarse_yx"], native_center_yx=point["native_center_yx"],
                                  geometry=geometry(decoded[case_number, point_number]))
                    points.append(output)
                    geometry_index[(package["case_id"], variant, tuple(point["coarse_yx"]))] = output["geometry"]
                result["cases"].append(dict(id=package["case_id"], point_count=81, sampled_record_count=729, points=points))
        else:
            for package in packages:
                result["cases"].append(dict(id=package["case_id"], point_count=81, sampled_record_count=729,
                                           points=None, geometry_omission_reason="Object serialization invariant did not select exactly one candidate"))
        outputs.append(result)
        write_json(args.output_dir / (object_id + ".json"), result)
        print(json.dumps(dict(object_id=object_id, serialization_status=validation["status"], selected_byte_order=validation["selected_byte_order"])), flush=True)
    require(useful_bytes == 419904, "Sparse useful-byte total differs")
    comparisons = []
    for case_id in expected_cases:
        points = []
        for cy in AXIS:
            for cx in AXIS:
                first = geometry_index.get((case_id, "result", (cy, cx)))
                second = geometry_index.get((case_id, "surface_volume", (cy, cx)))
                points.append(dict(coarse_yx=[cy, cx], comparison=compare_geometry(first, second) if first is not None and second is not None else None))
        comparisons.append(dict(id=case_id, points=points,
                                interpretation="Numerical second-minus-first comparison of different map representations; no common physical coordinate frame is established"))
    # Root-requested fixed-site coverage cross-tab; same 81 centers, no new sampling.
    coverage = []
    for case_id in expected_cases:
        arrays = cases[case_id]["arrays"]
        masks = {}
        for name, expected_dtype in (("surface_mask", "bool"), ("primary_mask", "bool"), ("known_mask", "bool"), ("labels", "uint8")):
            pin = arrays[name]
            pins[pin["path"]] = pin
            masks[name] = np.load(io.BytesIO(checked_bytes(pin, 1024 * 1024)), allow_pickle=False)
            require(masks[name].shape == (384, 384) and masks[name].dtype == np.dtype(expected_dtype), "Fixed-site literal mask shape/dtype differs")
        for variant in ("result", "surface_volume"):
            table = {name: dict(center_count=0, mapped_center_count=0, zero_sentinel_center_count=0, unresolved_center_count=0)
                     for name in ("all", "surface", "outside_surface", "primary", "outside_primary", "known", "primary_positive", "primary_negative")}
            zero_sites = []
            for cy in AXIS:
                for cx in AXIS:
                    surface, primary, known, label = (bool(masks[name][cy, cx]) for name in ("surface_mask", "primary_mask", "known_mask", "labels"))
                    site = geometry_index.get((case_id, variant, (cy, cx)))
                    groups = dict(all=True, surface=surface, outside_surface=not surface,
                                  primary=primary, outside_primary=not primary, known=known,
                                  primary_positive=primary and label, primary_negative=primary and not label)
                    status_key = "unresolved_center_count" if site is None else ("zero_sentinel_center_count" if site["center"]["zero_sentinel"] else "mapped_center_count")
                    for group, included in groups.items():
                        if included:
                            table[group]["center_count"] += 1
                            table[group][status_key] += 1
                    if status_key == "zero_sentinel_center_count":
                        zero_sites.append(dict(coarse_yx=[cy, cx], surface=surface, primary=primary, known=known, label=int(label)))
            coverage.append(dict(id=case_id, variant=variant, support_cross_tab=table, zero_sentinel_sites=zero_sites))
    for pin in pins.values():
        verify(pin)
    result = dict(status="PASS", plan=plan_pin, objects=outputs, same_point_variant_comparisons=comparisons,
                  fixed_site_support_cross_tab=coverage,
                  object_count=6, cases_per_object=2, centers_per_case=81, sampled_records_per_object=1458,
                  useful_sample_bytes=useful_bytes, byte_orders_tested=["little", "big"],
                  conditional_object_count=sum(o["validation"]["status"] == "conditional" for o in outputs),
                  unresolved_object_count=sum(o["validation"]["status"] == "unresolved" for o in outputs),
                  all_inputs_unchanged_before_after=True, network_calls=0, model_calls=0, checkpoint_reads=0, gpu_seconds=0,
                  command=[sys.executable, *sys.argv], wall_seconds=time.perf_counter() - started,
                  cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  runtime=dict(python=sys.version, numpy=np.__version__, threads=4),
                  resource_limits=dict(wall_seconds=600, cpu_seconds=1200, threads=4, peak_rss_bytes=3 * 1024**3,
                                       address_space_bytes=12 * 1024**3, output_bytes=256 * 1024**2),
                  interpretation="Invariant-consistent serialization under fixed publication/renderer hypotheses; not authenticated historical endianness, native CT frame, optical registration, or anatomical normal direction")
    write_json(args.output_dir / "input-integrity.json", dict(status="PASS", inputs=list(pins.values())))
    write_json(args.output_dir / "ppm-diagnostics.json", result)
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 3 * 1024**3, "Peak RSS cap exceeded")
    require(sum(p.stat().st_size for p in args.output_dir.rglob("*") if p.is_file()) <= 256 * 1024**2, "New retained output cap exceeded")
    require(time.perf_counter() - started <= 600, "Wall cap exceeded")
    signal.alarm(0)
    print(json.dumps(dict(status="PASS", output=record(args.output_dir / "ppm-diagnostics.json"))), flush=True)


if __name__ == "__main__":
    main()
