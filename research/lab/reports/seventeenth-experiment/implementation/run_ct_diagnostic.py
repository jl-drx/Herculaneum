"""CPU-only phase-17 producer, gated by exact frozen protocol and local input pins."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/seventeenth-matplotlib")
import sys
sys.dont_write_bytecode = True
import argparse
import gc
import io
import json
from pathlib import Path
import resource
import signal
import socket
import time


def offline(*args, **kwargs):
    raise RuntimeError("Network disabled for retained regional diagnostic")


socket.create_connection = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline

import numpy as np
from regional_common import checked_bytes, checked_json, coordinate_roundtrip, pool_triplet, record, require, verify, write_json
from regional_extract import RetainedCrops, CASE_IDS
from regional_stats import plane_stats, support_geometry


PROTOCOL_SHA = "08bc48f203d6722216c70d54b961d0d29ce59204359da49d2b6110661a9a9cc1"
DESCRIPTOR_SHA = "fef036010df97b465830da24a69a6ee583f61d0ac7831898f30b4e6e351d6e07"
RAW_SHA = "c9cc64e4c39ee063bbc67bcddf547fabd51c71af36143a1527c4130867e0185c"
CLARIFICATIONS_SHA = "22ed3e0749d56bd61113b6a2cd74f6ae7373082a34a1c090352ccba99eb2c643"
PPM_AMENDMENT_SHA = "ed3096856274505990fb9b4069ce6fdbb657eaeb93ae52b6bfae2144de6d9f9e"


def load_array(pin, shape, dtype, cap=16 * 1024 * 1024):
    arr = np.load(io.BytesIO(checked_bytes(pin, cap)), allow_pickle=False)
    require(arr.shape == shape and arr.dtype == np.dtype(dtype), "Literal array shape/dtype differs")
    require(np.isfinite(arr).all(), "Nonfinite literal array")
    return arr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1200))
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2, 256 * 1024**2))
    signal.alarm(600)
    plan_pin = record(args.plan)
    plan = checked_json(plan_pin)
    require(plan["kind"] == "phase17-frozen-ct-diagnostic", "Unexpected plan")
    require(plan["protocol"]["sha256"] == PROTOCOL_SHA and plan["descriptor"]["sha256"] == DESCRIPTOR_SHA
            and plan["raw_provenance"]["sha256"] == RAW_SHA, "Scientific protocol/input pins differ")
    verify(plan["protocol"])
    require(plan["clarifications"]["sha256"] == CLARIFICATIONS_SHA, "Clarifications pin differs")
    verify(plan["clarifications"])
    require(plan["ppm_amendment"]["sha256"] == PPM_AMENDMENT_SHA, "PPM amendment pin differs")
    verify(plan["ppm_amendment"])
    for pin in plan["code"]:
        verify(pin)
    descriptor = checked_json(plan["descriptor"])
    benchmark = checked_json(plan["benchmark"])
    benchmarks = {case["case_id"]: case for case in benchmark["selected_contexts"]}
    require([case["id"] for case in descriptor["cases"]] == CASE_IDS, "Frozen cases differ")
    pins = {pin["path"]: pin for pin in (plan_pin, plan["protocol"], plan["clarifications"], plan["ppm_amendment"], plan["descriptor"], plan["raw_provenance"], plan["benchmark"], *plan["code"])}
    # Verify every literal scientific array before the first numeric decode.
    for case in descriptor["cases"]:
        case_id = case["id"]
        require(case_id in plan["scores"], "Baseline score pin missing")
        reference = benchmarks[case_id]
        require(all(case[key] == reference[key] for key in ("fragment", "role", "parent_papyrus", "bbox_source_xywh", "native_bbox_yxhw")),
                "Benchmark/CT coordinate identity differs")
        for name, pin in case["arrays"].items():
            verify(pin)
            pins[pin["path"]] = pin
        for pin in plan["scores"][case_id].values():
            verify(pin)
            pins[pin["path"]] = pin
    extraction = RetainedCrops(plan["raw_provenance"], descriptor["cases"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from regional_plot import make_ct_panel
    results = []
    display_planes = {8, 20, 32, 44, 56}
    for case in descriptor["cases"]:
        case_id = case["id"]
        arrays = case["arrays"]
        labels = load_array(arrays["labels"], (384, 384), "uint8")
        primary = load_array(arrays["primary_mask"], (384, 384), "bool")
        surface = load_array(arrays["surface_mask"], (384, 384), "bool")
        known = load_array(arrays["known_mask"], (384, 384), "bool")
        pooled = load_array(arrays["surface"], (17, 384, 384), "float32")
        scores = load_array(plan["scores"][case_id]["scores"], (384, 384), "float32")
        require(np.isin(labels, [0, 1]).all() and np.all(primary <= known) and np.all(primary <= surface), "Literal label/support relationship differs")
        require(np.all((pooled >= 0) & (pooled <= 255)) and np.all((scores >= 0) & (scores <= 1)), "Published CT/score range differs")
        reference = benchmarks[case_id]
        require(int(primary.sum()) == reference["primary_count"] and int(surface.sum()) == reference["coarse_surface_count"]
                and int(known.sum()) == reference["coarse_known_count"], "Released support counts differ")
        coarse_masks = dict(all=np.ones((384, 384), dtype=bool), surface=surface,
                            primary_positive=primary & (labels == 1), primary_negative=primary & (labels == 0))
        require(int(coarse_masks["primary_positive"].sum()) == reference["primary_positive"]
                and int(coarse_masks["primary_negative"].sum()) == reference["primary_negative"], "Released class counts differ")
        native_masks = {name: np.repeat(np.repeat(mask, 3, axis=0), 3, axis=1) for name, mask in coarse_masks.items()}
        native_geometry, coarse_geometry = support_geometry(native_masks), support_geometry(coarse_masks)
        native_profiles, pooled_profiles, native_display = [], [], {}
        cross_row = np.empty((51, 1152), dtype=np.uint16)
        cross_col = np.empty((51, 1152), dtype=np.uint16)
        triplet = np.empty((3, 1152, 1152), dtype=np.uint16)
        checked_voxels = 0
        for plane_index, plane in extraction.planes(case_id):
            z = plane_index - 7
            native_profiles.append(dict(plane_index=plane_index, nominal_depth_um=3.24 * (plane_index - 32),
                                        supports=plane_stats(plane, native_geometry, 65535)))
            if plane_index in display_planes:
                native_display[plane_index] = plane.copy()
            cross_row[z], cross_col[z] = plane[577, :], plane[:, 577]
            triplet[z % 3] = plane
            if z % 3 == 2:
                j = z // 3
                reproduced = pool_triplet(triplet)
                require(np.array_equal(reproduced.view(np.uint32), pooled[j].view(np.uint32)), f"Exact pooling mismatch: {case_id}/{j}")
                checked_voxels += int(reproduced.size)
                center = 8 + 3 * j
                pooled_profiles.append(dict(pooled_index=j, plane_index=center, nominal_depth_um=3.24 * (center - 32),
                                            supports=plane_stats(pooled[j], coarse_geometry, 255)))
        geometry_case = dict(case, source_pixel_center_mapping=reference["source_pixel_center_mapping"])
        row = dict(id=case_id, fragment=case["fragment"], role=case["role"], parent_papyrus=case["parent_papyrus"],
                   bbox_source_xywh=case["bbox_source_xywh"], coordinate_roundtrip=coordinate_roundtrip(geometry_case),
                   support_counts={name: dict(coarse=int(mask.sum()), expanded_native=int(native_masks[name].sum())) for name, mask in coarse_masks.items()},
                   native_profiles=native_profiles, pooled_profiles=pooled_profiles,
                   exact_repooling=dict(status="PASS", float32_voxels_checked=checked_voxels, bitwise_equal=True),
                   saved_baseline_scores=plan["scores"][case_id]["scores"],
                   cross_sections=dict(native_local_row=577, native_local_column=577, coarse_center=192, nominal_spacing_um=3.24))
        panel = args.output_dir / f"{case_id}-ct-panel.png"
        make_ct_panel(panel, case, native_display, pooled,
                      dict(labels=labels, primary_mask=primary, surface_mask=surface), scores,
                      cross_row, cross_col, dict(native=native_profiles, pooled=pooled_profiles))
        row["panel"] = record(panel)
        results.append(row)
        write_json(args.output_dir / f"{case_id}-profiles.json", row)
        print(json.dumps(dict(case_id=case_id, status="PASS", exact_voxels=checked_voxels, elapsed_seconds=time.perf_counter() - started)), flush=True)
        del native_display, native_geometry, native_masks, coarse_geometry, triplet, plane, pooled, scores
        gc.collect()
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 3 * 1024**3, "RSS limit exceeded")
    summary = extraction.summary()
    require(summary["batch_count"] == 1530 and summary["useful_sample_bytes"] == 812187648, "Final complete extraction count differs")
    pins.update(extraction.input_pins)
    for pin in pins.values():
        verify(pin)
    output = dict(status="PASS", plan=plan_pin, protocol=plan["protocol"], cases=results, extraction=summary,
                  all_inputs_unchanged_before_after=True, model_calls=0, checkpoint_reads=0, network_calls=0,
                  gpu_seconds=0, wall_seconds=time.perf_counter() - started,
                  cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                  command=[sys.executable, *sys.argv],
                  resource_limits=dict(wall_seconds=600, cpu_seconds=1200, threads=4, peak_rss_bytes=3 * 1024**3,
                                       address_space_bytes=12 * 1024**3, output_bytes=256 * 1024**2),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  runtime=dict(python=sys.version, numpy=np.__version__, threads=4),
                  statistical_conventions=dict(quantile_method="linear", std_ddof=0, empty_statistics=None,
                      roughness="mean absolute adjacent difference in float64; both endpoints within support",
                      native_support="each literal coarse pixel repeated3x3; not a new native segmentation",
                      endpoints=dict(native=[0, 65535], pooled=[0, 255])),
                  interpretation="Descriptive supplied-coordinate checks, not measured optical registration or CT ink identification")
    write_json(args.output_dir / "input-integrity.json", dict(status="PASS", inputs=list(pins.values())))
    write_json(args.output_dir / "ct-diagnostics.json", output)
    require(sum(p.stat().st_size for p in args.output_dir.rglob("*") if p.is_file()) <= 256 * 1024**2, "Retained output cap exceeded")
    require(time.perf_counter() - started <= 600, "Wall limit exceeded")
    signal.alarm(0)
    print(json.dumps(dict(status="PASS", output=record(args.output_dir / "ct-diagnostics.json"))), flush=True)


if __name__ == "__main__":
    main()
