#!/usr/bin/env python3
"""Analyze a fixed 384-square PHerc0800 screen without ink/text accuracy claims."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import find_objects, label

CODE_PATH = Path(__file__).resolve()
CODE_SHA256 = hashlib.sha256(CODE_PATH.read_bytes()).hexdigest()
ADAPTER_SHA256 = "2cbdb280777e19794fc78857969f9c1d43965da17772f6a1659264e6f5a48a59"
UPSTREAM_SHA256 = "2165b5def9d0dede136dde8202993be800b121cb2feb13cf98e98cd0629d75df"
VILLA_REVISION = "e0bbb8b40a2db58b1d71864f286eb85717e59e64"
MODEL_REVISION = "7109667e2607db1b90c37c8b09cb876ea7fe7bb1"
INFERENCE_PLAN_SHA256 = "f0bfd83b50246af1077d61ea38a23e4eb9ed307e6edf59ac121442198871f725"
CHECKPOINTS = {
    "seed42": (138360039, "e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab"),
    "seed43": (138360231, "2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f"),
}
CONDITIONS = {
    "center": (7, "forward", ["seed42", "seed43"]),
    "minus2": (5, "forward", ["seed42"]),
    "plus2": (9, "forward", ["seed42"]),
    "reverse": (7, "reverse", ["seed42"]),
}
SHAPE = (384, 384)
STARTS = [0, 64, 128, 192, 256]
THRESHOLD = 0.5
QUANTILES = [0, 1, 5, 25, 50, 75, 95, 99, 100]
NULL_READING = {"letter": None, "language": None, "transcription": None,
                "diplomatic_text": None, "normalized_text": None, "translation": None}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(root, name):
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root.resolve()), f"Retained file escapes its directory: {name}")
    return path


class Inputs:
    def __init__(self):
        self.records = {}

    def verify(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        if path not in self.records:
            self.records[path] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
        record = self.records[path]
        require(expected is None or record["sha256"] == expected, f"Input SHA-256 mismatch: {path}")
        return path

    def json(self, path, expected=None):
        return json.loads(self.verify(path, expected).read_text(encoding="utf-8"))

    def array(self, path, expected, shape, dtype):
        array = np.load(self.verify(path, expected), mmap_mode="r", allow_pickle=False)
        require(array.shape == tuple(shape) and array.dtype == np.dtype(dtype), f"Wrong array shape/dtype: {path}")
        return array

    def recheck(self):
        for path, record in self.records.items():
            require(path.stat().st_size == record["size_bytes"] and sha(path) == record["sha256"],
                    f"Input changed during analysis: {path}")

    def manifest(self):
        return [{"path": str(path), **record} for path, record in sorted(self.records.items())]


def validate_data(data_dir, inputs):
    provenance_path = inputs.verify(data_dir / "provenance.json")
    provenance = inputs.json(provenance_path)
    require(provenance["sample_id"] == "PHerc0800" and provenance["target_volume_id"] == "20250521135224",
            "Wrong target scan identity")
    require(provenance["shape_zyx"] == [31, 384, 384] and provenance["dtype"] == "uint8",
            "Expected a native 31-by-384-by-384 uint8 source")
    require(provenance["pyramid_level"] == 0 and provenance["voxel_spacing_zyx_um"] == [8.64] * 3,
            "Wrong source sampling")
    bbox = provenance["bbox_source_xywh"]
    require(len(bbox) == 4 and bbox[2:] == [384, 384] and min(bbox[:2]) >= 0
            and bbox[0] % 128 == 0 and bbox[1] % 128 == 0, "Wrong aligned source bbox")
    plan = inputs.json(provenance["frozen_plan_path"], provenance["frozen_plan_sha256"])
    inputs.verify(data_dir / "frozen-plan.json", provenance["frozen_plan_sha256"])
    require(plan["bbox_source_xywh"] == bbox and plan["segment_id"] == provenance["segment_id"]
            and plan["surface_url"].rstrip("/") == provenance["surface_url"], "Source and frozen selection differ")
    require(plan["verified_inputs"] == provenance["verified_inputs"], "Frozen geometry input records differ")
    verified = {}
    for record in provenance["verified_inputs"]:
        path = inputs.verify(record["path"], record["sha256"])
        verified[path] = record["sha256"]
    inputs.verify(provenance["acquisition_code_path"], provenance["acquisition_code_sha256"])
    specs = {"surface.npy": ([31, 384, 384], "uint8"), "geometry-mask.npy": (SHAPE, "bool"),
             "ct-coordinates.xyz.npy": ((*SHAPE, 3), "float32"), "observed-mask.npy": (SHAPE, "bool")}
    require(set(provenance["derived_outputs"]) == set(specs), "Unexpected acquisition outputs")
    arrays = {}
    for name, (shape, dtype) in specs.items():
        record = provenance["derived_outputs"][name]
        require(record["shape"] == list(shape) and record["dtype"] == dtype, "Derived output metadata differs")
        arrays[name] = inputs.array(data_dir / name, record["sha256"], shape, dtype)
    require(provenance["array_sha256"] == provenance["derived_outputs"]["surface.npy"]["sha256"],
            "Surface pin differs between provenance fields")
    surface, geometry = arrays["surface.npy"], arrays["geometry-mask.npy"]
    coordinates, observed = arrays["ct-coordinates.xyz.npy"], arrays["observed-mask.npy"]
    require(np.isfinite(coordinates[geometry]).all(), "Geometry-valid pixels lack finite CT coordinates")
    require(len(provenance["metadata_sources"]) == 2, "Expected retained Zarr metadata and attributes")
    metadata = {}
    for record in provenance["metadata_sources"]:
        require(record["http_status"] == 200 and record["local_payload_role"] == "observed source",
                "Metadata was not observed")
        path = inputs.verify(inside(data_dir, record["local_file"]), record["sha256"])
        require(path.stat().st_size == record["size_bytes"], "Metadata byte count differs")
        metadata[path.name] = inputs.json(path)
    zarray = metadata["source.zarray.json"]
    require(zarray["zarr_format"] == 2 and zarray["shape"][0] == 31
            and zarray["chunks"] == [31, 128, 128] and zarray["dtype"] == "|u1"
            and zarray["order"] == "C" and zarray.get("compressor") is None
            and not zarray.get("filters") and zarray["fill_value"] == 0
            and zarray.get("dimension_separator") == "/", "Wrong native source chunk representation")
    require(inputs.json(plan["array_metadata_path"]) == zarray
            and inputs.json(plan["attrs_path"]) == metadata["source.zattrs.json"], "Frozen metadata changed")
    for key in ("geometry_mask_path", "coordinate_map_path", "array_metadata_path", "attrs_path"):
        require(Path(plan[key]).resolve() in verified, f"Frozen input lacks a SHA pin: {key}")
    full_geometry = inputs.array(plan["geometry_mask_path"], verified[Path(plan["geometry_mask_path"]).resolve()], zarray["shape"][1:], "bool")
    full_coordinates = inputs.array(plan["coordinate_map_path"], verified[Path(plan["coordinate_map_path"]).resolve()], (*zarray["shape"][1:], 3), "float32")
    x, y, width, height = bbox
    require(x + width <= zarray["shape"][2] and y + height <= zarray["shape"][1], "Crop escapes native source")
    require(np.array_equal(geometry, full_geometry[y:y + height, x:x + width]), "Crop geometry differs from frozen map")
    require(np.array_equal(coordinates, full_coordinates[y:y + height, x:x + width], equal_nan=True),
            "Crop CT coordinates differ from frozen map")
    expected_indices = [[0, row, col] for row in range(y // 128, y // 128 + 3)
                        for col in range(x // 128, x // 128 + 3)]
    chunks = provenance["source_chunks"]
    require([record["chunk_index_zyx"] for record in chunks] == expected_indices, "Wrong source chunk schedule/order")
    expected_observed = np.zeros(SHAPE, dtype=bool)
    missing, downloaded, error_bytes = 0, 0, 0
    for record in chunks:
        status = record["http_status"]
        require(status in (200, 404), "An HTTP failure was interpreted as fill")
        path = inside(data_dir, record["local_file"])
        if status == 200:
            inputs.verify(path, record["sha256"])
            require(record["local_payload_role"] == "observed source"
                    and record["size_bytes"] == record["observed_payload_bytes"] == 31 * 128 * 128,
                    "Wrong observed chunk accounting")
            downloaded += record["observed_payload_bytes"]
        else:
            missing += 1
            error = inputs.verify(inside(data_dir, record["error_record_file"]), record["sha256"])
            body = error.read_bytes()
            require(len(body) == record["size_bytes"] <= 65536, "Missing-object diagnosis exceeds its bound")
            require(b"<!DOCTYPE" not in body.upper() and b"<!ENTITY" not in body.upper(), "Unsupported XML declaration")
            root = ET.fromstring(body)
            require(root.tag == "Error" and root.findtext("Code") == "NoSuchKey"
                    and record["source_absence"] == "Explicit S3 NoSuchKey"
                    and record["reconstruction"] == "Declared Zarr v2 fill_value=0"
                    and record["observed_payload_bytes"] == 0, "Generic missing/error response is not declared fill")
            inputs.verify(path, record["reconstructed_sha256"])
            error_bytes += len(body)
        require(path.stat().st_size == 31 * 128 * 128, "Wrong retained chunk length")
        raw = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(31, 128, 128)
        if status == 404:
            require(not raw.any(), "Reconstructed chunk is not declared zero fill")
        _, row, col = record["chunk_index_zyx"]
        dy, dx = row * 128 - y, col * 128 - x
        require(np.array_equal(surface[:, dy:dy + 128, dx:dx + 128], raw), "Assembled crop differs from retained raw chunk")
        expected_observed[dy:dy + 128, dx:dx + 128] = status == 200
    require(np.array_equal(observed, expected_observed), "Observed mask does not distinguish HTTP 200 from reconstructed fill")
    require(provenance["downloaded_chunk_payload_bytes"] == downloaded
            and provenance["missing_chunks_reconstructed_as_declared_fill"] == missing,
            "Acquisition byte/fill accounting differs")
    response_bytes = downloaded + error_bytes + sum(record["size_bytes"] for record in provenance["metadata_sources"])
    require(provenance["total_response_body_bytes"] == response_bytes
            and response_bytes <= provenance["maximum_response_body_bytes"] == 6 * 1024 * 1024
            and provenance["maximum_chunk_payload_bytes"] == plan["maximum_chunk_payload_bytes"] == 9 * 31 * 128 * 128,
            "Bounded acquisition totals differ")
    require(provenance["observed_source_pixels"] == int(observed.sum())
            and provenance["geometry_valid_pixels"] == int(geometry.sum()), "Source mask counts differ")
    return provenance, surface, geometry, coordinates, observed


def validate_execution(runs_dir, data_dir, provenance, inputs):
    execution = inputs.json(runs_dir / "execution.json")
    require(execution["completed"] is True and execution["paid_resources_created"] is False,
            "Bounded CPU execution is incomplete or has an unexpected paid-resource record")
    require(execution["inference_plan_sha256"] == INFERENCE_PLAN_SHA256, "Inference plan differs from the predeclared freeze")
    plan = inputs.json(execution["inference_plan_path"], INFERENCE_PLAN_SHA256)
    canonical_conditions = [{"name": name, "layer_start": start, "direction": direction, "seeds": seeds}
                            for name, (start, direction, seeds) in CONDITIONS.items()]
    require(plan["conditions"] == canonical_conditions and plan["device"] == "cpu"
            and plan["threads"] == 4 and plan["maximum_scheduled_tiles"] == 125
            and plan["automatic_retries"] == 0 and plan["per_condition_timeout_seconds"] == 120,
            "Frozen inference bounds/conditions differ")
    method = plan["analysis"]
    require(method["crop_margin_pixels"] == 64 and method["score_midpoint"] == THRESHOLD
            and method["score_display_limits"] == [.25, .75] and method["source_display_limits"] == [0, 255],
            "Frozen domain/threshold/display settings differ")
    for record in plan["verified_inputs"]:
        path = inputs.verify(record["path"], record["sha256"])
        require(path.stat().st_size == record["size_bytes"], "Frozen inference input size differs")
    drivers = [record for record in plan["verified_inputs"] if Path(record["path"]).name == "run_screen.py"]
    require(len(drivers) == 1 and drivers[0]["sha256"] == execution["execution_code_sha256"],
            "Execution driver differs from the frozen source")
    provenance_path = inputs.verify(data_dir / "provenance.json")
    require(Path(execution["source_provenance_path"]).resolve() == provenance_path
            and execution["source_provenance_sha256"] == inputs.records[provenance_path]["sha256"],
            "Execution did not use the retained acquisition provenance")
    require([entry["name"] for entry in execution["conditions"]] == list(CONDITIONS)
            and all(entry["returncode"] == 0 for entry in execution["conditions"]),
            "A frozen condition failed or is missing")
    frozen_time = datetime.fromisoformat(plan["frozen_at_utc"])
    acquisition_time = datetime.fromisoformat(provenance["created_at_utc"])
    started_time = datetime.fromisoformat(execution["created_at_utc"])
    completed_time = datetime.fromisoformat(execution["completed_at_utc"])
    require(frozen_time <= acquisition_time <= started_time <= completed_time,
            "Recorded inference freeze/acquisition/execution chronology differs")
    hashes = {}
    for entry in execution["conditions"]:
        condition = entry["name"]
        require(0 <= entry["wall_seconds"] <= 120, "Condition exceeded its recorded timeout bound")
        metadata_path = inputs.verify(runs_dir / condition / "metadata.json", entry["metadata_sha256"])
        metadata_time = datetime.fromisoformat(inputs.json(metadata_path)["created_at_utc"])
        condition_time = datetime.fromisoformat(entry["started_at_utc"])
        require(started_time <= condition_time <= metadata_time <= completed_time,
                "Recorded condition chronology differs")
        inputs.verify(runs_dir / f"{condition}.log")
        hashes[condition] = entry["metadata_sha256"]
    return execution, hashes


def validate_run(run_dir, condition, metadata_sha256, inputs, data_dir, provenance, surface, geometry):
    start, direction, seeds = CONDITIONS[condition]
    metadata = inputs.json(run_dir / "metadata.json", metadata_sha256)
    require(metadata["schema_version"] == 1, "Wrong inference metadata schema")
    source = metadata["source"]
    surface_path = inputs.verify(data_dir / "surface.npy", provenance["array_sha256"])
    provenance_path = inputs.verify(data_dir / "provenance.json")
    require(Path(source["path"]).resolve() == surface_path and source["sha256"] == provenance["array_sha256"]
            and Path(source["provenance_path"]).resolve() == provenance_path
            and source["provenance_sha256"] == inputs.records[provenance_path]["sha256"]
            and source["provenance"] == provenance, f"Run {condition} did not use the retained source/provenance")
    mask_record = source["validity_mask"]
    require(mask_record is not None and Path(mask_record["path"]).resolve() == (data_dir / "geometry-mask.npy").resolve(),
            "Inference did not use the frozen geometry mask")
    require(np.array_equal(inputs.array(mask_record["path"], mask_record["sha256"], SHAPE, "bool"), geometry),
            "Inference geometry differs")
    require(metadata["adapter"]["sha256"] == ADAPTER_SHA256, "Inference adapter differs from unchanged baseline")
    inputs.verify(metadata["adapter"]["path"], ADAPTER_SHA256)
    upstream = metadata["focused_upstream"]
    require(upstream["commit"] == VILLA_REVISION and upstream["source_modules_unmodified"] is True
            and upstream["source_manifest_sha256"] == UPSTREAM_SHA256, "Model factory source differs")
    source_manifest_path = inputs.verify(upstream["source_manifest"], UPSTREAM_SHA256)
    source_manifest = inputs.json(source_manifest_path)
    require(source_manifest["source_commit"] == VILLA_REVISION, "Upstream manifest revision differs")
    for record in source_manifest["files"]:
        inputs.verify(inside(source_manifest_path.parent, record["path"]), record["sha256"])
    inputs.verify(source_manifest_path.parent / "UPSTREAM-LICENSE.txt", source_manifest["license_sha256"])
    processing = metadata["processing"]
    indices = list(range(start, start + 17))[::-1 if direction == "reverse" else 1]
    expected_settings = {
        "device": "cpu", "dtype": "float32", "threads": 4, "interop_threads": 1,
        "deterministic_algorithms": True, "batch_size": 1, "patch_size_zyx": [17, 128, 128],
        "stride_xy": [64, 64], "tile_starts_y": STARTS, "tile_starts_x": STARTS,
        "layer_window_start_stop": [start, start + 17], "source_layer_indices": indices,
        "depth_direction": direction, "display_rescaling_applied": False,
        "normalization": "Unmodified upstream normalize_robust over each complete ZYX patch: percentile clip 1/99, median/MAD*1.4826 with upstream fallbacks",
        "blend": "Nonperiodic torch Hann outer product, normalized maximum, floored at 0.001",
    }
    for key, expected in expected_settings.items():
        require(processing.get(key) == expected, f"Run {condition} processing mismatch: {key}")
    require([model["label"] for model in metadata["models"]] == seeds, "Wrong checkpoint labels/order")
    expected_outputs = {"validity-mask"} | set(seeds) | {seed + "-evaluated-mask" for seed in seeds}
    if len(seeds) == 2:
        expected_outputs |= {"ensemble-mean", "seed-absolute-difference"}
    require(set(metadata["outputs"]) == expected_outputs, "Unexpected inference output set")
    outputs = {}
    for name, record in metadata["outputs"].items():
        dtype = "bool" if name.endswith("mask") else "float32"
        require(record["dtype"] == dtype and record["shape_yx"] == list(SHAPE), "Output metadata shape/dtype differs")
        outputs[name] = inputs.array(inside(run_dir, record["path"]), record["sha256"], SHAPE, dtype)
    require(np.array_equal(outputs["validity-mask"], geometry), "Output validity differs from source geometry")
    profile = np.any(surface[start:start + 17] != 0, axis=0)
    expected_bboxes = [[x, y, 128, 128] for y in STARTS for x in STARTS]
    records = {}
    for model in metadata["models"]:
        seed = model["label"]
        size, digest = CHECKPOINTS[seed]
        require(model["sha256"] == digest and model["model_revision"] == MODEL_REVISION
                and model["model_repository"] == "scrollprize/ink_9um", "Checkpoint identity differs")
        checkpoint = inputs.verify(model["path"], digest)
        require(checkpoint.stat().st_size == model["size_bytes"] == size and model["strict_state_match"] is True
                and not model["missing_keys"] and not model["unexpected_keys"], "Checkpoint was not loaded strictly")
        require(model["deserialization"] == "torch.load(map_location='cpu', weights_only=True)", "Unsafe checkpoint deserialization")
        config = model["embedded_config"]
        require(config["patch_size"] == [17, 128, 128] and config["loss"]["bce_label_smoothing"] == 0.5,
                "Unexpected checkpoint input/smoothed-target recipe")
        scores, evaluated = outputs[seed], outputs[seed + "-evaluated-mask"]
        require(np.isfinite(scores[geometry]).all() and np.isnan(scores[~geometry]).all()
                and ((scores[geometry] >= 0) & (scores[geometry] <= 1)).all(), "Wrong score validity/range")
        require([tile["bbox_roi_xywh"] for tile in model["tiles"]] == expected_bboxes, "Wrong scheduled tile records")
        support, statuses = np.zeros(SHAPE, dtype=bool), []
        for tile in model["tiles"]:
            x, y, width, height = tile["bbox_roi_xywh"]
            window = np.s_[y:y + height, x:x + width]
            expected_status = "scored" if geometry[window].any() and profile[window].any() else "skipped"
            require(tile["status"] == expected_status, "Tile status differs from unchanged source/geometry scheduling")
            statuses.append(expected_status)
            if expected_status == "scored":
                support[window] = True
        require(np.array_equal(evaluated, support & geometry), "Evaluated mask differs from contributing tiles")
        require(not scores[geometry & ~evaluated].any(), "Unevaluated valid pixels lost stock zero-output semantics")
        require(model["tile_count"] == 25 and model["scored_tile_count"] == statuses.count("scored")
                and model["skipped_tile_count"] == statuses.count("skipped")
                and model["evaluated_valid_pixel_count"] == int(evaluated.sum())
                and model["valid_pixel_count"] == int(geometry.sum()), "Inference tile/pixel accounting differs")
        records[seed] = {"scores": scores, "evaluated": evaluated, "profile": profile, "model": model}
    if len(seeds) == 2:
        require(np.array_equal(outputs["ensemble-mean"], (outputs["seed42"] + outputs["seed43"]) * np.float32(.5), equal_nan=True),
                "Retained ensemble differs from raw outputs")
        require(np.array_equal(outputs["seed-absolute-difference"], np.abs(outputs["seed42"] - outputs["seed43"]), equal_nan=True),
                "Retained difference differs from raw outputs")
    return metadata, records


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    require(np.isfinite(values).all(), "Nonfinite score entered the fixed analysis domain")
    if not values.size:
        return {"count": 0, "mean": None, "std": None, "quantiles_percent": QUANTILES, "quantiles": None}
    return {"count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "quantiles_percent": QUANTILES, "quantiles": np.percentile(values, QUANTILES).tolist()}


def compare(a, b, mask):
    left, right = a[mask].astype(np.float64), b[mask].astype(np.float64)
    above_a, above_b = mask & (a > THRESHOLD), mask & (b > THRESHOLD)
    intersection, union = int((above_a & above_b).sum()), int((above_a | above_b).sum())
    return {"pearson_correlation": float(np.corrcoef(left, right)[0, 1])
            if left.size > 1 and left.std() > 0 and right.std() > 0 else None,
            "signed_difference_b_minus_a": describe(right - left),
            "absolute_difference": describe(np.abs(right - left)),
            "a_strict_gt_0_5_pixels": int(above_a.sum()), "b_strict_gt_0_5_pixels": int(above_b.sum()),
            "both_strict_gt_0_5_pixels": intersection, "either_strict_gt_0_5_pixels": union,
            "threshold_flip_pixels": int((above_a ^ above_b).sum()),
            "threshold_jaccard": float(intersection / union) if union else None,
            "empty_union_policy": "Jaccard is null when neither output exceeds the threshold."}


def components(shared, scores, coordinates, bbox):
    labels, count = label(shared, structure=np.ones((3, 3), dtype=np.uint8))
    rows = []
    minimum = np.minimum(scores["center_seed42"], scores["center_seed43"])
    for component_id, slices in enumerate(find_objects(labels), 1):
        if slices is None:
            continue
        ys, xs = slices
        member = labels[slices] == component_id
        yy, xx = np.where(member)
        best = int(np.argmax(minimum[slices][member]))
        py, px = int(ys.start + yy[best]), int(xs.start + xx[best])
        xyz = coordinates[slices][member].astype(np.float64)
        local_bbox = [xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start]
        rows.append({"component_label": component_id, "area_pixels": int(member.sum()),
                     "bbox_roi_xywh": local_bbox,
                     "bbox_source_xywh": [bbox[0] + xs.start, bbox[1] + ys.start, local_bbox[2], local_bbox[3]],
                     "peak_roi_xy": [px, py], "peak_source_xy": [bbox[0] + px, bbox[1] + py],
                     "peak_ct_xyz_native_voxels": coordinates[py, px].astype(float).tolist(),
                     "ct_bbox_xyz_native_voxels": [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
                     "peak_raw_scores": {name: float(array[py, px]) for name, array in scores.items()},
                     "peak_minimum_center_seed_raw_score": float(minimum[py, px]),
                     "reading": dict(NULL_READING)})
    rows.sort(key=lambda row: (-row["area_pixels"], -row["peak_minimum_center_seed_raw_score"],
                               row["bbox_roi_xywh"][1], row["bbox_roi_xywh"][0]))
    for rank, row in enumerate(rows, 1):
        row["diagnostic_rank"] = rank
    return labels.astype(np.int32), {"connectivity": "8-neighbor", "count": int(count),
                                    "all_components": rows, "top_components": rows[:10],
                                    "total_area_pixels": int(shared.sum()),
                                    "interpretation": "Diagnostic detector components; no ink, glyph or reading is established."}


def analyze_domain(mask, scores, coordinates, bbox):
    center42, center43 = scores["center_seed42"], scores["center_seed43"]
    shared = mask & (center42 > THRESHOLD) & (center43 > THRESHOLD)
    component_labels, component_records = components(shared, scores, coordinates, bbox)
    center_comparison = compare(center42, center43, mask)
    sensitivity = {}
    for condition in ("minus2", "plus2", "reverse"):
        comparison = compare(center42, scores[condition + "_seed42"], mask)
        retained = int((shared & (scores[condition + "_seed42"] > THRESHOLD)).sum())
        comparison.update(center_shared_pixels_retained=retained,
                          fraction_of_center_shared_pixels_retained=float(retained / shared.sum()) if shared.any() else None)
        sensitivity[condition] = comparison
    return {"pixels": int(mask.sum()), "raw_score_distributions": {name: describe(array[mask]) for name, array in scores.items()},
            "seed_comparison": center_comparison, "seed_jaccard": center_comparison["threshold_jaccard"],
            "shared_strict_gt_0_5_pixels": int(shared.sum()),
            "shared_fraction_of_domain": float(shared.sum() / mask.sum()) if mask.any() else None,
            "condition_sensitivity_vs_center_seed42": sensitivity, "components": component_records}, shared, component_labels


def figures(output, surface, geometry, observed, scores, records, primary, strict, bbox):
    x, y, width, height = bbox
    extent = (x - .5, x + width - .5, y + height - .5, y - .5)
    score_cmap = plt.get_cmap("magma").copy()
    score_cmap.set_bad("#b7b7b7")
    source_cmap = plt.get_cmap("gray").copy()
    source_cmap.set_bad("#b7b7b7")
    visible = geometry & observed
    shared = primary & (scores["center_seed42"] > THRESHOLD) & (scores["center_seed43"] > THRESHOLD)
    support = np.zeros(SHAPE, dtype=np.uint8)
    support[primary] = 1
    support[strict] = 2
    views = [
        ("Observed center layer 15 · gray 0–255", np.ma.array(surface[15], mask=~visible), source_cmap, 0, 255),
        ("MAX projection, layers 7:24 · gray 0–255", np.ma.array(surface[7:24].max(axis=0), mask=~visible), source_cmap, 0, 255),
        ("Center seed42 · raw score 0.25–0.75", np.ma.array(scores["center_seed42"], mask=~(visible & records["center_seed42"]["evaluated"])), score_cmap, .25, .75),
        ("Center seed43 · raw score 0.25–0.75", np.ma.array(scores["center_seed43"], mask=~(visible & records["center_seed43"]["evaluated"])), score_cmap, .25, .75),
        ("Shared raw score >0.5 · primary domain", np.ma.array(shared.astype(np.uint8), mask=~primary), source_cmap, 0, 1),
        ("Fixed common support · primary / strict", support, ListedColormap(["#333333", "#e6ad43", "#237cab"]), -.5, 2.5),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for index, (ax, (title, array, cmap, low, high)) in enumerate(zip(axes.ravel(), views)):
        rendered = ax.imshow(array, cmap=cmap, vmin=low, vmax=high, extent=extent, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Native source X pixel")
        ax.set_ylabel("Native source Y pixel")
        if index < 4:
            fig.colorbar(rendered, ax=ax, fraction=.046, pad=.02)
    axes.ravel()[-1].legend(handles=[Patch(color="#333333", label="Excluded"), Patch(color="#e6ad43", label="Primary only"),
                                    Patch(color="#237cab", label="Strict subset")], loc="upper right", fontsize=8)
    fig.suptitle("384² PHerc0800 screen · detector diagnostics · no reading · source display fixed, no fitted contrast", fontsize=13)
    fig.savefig(output / "screen-overview.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    center = scores["center_seed42"]
    for column, condition in enumerate(CONDITIONS):
        name = condition + "_seed42"
        array = np.ma.array(scores[name], mask=~primary)
        image = axes[0, column].imshow(array, cmap=score_cmap, vmin=.25, vmax=.75, extent=extent, interpolation="nearest")
        axes[0, column].set_title(f"{condition} seed42 · raw 0.25–0.75")
        fig.colorbar(image, ax=axes[0, column], fraction=.046, pad=.02)
        difference = np.ma.array(np.abs(scores[name] - center), mask=~primary)
        image = axes[1, column].imshow(difference, cmap=score_cmap, vmin=0, vmax=.25, extent=extent, interpolation="nearest")
        axes[1, column].set_title("|Difference from center42| · 0–0.25")
        fig.colorbar(image, ax=axes[1, column], fraction=.046, pad=.02)
    for ax in axes.ravel():
        ax.set_xlabel("Native source X pixel")
        ax.set_ylabel("Native source Y pixel")
    fig.suptitle("Depth/orientation sensitivity on the same fixed primary domain · all displayed limits fixed", fontsize=13)
    fig.savefig(output / "condition-sensitivity.png", dpi=160)
    plt.close(fig)


def write_html(output, summary):
    domains = summary["domains"]
    rows = []
    for name, domain in domains.items():
        correlation = domain["seed_comparison"]["pearson_correlation"]
        jaccard = domain["seed_jaccard"]
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{domain['pixels']:,}</td>"
                    f"<td>{domain['shared_strict_gt_0_5_pixels']:,}</td><td>{domain['components']['count']}</td>"
                    f"<td>{'null' if correlation is None else f'{correlation:.4f}'}</td>"
                    f"<td>{'null' if jaccard is None else f'{jaccard:.4f}'}</td></tr>")
    source = summary["source"]
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in summary["limitations"])
    document = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PHerc0800 bounded screen</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#222}}img{{width:100%;height:auto}}table{{border-collapse:collapse}}th,td{{padding:.45rem .8rem;border:1px solid #bbb;text-align:left}}code{{overflow-wrap:anywhere}}figcaption{{color:#444}}a{{color:#1766a0}}</style>
<main><h1>PHerc0800 bounded detector screen</h1>
<p>384 × 384 native surface pixels, nominal 3.32 mm context. Segment <code>{html.escape(source['segment_id'])}</code>,
source bbox <code>{html.escape(str(source['bbox_source_xywh']))}</code>. Physical independence remains unresolved.
Both seeds share training data. No target labels, letters, transcription or translation are established.</p>
<table><thead><tr><th>Fixed domain</th><th>Pixels</th><th>Both center seeds &gt;0.5</th><th>Components</th><th>Seed Pearson</th><th>Seed Jaccard</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<figure><img src="screen-overview.png" alt="Fixed-scale source, center predictions and common support">
<figcaption>Full-field source images use fixed grayscale 0–255 without fitted contrast. The maximum projection combines layers 7 through 23;
it is not a single source plane. Seed panels use raw score limits 0.25–0.75; gray indicates excluded or unevaluated pixels.
Statistics and shared diagnostics use the frozen common domain.</figcaption></figure>
<figure><img src="condition-sensitivity.png" alt="Fixed-domain depth and orientation sensitivity">
<figcaption>All controls use seed42 on the same fixed primary domain. Raw score limits remain 0.25–0.75;
absolute-difference displays use fixed 0–0.25. No display transform is applied to saved raw arrays.</figcaption></figure>
<p>The primary mask combines the 64-pixel crop margin, geometry, observed HTTP 200 source, evaluation by all conditions,
and a nonzero profile in every selected 17-layer window. The strict subset removes the entire support of any scored tile
that includes invalid geometry, unobserved source or an all-zero profile under its condition. This is a conservative support rule.</p>
<ul>{limitations}</ul><p><a href="summary.json">Machine-readable metrics and diagnostic coordinates</a> ·
<a href="manifest.json">SHA-256 input/output manifest</a></p></main></html>'''
    (output / "review.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    data_dir, runs_dir = args.data_dir.resolve(strict=True), args.runs_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    require(not output.exists(), "Use a fresh output directory; previous results are preserved")
    require(not output.is_relative_to(data_dir) and not output.is_relative_to(runs_dir), "Output must be outside immutable input directories")
    inputs = Inputs()
    inputs.verify(CODE_PATH, CODE_SHA256)
    provenance, surface, geometry, coordinates, observed = validate_data(data_dir, inputs)
    execution, metadata_hashes = validate_execution(runs_dir, data_dir, provenance, inputs)
    records, run_metadata = {}, {}
    for condition in CONDITIONS:
        metadata, condition_records = validate_run(runs_dir / condition, condition, metadata_hashes[condition],
                                                   inputs, data_dir, provenance, surface, geometry)
        run_metadata[condition] = metadata
        records.update({condition + "_" + seed: record for seed, record in condition_records.items()})
    edge = np.zeros(SHAPE, dtype=bool)
    edge[64:-64, 64:-64] = True
    common_evaluated = np.logical_and.reduce([record["evaluated"] for record in records.values()])
    common_profile = np.logical_and.reduce([record["profile"] for record in records.values()])
    primary = edge & geometry & observed & common_evaluated & common_profile
    bad_support, bad_tiles = np.zeros(SHAPE, dtype=bool), []
    for name, record in records.items():
        for tile in record["model"]["tiles"]:
            if tile["status"] != "scored":
                continue
            x, y, width, height = tile["bbox_roi_xywh"]
            window = np.s_[y:y + height, x:x + width]
            counts = {"invalid_geometry_pixels": int((~geometry[window]).sum()),
                      "unobserved_source_pixels": int((~observed[window]).sum()),
                      "selected17_zero_profile_pixels": int((~record["profile"][window]).sum())}
            if any(counts.values()):
                bad_support[window] = True
                bad_tiles.append({"condition_seed": name, "bbox_roi_xywh": tile["bbox_roi_xywh"], **counts})
    strict = primary & ~bad_support
    scores = {name: record["scores"] for name, record in records.items()}
    results, saved = {}, {"crop-margin64-mask.npy": edge, "all-evaluated-mask.npy": common_evaluated,
                           "all-selected17-nonzero-mask.npy": common_profile, "primary-common-mask.npy": primary,
                           "strict-bad-tile-support-mask.npy": bad_support, "strict-common-mask.npy": strict}
    for name, mask in (("primary", primary), ("strict", strict)):
        result, shared, component_labels = analyze_domain(mask, scores, coordinates, provenance["bbox_source_xywh"])
        results[name] = result
        saved[name + "-shared-diagnostic-mask.npy"] = shared
        saved[name + "-component-labels.npy"] = component_labels
    summary = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_code_sha256": CODE_SHA256,
        "inference_plan_sha256": INFERENCE_PLAN_SHA256,
        "execution_manifest_sha256": inputs.records[(runs_dir / "execution.json").resolve()]["sha256"],
        "source": {key: provenance[key] for key in ("sample_id", "target_volume_id", "segment_id", "surface_url",
                   "bbox_source_xywh", "shape_zyx", "voxel_spacing_zyx_um", "array_sha256", "frozen_plan_sha256", "render_provenance")},
        "nominal_full_context_mm_xy": [3.31776, 3.31776], "physical_independence": "unresolved",
        "reading": dict(NULL_READING),
        "method": {"primary": "crop edge margin64 AND geometry AND observed HTTP200 source AND evaluated by all5 outputs AND nonzero among selected17 slices for every condition",
                   "strict": "primary minus entire128² support of each scored tile containing any invalid geometry, unobserved source or condition-specific all-zero17profile",
                   "global_geometry_distance_requirement": None,
                   "profile_support": "At least one nonzero voxel in the full selected17profile; no requirement that every layer is nonzero.",
                   "threshold_operator": ">", "threshold": THRESHOLD,
                   "score_semantics": "Unmodified raw sigmoid scores with BCE-smoothed targets0.25/0.75; not calibrated ink confidence.",
                   "score_display_limits": [.25, .75], "source_display_limits": [0, 255],
                   "source_projection": "Maximum of native layers7:24, not a single plane", "fitted_display_contrast": False},
        "conditions": {condition: {"layer_window_start_stop": [start, start + 17], "depth_direction": direction,
                       "seeds": seeds, "source_layer_indices": list(range(start, start + 17))[::-1 if direction == "reverse" else 1]}
                       for condition, (start, direction, seeds) in CONDITIONS.items()},
        "mask_counts": {"full_field": 384 * 384, "crop_margin64": int(edge.sum()), "geometry": int(geometry.sum()),
                        "observed_http200_source": int(observed.sum()), "all_evaluated": int(common_evaluated.sum()),
                        "all_condition_nonzero_profiles": int(common_profile.sum()), "primary": int(primary.sum()),
                        "strict": int(strict.sum()), "bad_tile_support_union": int(bad_support.sum()),
                        "primary_removed_by_strict": int((primary & bad_support).sum())},
        "strict_bad_scored_tile_records": bad_tiles, "domains": results,
        "limitations": ["No target ink labels were used; accuracy, precision, recall and authenticity are unestablished.",
                        "Two seeds share a training corpus and architecture; they are not independent evidence or extra samples.",
                        "Depth shifts and reversal measure model sensitivity, not ink-versus-substrate classification.",
                        "Physical independence from previously inspected papyrus remains unresolved; base coordinates do not establish complete sampled-volume separation.",
                        "The nominal3.32mm source canvas and smaller interior provide limited physical context.",
                        "Native8.64µm sampling differs from documented training sources; scores are descriptive.",
                        "Zero profiles and reconstructed fill describe missing/empty support, not ink absence.",
                        "The64pixel crop margin does not guarantee all contributing128tiles have valid support; the strict mask addresses this conservatively.",
                        "Every letter, transcription and translation field remains null; diagnostic components are not a reading."],
        "runtime_packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
    }
    inputs.recheck()
    output.mkdir(parents=True, exist_ok=False)
    for name, array in saved.items():
        np.save(output / name, array, allow_pickle=False)
    figures(output, surface, geometry, observed, scores, records, primary, strict, provenance["bbox_source_xywh"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_html(output, summary)
    inputs.recheck()
    output_records = [{"path": path.name, "sha256": sha(path), "size_bytes": path.stat().st_size}
                      for path in sorted(output.iterdir()) if path.is_file()]
    manifest = {"schema_version": 1, "status": "analysis completed and input/output hashes verified",
                "inputs": inputs.manifest(), "outputs": output_records,
                "manifest_self_hash": "Intentionally omitted; this manifest is not included in its own output list."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "mask_counts": summary["mask_counts"],
                      "shared_counts": {name: result["shared_strict_gt_0_5_pixels"] for name, result in results.items()},
                      "reading": dict(NULL_READING)}, indent=2))


if __name__ == "__main__":
    main()
