#!/usr/bin/env python3
"""Describe a frozen PHerc0800 model run; no ink accuracy or reading is inferred."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label

LAB = Path(__file__).resolve().parent
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
HF_REVISION = "7109667e2607db1b90c37c8b09cb876ea7fe7bb1"
VILLA_REVISION = "e0bbb8b40a2db58b1d71864f286eb85717e59e64"
FROZEN_HASHES = {
    "metadata": "60c331b1492ce1a275e36a49b6d9ae00e7b0224e8ee798885c5ff4f691e258dc",
    "source": "67b986675272364c0d677b3cf0741c42087d5950536b7f89adbc36b1c05f56b0",
    "provenance": "537f42b286d37bd98c889fa8196fdceee10b714f79a69e2f321513b5a87feb49",
    "acquisition_plan": "73b3ba464625dd1d38db24cf0977546c4c09c9144a0cab0e0f8e141c51553cf1",
    "adapter": "54170b298c716c4b8cb56da539bb74812a20cb8666888d02ddaf2984ddea3fbf",
    "upstream_manifest": "2165b5def9d0dede136dde8202993be800b121cb2feb13cf98e98cd0629d75df",
}
CHECKPOINTS = {
    "seed42": (138360039, "e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab"),
    "seed43": (138360231, "2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f"),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Inputs:
    """Check each retained input before reading and again after analysis."""
    def __init__(self):
        self.hashes = {}

    def verify(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha(path)
        require(expected is None or digest == expected, f"Checksum mismatch: {path}")
        require(path not in self.hashes or self.hashes[path] == digest, f"Input changed: {path}")
        self.hashes[path] = digest
        return path

    def json(self, path, expected=None):
        return json.loads(self.verify(path, expected).read_text(encoding="utf-8"))

    def array(self, path, expected, shape, dtype):
        array = np.load(self.verify(path, expected), mmap_mode="r", allow_pickle=False)
        require(array.shape == tuple(shape), f"Wrong array shape: {path}")
        require(array.dtype == np.dtype(dtype), f"Wrong array dtype: {path}")
        return array

    def recheck(self):
        for path, expected in self.hashes.items():
            require(sha(path) == expected, f"Input changed during analysis: {path}")


def describe(values):
    values = np.asarray(values)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None, "percentiles_1_5_50_95_99": None}
    require(np.isfinite(values).all(), "Nonfinite scores entered an analysis domain.")
    return {"count": int(values.size), "mean": float(values.mean(dtype=np.float64)),
            "median": float(np.median(values)), "std": float(values.std(dtype=np.float64)),
            "min": float(values.min()), "max": float(values.max()),
            "percentiles_1_5_50_95_99": np.percentile(values, [1, 5, 50, 95, 99]).tolist()}


def global_geometry_distance(full_mask, source_bbox):
    """Euclidean pixels to invalid full-source geometry, including its border."""
    x, y, width, height = source_bbox
    distance = distance_transform_edt(np.pad(full_mask, 1, constant_values=False))[1:-1, 1:-1]
    return distance[y:y + height, x:x + width].astype(np.float32)


def strict_bad_tile_support(shape, geometry, observed, models):
    support = np.zeros(shape, dtype=bool)
    records = []
    seen = set()
    for model in models:
        for tile in model["tiles"]:
            if tile["status"] != "scored":
                continue
            bbox = tuple(tile["bbox_roi_xywh"])
            if bbox in seen:
                continue
            seen.add(bbox)
            x, y, width, height = bbox
            g = geometry[y:y + height, x:x + width]
            o = observed[y:y + height, x:x + width]
            if not g.all() or not o.all():
                support[y:y + height, x:x + width] = True
                records.append({"bbox_roi_xywh": list(bbox),
                                "invalid_geometry_pixels": int((~g).sum()),
                                "selected_window_zero_profile_pixels": int((~o).sum())})
    return support, records


def component_summary(shared, minimum, seed42, seed43, coordinates, source_bbox):
    """All 8-connected diagnostic components, sorted by area with stable ties."""
    components, count = label(shared, structure=np.ones((3, 3), dtype=np.uint8))
    rows = []
    origin_x, origin_y = source_bbox[:2]
    for component_id, slices in enumerate(find_objects(components), start=1):
        if slices is None:
            continue
        ys, xs = slices
        local_component = components[slices] == component_id
        yy, xx = np.where(local_component)
        component_scores = minimum[slices][local_component]
        peak_index = int(np.argmax(component_scores))
        peak_y, peak_x = int(ys.start + yy[peak_index]), int(xs.start + xx[peak_index])
        bbox = [int(xs.start), int(ys.start), int(xs.stop - xs.start), int(ys.stop - ys.start)]
        xyz = coordinates[peak_y, peak_x]
        require(np.isfinite(xyz).all(), "Diagnostic peak has no finite native CT coordinates.")
        rows.append({"component_label": component_id, "area_pixels": int(local_component.sum()),
                     "bbox_roi_xywh": bbox,
                     "bbox_source_xywh": [bbox[0] + origin_x, bbox[1] + origin_y, bbox[2], bbox[3]],
                     "peak_roi_xy": [peak_x, peak_y], "peak_source_xy": [peak_x + origin_x, peak_y + origin_y],
                     "peak_base_ct_xyz_voxels": xyz.astype(float).tolist(),
                     "peak_seed42_raw_score": float(seed42[peak_y, peak_x]),
                     "peak_seed43_raw_score": float(seed43[peak_y, peak_x]),
                     "peak_minimum_seed_raw_score": float(minimum[peak_y, peak_x]),
                     "reading": {"status": "unread", "language": None, "diplomatic_text": None,
                                 "normalized_text": None, "translation": None}})
    rows.sort(key=lambda row: (-row["area_pixels"], -row["peak_minimum_seed_raw_score"],
                               row["bbox_roi_xywh"][1], row["bbox_roi_xywh"][0]))
    for rank, row in enumerate(rows, start=1):
        row["diagnostic_rank"] = rank
    return components.astype(np.int32), {"connectivity": "8-neighbor", "count": int(count),
                                          "all_components": rows, "top_components": rows[:20],
                                          "area_total_pixels": int(shared.sum()),
                                          "interpretation": "Threshold components are detector leads, not ink or letters."}


def validate_run(run_dir, inputs):
    metadata_path = inputs.verify(run_dir / "metadata.json", FROZEN_HASHES["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = metadata["source"]
    require(source["sha256"] == FROZEN_HASHES["source"] and source["provenance_sha256"] == FROZEN_HASHES["provenance"], "Source/provenance differ from independently checked frozen run pins.")
    provenance = inputs.json(source["provenance_path"], source["provenance_sha256"])
    require(provenance == source["provenance"], "Embedded and retained input provenance differ.")
    require(provenance["sample_id"] == "PHerc0800" and provenance["target_volume_id"] == "20250521135224", "Wrong target scan.")
    require(provenance["segment_id"] == "20251028222030" and provenance["bbox_source_xywh"] == [512, 512, 1536, 1536], "Wrong frozen crop.")
    require(provenance["shape_zyx"] == [31, 1536, 1536] and provenance["dtype"] == "uint8", "Wrong source representation.")
    require(provenance["voxel_spacing_zyx_um"] == [8.64, 8.64, 8.64] and provenance["pyramid_level"] == 0, "Wrong source sampling.")
    require(provenance["array_sha256"] == source["sha256"], "Source and provenance hashes differ.")
    source_path = inputs.verify(source["path"], source["sha256"])
    surface = inputs.array(source_path, source["sha256"], [31, 1536, 1536], "uint8")
    shape = surface.shape[1:]
    verified = {str(inputs.verify(record["path"], record["sha256"])): record["sha256"] for record in provenance["verified_inputs"]}
    require(provenance["frozen_plan_sha256"] == FROZEN_HASHES["acquisition_plan"], "Wrong frozen acquisition plan pin.")
    frozen = inputs.json(provenance["frozen_plan_path"], FROZEN_HASHES["acquisition_plan"])
    require(frozen["bbox_source_xywh"] == provenance["bbox_source_xywh"] and frozen["segment_id"] == provenance["segment_id"], "Frozen acquisition plan does not match input.")
    for name, record in provenance["derived_outputs"].items():
        inputs.array(source_path.parent / name, record["sha256"], record["shape"], record["dtype"])
    coordinates = inputs.array(source_path.parent / "ct-coordinates.xyz.npy", provenance["derived_outputs"]["ct-coordinates.xyz.npy"]["sha256"], [1536, 1536, 3], "float32")
    global_mask_path = Path(frozen["geometry_mask_path"]).resolve()
    global_map_path = Path(frozen["coordinate_map_path"]).resolve()
    require(str(global_mask_path) in verified and str(global_map_path) in verified, "Global geometry lacks frozen input hashes.")
    full_mask = inputs.array(global_mask_path, verified[str(global_mask_path)], [2580, 2580], "bool")
    full_map = inputs.array(global_map_path, verified[str(global_map_path)], [2580, 2580, 3], "float32")
    require(np.array_equal(coordinates, full_map[512:2048, 512:2048], equal_nan=True), "Crop CT coordinates differ from global geometry.")
    processing = metadata["processing"]
    settings = {"device": "cpu", "dtype": "float32", "threads": 4, "interop_threads": 1,
                "batch_size": 1, "deterministic_algorithms": True, "patch_size_zyx": [17, 128, 128],
                "stride_xy": [64, 64], "layer_window_start_stop": [7, 24],
                "source_layer_indices": list(range(7, 24)), "depth_direction": "forward", "display_rescaling_applied": False,
                "normalization": "Unmodified upstream normalize_robust over each complete ZYX patch: percentile clip 1/99, median/MAD*1.4826 with upstream fallbacks",
                "blend": "Nonperiodic torch Hann outer product, normalized maximum, floored at 0.001"}
    for key, expected in settings.items():
        require(processing.get(key) == expected, f"Frozen processing mismatch: {key}")
    starts = list(range(0, 1536 - 128 + 1, 64))
    require(processing["tile_starts_y"] == starts and processing["tile_starts_x"] == starts, "Wrong tile schedule.")
    require(metadata["adapter"]["sha256"] == FROZEN_HASHES["adapter"], "Wrong frozen inference adapter pin.")
    inputs.verify(metadata["adapter"]["path"], FROZEN_HASHES["adapter"])
    upstream = metadata["focused_upstream"]
    require(upstream["commit"] == VILLA_REVISION and upstream["source_modules_unmodified"], "Wrong focused model source.")
    require(upstream["source_manifest_sha256"] == FROZEN_HASHES["upstream_manifest"], "Wrong focused source-manifest pin.")
    manifest_path = inputs.verify(upstream["source_manifest"], FROZEN_HASHES["upstream_manifest"])
    manifest = inputs.json(manifest_path)
    require(manifest["source_commit"] == VILLA_REVISION, "Wrong source manifest revision.")
    for record in manifest["files"]:
        path = (manifest_path.parent / record["path"]).resolve()
        require(path.is_relative_to(manifest_path.parent), "Vendor file leaves manifest root.")
        inputs.verify(path, record["sha256"])
    inputs.verify(manifest_path.parent / "UPSTREAM-LICENSE.txt", manifest["license_sha256"])
    models = metadata["models"]
    require([model["label"] for model in models] == ["seed42", "seed43"], "Both pinned seeds are required in order.")
    outputs = {}
    require(set(metadata["outputs"]) == {"seed42", "seed43", "seed42-evaluated-mask", "seed43-evaluated-mask", "validity-mask", "ensemble-mean", "seed-absolute-difference"}, "Unexpected output set.")
    for name, record in metadata["outputs"].items():
        path = (run_dir / record["path"]).resolve()
        require(path.is_relative_to(run_dir.resolve()), "Run output leaves run directory.")
        outputs[name] = inputs.array(path, record["sha256"], shape, record["dtype"])
        require(record["shape_yx"] == list(shape), "Output metadata shape differs from source.")
        expected_dtype = "bool" if name.endswith("mask") else "float32"
        require(outputs[name].dtype == np.dtype(expected_dtype), "Output dtype differs from frozen score/mask contract.")
    geometry = outputs["validity-mask"]
    require(geometry.dtype == np.bool_ and np.array_equal(geometry, full_mask[512:2048, 512:2048]), "Output validity differs from global source geometry.")
    mask_record = source["validity_mask"]
    original_geometry = inputs.array(mask_record["path"], mask_record["sha256"], shape, "bool")
    require(np.array_equal(geometry, original_geometry), "Inference mask changed geometry.")
    observed = np.any(surface[7:24] != 0, axis=0)
    retained_observed = inputs.array(source_path.parent / "selected-depth-nonzero-mask.npy", provenance["derived_outputs"]["selected-depth-nonzero-mask.npy"]["sha256"], shape, "bool")
    require(np.array_equal(observed, retained_observed), "Retained depth-support mask differs from source.")
    for model in models:
        seed = model["label"]
        size, digest = CHECKPOINTS[seed]
        require(model["sha256"] == digest and model["model_revision"] == HF_REVISION, "Wrong checkpoint pin.")
        path = inputs.verify(model["path"], digest)
        require(path.stat().st_size == size and model["size_bytes"] == size, "Wrong checkpoint size.")
        require(model["strict_state_match"] is True and not model["missing_keys"] and not model["unexpected_keys"], "Checkpoint state was not strict.")
        require(model["deserialization"] == "torch.load(map_location='cpu', weights_only=True)", "Unexpected checkpoint deserialization.")
        config = model["embedded_config"]
        require(config["patch_size"] == [17, 128, 128] and config["loss"]["bce_label_smoothing"] == 0.5, "Unexpected embedded input/loss recipe.")
        scores = outputs[seed]
        evaluated = outputs[seed + "-evaluated-mask"]
        require(scores.dtype == np.float32 and evaluated.dtype == np.bool_, "Wrong score/mask dtype.")
        require(np.isfinite(scores[geometry]).all() and np.isnan(scores[~geometry]).all(), "Invalid finite/NaN score semantics.")
        require(((scores[geometry] >= 0) & (scores[geometry] <= 1)).all(), "Scores outside [0,1].")
        require(not np.any(evaluated & ~geometry), "Evaluated mask contains invalid geometry.")
        expected_bboxes = [[x, y, 128, 128] for y in starts for x in starts]
        require([tile["bbox_roi_xywh"] for tile in model["tiles"]] == expected_bboxes, "Wrong tile records/order.")
        evaluated_support = np.zeros(shape, dtype=bool)
        statuses = []
        for tile in model["tiles"]:
            x, y, width, height = tile["bbox_roi_xywh"]
            has_geometry = np.any(geometry[y:y + height, x:x + width])
            has_source = np.any(observed[y:y + height, x:x + width])
            expected_status = "scored" if has_geometry and has_source else "skipped"
            require(tile["status"] == expected_status, "Tile status disagrees with source/mask scheduling.")
            statuses.append(expected_status)
            if expected_status == "scored":
                evaluated_support[y:y + height, x:x + width] = True
        require(np.array_equal(evaluated, evaluated_support & geometry), "Evaluated mask disagrees with scored-tile support.")
        require(model["tile_count"] == len(expected_bboxes) and model["scored_tile_count"] == statuses.count("scored") and model["skipped_tile_count"] == statuses.count("skipped"), "Wrong tile accounting.")
    require(np.array_equal(outputs["ensemble-mean"], (outputs["seed42"] + outputs["seed43"]) * np.float32(0.5), equal_nan=True), "Retained mean differs from raw seeds.")
    require(np.array_equal(outputs["seed-absolute-difference"], np.abs(outputs["seed42"] - outputs["seed43"]), equal_nan=True), "Retained seed difference differs from raw seeds.")
    return metadata, provenance, surface, coordinates, full_mask, geometry, observed, outputs


def analyze_domain(mask, seed42, seed43, coordinates, source_bbox):
    a, b = seed42[mask], seed43[mask]
    minimum = np.minimum(seed42, seed43)
    shared = mask & (seed42 > 0.5) & (seed43 > 0.5)
    components, records = component_summary(shared, minimum, seed42, seed43, coordinates, source_bbox)
    correlation = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() > 0 and b.std() > 0 else None
    result = {"analysis_pixels": int(mask.sum()), "seed42_raw_scores": describe(a), "seed43_raw_scores": describe(b),
              "minimum_seed_raw_scores": describe(minimum[mask]), "absolute_seed_difference": describe(np.abs(a - b)),
              "seed_pearson_correlation": correlation, "shared_strict_gt_0_5_pixels": int(shared.sum()),
              "shared_strict_gt_0_5_fraction": float(shared.sum() / mask.sum()) if mask.any() else None,
              "threshold": {"operator": ">", "value": 0.5, "both_seeds_required": True,
                            "interpretation": "Smoothed-target midpoint detector response, not calibrated ink/authenticity probability."},
              "components": records}
    return result, shared, components


def save_array(output_dir, name, array, outputs):
    path = output_dir / name
    np.save(path, array, allow_pickle=False)
    outputs[name] = {"path": name, "sha256": sha(path), "shape": list(array.shape), "dtype": str(array.dtype)}


def contrast(layer, mask):
    values = layer[mask]
    require(values.size > 0, "No valid source data for display.")
    lower, upper = np.percentile(values, [1, 99]).astype(float)
    if upper <= lower:
        upper = lower + 1
    return float(lower), float(upper)


def write_figures(output_dir, surface, seed42, seed43, geometry, primary, strict, summary):
    bbox = summary["source"]["bbox_source_xywh"]
    x, y, width, height = bbox
    extent = (x - 0.5, x + width - 0.5, y + height - 0.5, y - 0.5)
    score_cmap = plt.get_cmap("magma").copy()
    score_cmap.set_bad("#d5d5d5")
    support = np.zeros(geometry.shape, dtype=np.uint8)
    support[geometry] = 1
    support[primary] = 2
    support[strict] = 3
    low, high = contrast(surface[15], geometry)
    source_view = np.ma.array(surface[15], mask=~geometry)
    views = [("Observed depth15 · contrast1–99%", source_view, "gray", low, high),
             ("Seed42 raw score · fixed0.25–0.75", np.ma.array(seed42, mask=~geometry), score_cmap, 0.25, 0.75),
             ("Seed43 raw score · fixed0.25–0.75", np.ma.array(seed43, mask=~geometry), score_cmap, 0.25, 0.75),
             ("Minimum raw score · fixed0.25–0.75", np.ma.array(np.minimum(seed42, seed43), mask=~geometry), score_cmap, 0.25, 0.75),
             ("Absolute seed difference · fixed0–0.25", np.ma.array(np.abs(seed42 - seed43), mask=~geometry), score_cmap, 0, 0.25),
             ("Analysis support · categorical", support, ListedColormap(["#343434", "#c6c6c6", "#e9ac46", "#287da3"]), -0.5, 3.5)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), squeeze=False)
    for ax, (title, values, cmap, vmin, vmax) in zip(axes.ravel(), views):
        shown = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Native surface X")
        ax.set_ylabel("Native surface Y")
        if title.startswith("Analysis support"):
            ax.legend(handles=[Patch(color=color, label=text) for color, text in
                               [("#343434", "Invalid geometry"), ("#c6c6c6", "Valid outside primary"),
                                ("#e9ac46", "Primary only"), ("#287da3", "Strict secondary")]], loc="lower left", fontsize=8)
        else:
            fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.03)
        ax.add_patch(Rectangle((x + 63.5, y + 63.5), width - 128, height - 128, fill=False, color="cyan", linewidth=0.8))
    for row in summary["domains"]["primary"]["components"]["top_components"][:5]:
        xx, yy, ww, hh = row["bbox_source_xywh"]
        for ax in (axes[0, 0], axes[1, 0]):
            ax.add_patch(Rectangle((xx - 0.5, yy - 0.5), ww, hh, fill=False, color="#76ed90", linewidth=0.8))
            ax.text(xx, yy, str(row["diagnostic_rank"]), color="#76ed90", fontsize=8)
    fig.suptitle("PHerc0800 larger context · raw model behavior, no authenticated ink or reading\n"
                 "Cyan =64px crop interior; green =top primary diagnostic components; geometric/source exclusions remain separate", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "large-field-overview.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    summary["visuals"] = {path.name: {"sha256": sha(path), "source_layer": 15,
                                      "source_display_limits": [low, high], "source_display_percentiles": [1, 99],
                                      "percentile_population": "All geometry-valid pixels in the source crop",
                                      "score_display_limits": [0.25, 0.75], "seed_difference_display_limits": [0, 0.25]}}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
    layers = []
    for ax, layer_index in zip(axes.ravel(), [11, 15, 19]):
        lo, hi = contrast(surface[layer_index], geometry)
        shown = ax.imshow(np.ma.array(surface[layer_index], mask=~geometry), cmap="gray", vmin=lo, vmax=hi,
                          extent=extent, interpolation="nearest")
        ax.set_title(f"Observed layer{layer_index} · offset{layer_index - 15:+d}\nDisplay contrast1–99%; raw uint8 retained")
        ax.set_xlabel("Native surface X")
        ax.set_ylabel("Native surface Y")
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.03)
        ax.add_patch(Rectangle((x + 63.5, y + 63.5), width - 128, height - 128, fill=False, color="cyan", linewidth=0.8))
        layers.append({"source_layer": layer_index, "normal_offset_native_voxels": layer_index - 15,
                       "display_percentiles": [1, 99], "display_limits": [lo, hi]})
    fig.suptitle("PHerc0800 observed native layers · contrast is a recorded display adjustment", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = output_dir / "observed-layers-11-15-19.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    summary["visuals"][path.name] = {"sha256": sha(path), "layers": layers,
                                   "percentile_population": "All geometry-valid pixels in each source crop layer"}


def run(run_dir, output_dir):
    run_dir = run_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), "Choose a fresh analysis output directory; existing files are preserved.")
    inputs = Inputs()
    metadata, provenance, surface, coordinates, full_mask, geometry, observed, arrays = validate_run(run_dir, inputs)
    distance = global_geometry_distance(full_mask, provenance["bbox_source_xywh"])
    edge = np.zeros(geometry.shape, dtype=bool)
    edge[64:-64, 64:-64] = True
    common_evaluated = arrays["seed42-evaluated-mask"] & arrays["seed43-evaluated-mask"]
    global_margin = distance >= 64
    primary = geometry & common_evaluated & observed & edge & global_margin
    require(primary.any(), "No pixels satisfy the frozen primary domain.")
    bad_support, bad_tiles = strict_bad_tile_support(geometry.shape, geometry, observed, metadata["models"])
    strict = primary & ~bad_support
    domains, stored = {}, {}
    for name, mask in [("primary", primary), ("strict_secondary", strict)]:
        record, shared, components = analyze_domain(mask, arrays["seed42"], arrays["seed43"], coordinates, provenance["bbox_source_xywh"])
        domains[name] = record
        stored[name + "-analysis-mask.npy"] = mask
        stored[name + "-shared-gt-0_5-mask.npy"] = shared
        stored[name + "-shared-component-labels.npy"] = components
    counts = {"canvas_pixels": int(geometry.size), "geometry_valid_pixels": int(geometry.sum()),
              "selected17_nonzero_pixels": int(observed.sum()), "both_seeds_evaluated_pixels": int(common_evaluated.sum()),
              "crop64_interior_pixels": int(edge.sum()), "global_geometry_distance_ge64_pixels": int(global_margin.sum()),
              "geometry_and_evaluated_pixels": int((geometry & common_evaluated).sum()),
              "geometry_evaluated_selected17_pixels": int((geometry & common_evaluated & observed).sum()),
              "geometry_evaluated_selected17_crop64_pixels": int((geometry & common_evaluated & observed & edge).sum()),
              "primary_pixels": int(primary.sum()), "strict_secondary_pixels": int(strict.sum()),
              "bad_scored_tiles_unique": len(bad_tiles), "bad_scored_tile_support_pixels": int(bad_support.sum()),
              "primary_removed_by_strict_pixels": int((primary & bad_support).sum())}
    summary = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "analysis_code": {"path": str(Path(__file__).resolve()), "sha256": CODE_SHA256},
               "run_metadata": {"path": str(run_dir / "metadata.json"), "sha256": inputs.hashes[run_dir / "metadata.json"]},
               "source": {"sample_id": provenance["sample_id"], "target_volume_id": provenance["target_volume_id"],
                          "segment_id": provenance["segment_id"], "bbox_source_xywh": provenance["bbox_source_xywh"],
                          "shape_zyx": list(surface.shape), "voxel_spacing_zyx_um": provenance["voxel_spacing_zyx_um"],
                          "source_path": str(Path(metadata["source"]["path"])), "source_sha256": metadata["source"]["sha256"],
                          "provenance_path": metadata["source"]["provenance_path"], "provenance_sha256": metadata["source"]["provenance_sha256"],
                          "coordinate_convention": "Source X/Y are native rendered canvas indices; peak XYZ is base-surface native CT voxel XYZ, without depth normal offsets."},
               "methods": {"primary": "Geometry valid AND both evaluated AND any nonzero selected7:24 raw intensity AND local64px crop margin AND global geometry Euclidean distance>=64px",
                           "global_geometry_distance": "scipy Euclidean EDT on the full2580×2580 source geometry mask, padded by1false pixel, then cropped at absolute512:2048; units native surface canvas pixels",
                           "strict_secondary": "Primary minus entire128square support of any scored tile containing invalid geometry or any selected17-layer allzero profile; union across both seeds",
                           "strict_interpretation": "Conservative context support diagnostic; exclusions do not mean absence of papyrus or ink.",
                           "source_window": [7, 24], "depth_direction": "forward", "score_arrays": "Unquantized float32 retained model outputs; display limits do not transform analysis values",
                           "component_connectivity": "8-neighbor", "component_peak_rule": "Maximum minimum(seed42,seed43); first row-major pixel resolves ties",
                           "component_ranking": "Area descending, peak minimum score descending, bboxY thenX ascending; no minimum-area filter"},
               "fixed_mask_counts": counts, "bad_scored_tiles": bad_tiles, "domains": domains, "outputs": {},
               "verified_inputs": [{"path": str(path), "sha256": digest} for path, digest in inputs.hashes.items()],
               "reading": {"status": "unread", "language": None, "diplomatic_text": None, "normalized_text": None, "translation": None},
               "limitations": ["No trusted PHerc0800 ink labels or vetted reading; accuracy, precision/recall and new text are not established.",
                               "Two seeds share a training corpus; pixels, components and depth variants are dependent observations.",
                               "Raw>0.5 is a smoothed-target midpoint detector response, not an authenticity probability.",
                               "Larger canvas remains within the same scroll/CT neighborhood; segment identity does not prove independent papyrus.",
                               "64px margin does not guarantee complete support for every contributing128tile; secondary results expose that sensitivity.",
                               "Allzero selected profiles are a source-support diagnostic, not an ink-negative class.",
                               "Component selection is post hoc to fixed-threshold outputs; ranks and peaks are diagnostic leads only.",
                               "Source contrast and displayed score/difference clipping affect images only; stored raw arrays remain unchanged."]}
    inputs.recheck()
    output_dir.mkdir(parents=True, exist_ok=False)
    stored.update({"global-geometry-distance-pixels.npy": distance, "geometry-valid-mask.npy": np.asarray(geometry),
                   "selected17-nonzero-mask.npy": observed, "both-evaluated-mask.npy": common_evaluated,
                   "crop64-interior-mask.npy": edge, "global-geometry-distance-ge64-mask.npy": global_margin,
                   "bad-scored-tile-support-mask.npy": bad_support})
    for name, array in stored.items():
        save_array(output_dir, name, array, summary["outputs"])
    write_figures(output_dir, surface, arrays["seed42"], arrays["seed43"], geometry, primary, strict, summary)
    inputs.recheck()
    require(sha(Path(__file__)) == CODE_SHA256, "Analysis code changed during execution.")
    summary["source_integrity_before_and_after"] = "All recorded inputs and analysis source remained hash-identical."
    summary_path = output_dir / "large-analysis-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "fixed_mask_counts": counts,
                      "domains": {name: {key: value[key] for key in ["analysis_pixels", "shared_strict_gt_0_5_pixels", "shared_strict_gt_0_5_fraction", "seed_pearson_correlation"]} for name, value in domains.items()},
                      "visuals": list(summary["visuals"])}, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
