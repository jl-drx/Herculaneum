"""Assemble the source-contract evidence from metadata and text only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
LAB = Path("/workspace/herculaneum-lab")
OLD = LAB / "reports/tenth-experiment/reference-audit"
records = {}


def record(path):
    path = Path(path)
    data = path.read_bytes()
    rec = {"path": str(path), "byte_length": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    records[str(path)] = rec
    return rec


def read(path):
    record(path)
    return json.loads(Path(path).read_text())


fragment = read(OLD / "retained-fragment-provenance.json")["samples"]["PHerc0009B"]
surface_attrs = read(OLD / "PHerc0009B-surface_attributes.json")
surface_array = read(OLD / "PHerc0009B-surface_level2_array.json")
ink_array = read(OLD / "PHerc0009B-labels_level2_metadata.json")
source_commit = "530584554c031c982ba3aa243e1ea39c4bef9818"
source_dir = LAB / "data/experiments/pherc0800-native-ct-consistency/export-source"
for name in ["vc_render_tifxyz.cpp", "QuadSurface.cpp", "Geometry.cpp", "Slicing.cpp", "LICENSE"]:
    record(source_dir / name)
for name in ["archived-renderer-zarr-hpp.txt", "archived-renderer-zarr-cpp.txt"]:
    record(ROOT / name)
assert surface_attrs["num_slices"] == 109 and surface_attrs["slice_step"] == 1.0
assert surface_array["shape"] == [109, 6340, 5130]
assert surface_array["chunks"] == [109, 128, 128] and surface_array["compressor"] is None

ledger = read(ROOT / "request-ledger.json")
contract = read(ROOT / "lookup-contract.json")
assert len(ledger["requests"]) <= contract["maximum_https_attempts"]
assert ledger["response_body_bytes"] == sum(x["body_bytes"] for x in ledger["requests"])
assert ledger["response_body_bytes"] <= contract["aggregate_response_body_cap_bytes"]
for entry in ledger["requests"]:
    assert entry["body_bytes"] <= contract["per_response_body_cap_bytes"]
    if "local_path" in entry:
        rec = record(entry["local_path"])
        assert rec["byte_length"] == entry["body_bytes"] and rec["sha256"] == entry["sha256"]

ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
xml = ET.fromstring((ROOT / "pherc0009b-label-package-inventory.xml").read_bytes())
assert xml.findtext("s:IsTruncated", namespaces=ns) == "false"
inventory = [{"key": c.findtext("s:Key", namespaces=ns),
              "byte_length": int(c.findtext("s:Size", namespaces=ns)),
              "etag": c.findtext("s:ETag", namespaces=ns)} for c in xml.findall("s:Contents", ns)]
assert len(inventory) == 39
prefix = contract["candidate_label_prefix"].split("amazonaws.com/", 1)[1]
groups_found = sorted({a["key"][len(prefix):].split("/", 1)[0] for a in inventory})
assert groups_found == ["inklabels.zarr", "supervision.zarr", "validation.zarr"]
group_metadata = {}
array_metadata = {"inklabels": ink_array}
for role in ["inklabels", "supervision", "validation"]:
    group = read(ROOT / ("pherc0009b-" + role + "-group.json"))
    attrs = group["attributes"]
    assert attrs["canvas_size"] == [25360, 20520]  # YX, unlike source canvas XY.
    assert attrs["ink_republish"]["canvas_offset_yx"] == [-38.351, -44.944]
    assert attrs["ink_republish"]["atlas_segment"] == fragment["segment_long_id"]
    assert attrs["ink_republish"]["package_commit"] == "bfef6abe073510ee8b595da390ed1d639316abc8-dirty"
    transforms = next(x["coordinateTransformations"] for x in attrs["ome"]["multiscales"][0]["datasets"] if x["path"] == "2")
    assert transforms == [{"type": "scale", "scale": [9.604, 9.604]}]
    group_metadata[role] = attrs
    if role != "inklabels":
        array_metadata[role] = read(ROOT / ("pherc0009b-" + role + "-level2.json"))
    assert array_metadata[role]["shape"] == [6340, 5130]
    assert array_metadata[role]["zarr_format"] == 3 and array_metadata[role]["data_type"] == "uint8"

head = read(ROOT / "villa-merge-branch-head.json")["object"]["sha"]
directory = read(ROOT / "villa-merge-label-transfer-directory.json")
blob_receipts = []
for name in ["README.md", "core.py", "make_label_zarrs.py", "transfer.py", "io.py"]:
    path = ROOT / ("transfer-source-" + name.replace(".", "-") + ".txt")
    rec = record(path)
    data = path.read_bytes()
    blob_sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    expected = next(x["sha"] for x in directory if x["name"] == name)
    assert blob_sha == expected
    blob_receipts.append({"name": name, "blob_sha1": blob_sha, "verified_against_discovered_directory": True, "record": rec})
record(ROOT / "transfer-source-theory-md.txt")
ci = fragment["label_package"]["creation_info"]
result = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "partial_source_contract; no frozen acquisition or inference contract",
    "specimen": "PHerc0009B", "segment_long_id": fragment["segment_long_id"],
    "volume_id": ci["target"]["volume"],
    "candidate_arrays_or_predictions_inspected": False,
    "proven_facts": {
        "published_ct_level2": surface_array,
        "source_spacing_zyx_um": [2.401, 9.604, 9.604],
        "label_level2": array_metadata,
        "mask_group_metadata": group_metadata,
        "label_package_inventory_complete": True,
        "label_package_inventory": inventory,
        "existing_published_groups": groups_found,
        "level2_single_shard_bytes": {
            role: next(x["byte_length"] for x in inventory if x["key"] == prefix + role + ".zarr/2/c/0/0") for role in ["inklabels", "supervision", "validation"]},
    },
    "source_to_label_transform": {
        "intended_algorithm_status": "verified in pinned public source",
        "pinned_library_commit": head,
        "package_commit_declared_by_release": ci["package_commit"],
        "offset_yx_full_resolution_source_canvas_pixels": [-38.351, -44.944],
        "operation": "Subtract source-canvas offset from mapped source label coordinates before categorical gathering; write at target-canvas indices.",
        "extra_offset_to_apply_to_a_correctly_transferred_output": [0.0, 0.0],
        "actual_published_runtime_transform_execution_verified": False,
        "actual_published_runtime_transform": None,
        "why_not_verified": "Public actual transfer reports/provenance are absent; dirty publisher cannot be reconstructed from clean base or current helper. No label/prediction alignment inspection was performed.",
        "declared_affine": ci["geometry"]["affine"],
        "null_affine_is_not_assumed_to_prove_identity": True,
        "residuals_must_not_be_applied_again": True,
        "release_ome_translations": "none; scales only",
        "same_mask_shape_does_not_prove_registration": True,
    },
    "mapping_validity": {
        "catalog_claim": ci["conventions"],
        "actual_array_in_complete_published_prefix": False,
        "alternate_public_validity_path_found": False,
        "actual_array_encoding_or_alignment_verified": False,
        "declared_values_only": {"unmapped": 0, "seam_filled": 128, "measured": 255},
        "measured_only_mask_currently_constructible": False,
        "S_or_V_not_substituted_for_geometry_validity": True,
    },
    "render_depth_contract": {
        "renderer_commit": source_commit,
        "recorded_flip_normals": True, "recorded_xy_flip": -1, "recorded_rotate": 0,
        "num_slices": 109, "explicit_source_attrs_slice_step": 1.0,
        "nominal_zero_offset_index_zero_based": 54,
        "position_formula_native_ct_xyz": "B + (k - 54) * D, D=normalized negative cached geometric normal, under no hidden accumulation override",
        "distance_formula_um": "(k - 54) * 2.401 along D; not native CT Z",
        "normal_sampling": "nearest cached mesh-node normal, not bilinear normal interpolation",
        "xyz_sampling": "float32 bilinear surface coordinates followed by trilinear native CT sampling; clipped uint8 truncation",
        "accumulation": "recorded accum-type=max; accum-step omitted, archived default0, so no projection under recorded/default settings; full argv unavailable",
        "supplied_annotation_plane_index": None,
        "supplied_annotation_projection_depth_range": None,
        "annotation_original_overview_filename_only": ci["source"]["rasters"]["overview"],
        "filename22_42_not_treated_as_current109_depth_indices": True,
        "source_depth_pooling_selection": None,
        "prepared_depth_count": None,
        "pooled_label_plane": None,
        "no109_count_based_borrowing_of13_97": True,
    },
    "xy_representation_phase": {
        "source_canvas_size_xy": surface_attrs["canvas_size"],
        "label_canvas_size_yx": group_metadata["inklabels"]["canvas_size"],
        "label_pyramid_declared_rule": "[::2, ::2] categorical decimation",
        "ct_pyramid_source_code_rule": "2 × 2 mean, rounded (sum+cnt//2)//cnt at each level, preserves source Z",
        "level2_label_representative_level0_index": "(4i,4j)",
        "level2_ct_level0_support_interior": "rows[4i,4i+4), cols[4j,4j+4); centroid(4i+1.5,4j+1.5)",
        "phase_difference_original_pixels_yx": [1.5, 1.5],
        "phase_difference_nominal_canvas_um_yx": [3.6015, 3.6015],
        "phase_difference_level2_pixels_yx": [0.375, 0.375],
        "phase_is_kernel_support_centroid_not_fitted_registration_residual": True,
        "native3d_displacement": "mesh-dependent and unverified; no mesh arrays read",
        "automatic_shift_or_resampling_chosen": False,
    },
    "pooling_math_only_not_a_selected_window": {
        "four_plane_mean_block_centroid_source_index": "a + 4j + 1.5",
        "signed_centroid_distance_um": "(a + 4j + 1.5 - 54) * 2.401",
        "integer_four_slice_block_cannot_center_exactly_on_integer54": True,
        "borrowing_a13_21planes_label10_would_centroid_at": {"source_index": 54.5, "normal_distance_um": 1.2005},
        "this_arithmetic_does_not_establish_annotation_depth": True,
    },
    "unresolved_execution_gates": [
        "Actual transfer report/source-target mesh/affine/offset execution or equally strong published provenance",
        "Mapping-validity array with verified canonical alignment, or explicitly revised scientific participation policy without a measured-only claim",
        "Annotation/render depth or projection provenance sufficient to choose an explicit model window/pooling phase",
        "Actual mask binary encodings and participating classes via a future frozen label-only acquisition",
    ],
    "limits": {"annotation_source": "transferred; annotated_on unknown; review against an ink prediction; human-only/optical ancestry unverified",
               "explicit_original29_training_roots": "PHerc0009B absent; teacher/pretraining exposure still unknown",
               "CT_license": "CC BY-NC 4.0; annotation terms before redistribution unresolved",
               "current_research_authorization": "standing user authorization persists; execution contract not frozen"},
    "network": {"https_attempts": len(ledger["requests"]), "response_body_bytes": ledger["response_body_bytes"],
                "http200": sum(x["status"] == 200 for x in ledger["requests"]),
                "http404": sum(x["status"] == 404 for x in ledger["requests"]),
                "redirects_followed": 0, "automatic_retries": 0, "arrays_or_images_read": 0},
}
(ROOT / "surface-contract.json").write_text(json.dumps(result, indent=2) + "\n")
for path, rec in records.items():
    data = Path(path).read_bytes()
    assert len(data) == rec["byte_length"] and hashlib.sha256(data).hexdigest() == rec["sha256"]
receipts = {"status": "public_source_contract_receipts", "source_records": list(records.values()),
            "library_blob_checks": blob_receipts,
            "library_revision": head,
            "renderer_revision": source_commit,
            "network": result["network"],
            "all_source_records_verified_unchanged_before_after_builder": True,
            "scope": "metadata and text only, no arrays/predictions/tensors/models, prior files unchanged"}
(ROOT / "SOURCE_RECEIPTS.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps({"status": result["status"], "sources": len(records), "network": result["network"]}))
