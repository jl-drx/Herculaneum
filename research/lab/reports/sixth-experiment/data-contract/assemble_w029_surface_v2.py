"""Propose, then execute a separately SHA-pinned four-chunk surface assembly.

The proposal is offline. Acquisition requires a reviewed assembly-plan SHA and
this script's SHA. It never reads a model, checkpoint, prediction, or native CT.
Frozen selection files/dependencies are read only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checked_inputs(assembly):
    from select_w029_labels_v2 import verify_inputs
    plan_path = Path(assembly["label_plan_path"])
    plan = verify_inputs(plan_path, assembly["label_plan_sha256"], assembly["selector_sha256"])
    selection_path = Path(assembly["selection_path"])
    if sha(selection_path) != assembly["selection_sha256"]:
        raise ValueError("Selected label record changed")
    selection = json.loads(selection_path.read_text())
    if selection["plan_sha256"] != assembly["label_plan_sha256"] or selection["status"] != "label_selection_complete_scan_not_acquired":
        raise ValueError("Selection is not a completed label-only stage for this plan")
    for row in selection["arrays"].values():
        path = selection_path.parent / row["path"]
        if sha(path) != row["sha256"] or path.stat().st_size != row["byte_length"]:
            raise ValueError("Selected annotation array changed")
    if sha(__file__) != assembly["assembler_sha256"]:
        raise ValueError("Assembler source changed")
    if assembly["bbox_source_xywh"] != selection["selected"]["bbox_source_xywh"]:
        raise ValueError("Assembly crop differs from frozen label selection")
    row = assembly["archived_render_evidence"]
    if sha(row["path"]) != row["sha256"]:
        raise ValueError("Archived render evidence changed")
    evidence = json.loads(Path(row["path"]).read_text())
    snapshot = evidence["source_snapshot"]
    if sha(snapshot["path"]) != snapshot["sha256"]:
        raise ValueError("Original archived metadata snapshot changed")
    params = evidence["render_record"]["creation_info"]["provenance"]["parameters"]
    if evidence["target_volume_id"] != plan["target_volume_id"] or evidence["segment_id"] != plan["segment_id"] or params["num-slices"] != "109" or params["flip-normals"] != "true":
        raise ValueError("Archived renderer does not match this surface")
    x0, y0, width, height = assembly["bbox_source_xywh"]
    expected_keys = {(0, y0//128+dy, x0//128+dx) for dy in [0, 1] for dx in [0, 1]}
    if (width, height) != (256, 256) or x0 % 128 or y0 % 128:
        raise ValueError("Unexpected assembly crop")
    if len(assembly["source_objects"]) != 4 or {tuple(o["chunk_zyx"]) for o in assembly["source_objects"]} != expected_keys:
        raise ValueError("Assembly is not exactly the four selected source chunks")
    for obj in assembly["source_objects"]:
        key = "/".join(map(str, obj["chunk_zyx"]))
        if obj["url"] != plan["source_surface_url"] + "/2/" + key or obj["expected_byte_length"] != 109*128*128:
            raise ValueError("Source URL/length differs from selected source contract")
    return plan, selection, selection_path.parent


def propose(args):
    from select_w029_labels_v2 import verify_inputs
    plan = verify_inputs(args.label_plan, args.label_plan_sha256, args.selector_sha256)
    selection = json.loads(args.selection.read_text())
    if selection["plan_sha256"] != args.label_plan_sha256 or selection["status"] != "label_selection_complete_scan_not_acquired":
        raise ValueError("A completed label-only selection is required")
    x0, y0, width, height = selection["selected"]["bbox_source_xywh"]
    if width != 256 or height != 256 or x0 % 128 or y0 % 128:
        raise ValueError("Unexpected selected crop")
    objects = [{"url": plan["source_surface_url"] + f"/2/0/{y0//128+dy}/{x0//128+dx}",
                "chunk_zyx": [0, y0//128+dy, x0//128+dx], "expected_byte_length": 109 * 128 * 128}
               for dy in [0, 1] for dx in [0, 1]]
    check_rows = [plan["metadata_files"][name] for name in ["w029-surface-zattrs.json", "w029-surface-level2-zarray.json"]]
    assembly = {"schema": "w029-four-chunk-assembly-v2", "status": "draft_awaiting_parent_review_and_freeze",
                "label_plan_path": str(args.label_plan.resolve()), "label_plan_sha256": args.label_plan_sha256,
                "selector_sha256": args.selector_sha256, "selection_path": str(args.selection.resolve()),
                "selection_sha256": sha(args.selection), "assembler_sha256": sha(__file__),
                "archived_render_evidence": {"path": str((ROOT / "archived-w029-render-evidence.json").resolve()), "sha256": sha(ROOT / "archived-w029-render-evidence.json")},
                "bbox_source_xywh": [x0, y0, 256, 256], "source_objects": objects, "metadata_checks": check_rows,
                "successful_scan_body_bytes": sum(o["expected_byte_length"] for o in objects),
                "successful_pre_post_metadata_body_bytes": 2 * sum(r["byte_length"] for r in check_rows),
                "response_body_cap_bytes": 64 * 1024 * 1024, "maximum_http_request_attempts": 48,
                "maximum_http_requests_per_object": 6, "automatic_retries": 0,
                "missing_source_policy": "stop; no fill", "surface_output_shape_zyx": [21, 256, 256],
                "source_depth_slice": [13, 97], "model_layer_slice": [2, 19],
                "claim": "public mask-defined developmental annotation-agreement source; historical/physical/teacher holdout unverified"}
    checked_inputs(assembly)
    if args.assembly_plan.exists():
        raise FileExistsError("Refusing to replace assembly proposal")
    write_json(args.assembly_plan, assembly)
    print(json.dumps({"assembly_plan": str(args.assembly_plan), "sha256": sha(args.assembly_plan),
                      "successful_scan_body_bytes": assembly["successful_scan_body_bytes"],
                      "successful_pre_post_metadata_body_bytes": assembly["successful_pre_post_metadata_body_bytes"]}))


def acquire(args):
    if sha(args.assembly_plan) != args.assembly_plan_sha256:
        raise ValueError("Assembly plan does not match approved SHA")
    assembly = json.loads(args.assembly_plan.read_text())
    if sha(__file__) != args.assembler_sha256 or assembly["assembler_sha256"] != args.assembler_sha256:
        raise ValueError("Assembler does not match approved SHA")
    plan, selection, output = checked_inputs(assembly)
    if (output / "provenance.json").exists():
        raise FileExistsError("Source already published; inspect without rerunning")
    from bounded_http_v2 import fetch_object
    from select_w029_labels_v2 import decode_chunk, crop_statistics

    def retrieve(obj, namespace=""):
        return fetch_object(obj, output / "raw-surface", args.assembly_plan_sha256,
                            cap=assembly["response_body_cap_bytes"], ledger_name="source-request-ledger.json",
                            maximum_requests=assembly["maximum_http_request_attempts"], maximum_requests_per_object=6, namespace=namespace)[0]

    def check_metadata(stage):
        for row in assembly["metadata_checks"]:
            body = retrieve({"url": row["url"], "expected_byte_length": row["byte_length"]}, namespace=stage)
            if hashlib.sha256(body).hexdigest() != row["sha256"]:
                raise ValueError("Remote source metadata differs from frozen contract")

    check_metadata("before")
    spec_row = plan["metadata_files"]["w029-surface-level2-zarray.json"]
    spec = json.loads((Path(assembly["label_plan_path"]).parent / spec_row["path"]).read_text())
    if spec["shape"] != [109, 9500, 7830] or spec["chunks"] != [109, 128, 128] or spec.get("compressor") is not None or spec.get("dimension_separator") != "/":
        raise ValueError("Source contract differs from reviewed four-chunk contract")
    raw = np.zeros((109, 256, 256), dtype=np.uint8)
    observed = np.zeros((256, 256), dtype=bool)
    x0, y0, _, _ = assembly["bbox_source_xywh"]
    for obj in assembly["source_objects"]:
        _, cy, cx = obj["chunk_zyx"]
        dy, dx = cy - y0//128, cx - x0//128
        if dy not in [0, 1] or dx not in [0, 1]:
            raise ValueError("Source object lies outside selected crop")
        raw[:, dy*128:(dy+1)*128, dx*128:(dx+1)*128] = decode_chunk(retrieve(obj), spec)
        observed[dy*128:(dy+1)*128, dx*128:(dx+1)*128] = True
    if not np.all(observed):
        raise ValueError("Incomplete actual source acquisition")
    surface = np.rint(raw[13:97].astype(np.float32).reshape(21, 4, 256, 256).mean(axis=1)).astype(np.uint8)
    annotations = {k: np.load(output / row["path"], allow_pickle=False) for k, row in selection["arrays"].items()}
    stats = crop_statistics(annotations)
    if stats != {k: selection["selected"][k] for k in stats}:
        raise ValueError("Selected annotations/statistics changed")
    if stats["primary_positive_pixels"] < 256 or stats["primary_negative_pixels"] < 256:
        raise ValueError("Fixed class-count gate fails")
    check_metadata("after")
    checked_inputs(assembly)
    if sha(args.assembly_plan) != args.assembly_plan_sha256:
        raise ValueError("Executing assembly plan changed")
    validation = annotations["validation_mask"] > 0
    supervision = annotations["supervision_mask"] > 0
    interior = np.zeros((256, 256), dtype=bool)
    interior[64:192, 64:192] = True
    arrays = dict(selection["arrays"])
    derived = {"surface": ("surface.npy", surface), "validation_membership": ("validation-membership.npy", validation),
               "supervision_membership": ("supervision-membership.npy", supervision),
               "current_loader_label_mask": ("current-loader-label-mask.npy", supervision & ~validation),
               "acquisition_observed_mask": ("acquisition-observed-mask.npy", observed),
               "source_observed_nonzero_mask": ("source-observed-nonzero-mask.npy", np.any(surface[2:19] != 0, axis=0)),
               "primary_annotation_mask": ("primary-annotation-mask.npy", validation & observed & interior)}
    for key, (filename, array) in derived.items():
        target = output / filename
        if target.exists():
            raise FileExistsError("Refusing to replace derived source input")
        np.save(target, array, allow_pickle=False)
        arrays[key] = {"path": filename, "sha256": sha(target), "byte_length": target.stat().st_size,
                       "shape": list(array.shape), "dtype": str(array.dtype)}
    checked_inputs(assembly)
    if sha(args.assembly_plan) != args.assembly_plan_sha256:
        raise ValueError("Assembly plan changed before publication")
    provenance = {"schema": "w029-reference-source-v1", "status": "source_complete_no_inference",
                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "case": plan["case"], "segment": plan["case"],
                  "sample_id": plan["sample_id"], "segment_id": plan["segment_id"], "segment_long_id": plan["segment_long_id"],
                  "target_volume_id": plan["target_volume_id"], "source_surface_url": plan["source_surface_url"],
                  "bbox_source_xywh": assembly["bbox_source_xywh"], "source_level": 2, "source_shape_zyx": [109, 9500, 7830],
                  "shape_zyx": [21, 256, 256], "dtype": "uint8", "array_sha256": arrays["surface"]["sha256"],
                  "arrays": arrays, "binary_foreground": selection["binary_foreground"],
                  "annotation_encoding_assertions": {role: {"observed_unique_values": stats["unique_values"][role],
                      "foreground_present_in_observed_values": selection["binary_foreground"][role] in stats["unique_values"][role],
                      "membership_rule": "raw value > 0"} for role in selection["binary_foreground"]},
                  "pixel_spacing_um": 9.596, "spacing_zyx_um": [9.596, 9.596, 9.596],
                  "source_depth_slice": [13, 97], "depth_pooling": "float32 mean4, np.rint nearest-even, uint8",
                  "model_layer_slice": [2, 19], "model_source_depth_slice": [21, 89], "annotation_center_layer": 10,
                  "pooled_plane_source_ranges": [[13+4*i, 17+4*i] for i in range(21)],
                  "annotation_center_source_planes": [53, 57], "annotation_center_source_mean_index": 54.5,
                  "depth_orientation": "Increasing published source Z retained. The hashed archived render record documents flip-normals=true; no fitted offset/sign/reversal applied.",
                  "archived_render_evidence": assembly["archived_render_evidence"],
                  "selection_before_predictions": True, "selection_sha256": assembly["selection_sha256"],
                  "label_plan_sha256": assembly["label_plan_sha256"], "assembly_plan_sha256": args.assembly_plan_sha256,
                  "assembler_sha256": args.assembler_sha256, "correction_version": "Actual attrs annotation_source_results=results-fixed and aligned-public-21slice-v2 rebuild; object inventory dates 18 August 2026.",
                  "mask_statistics": stats, "primary_domain": plan["primary_domain"],
                  "source_observed_nonzero_mask_definition": "np.any(surface[2:19] !=0,axis=0); occupancy diagnostic only",
                  "actual_historical_training_overlap": plan["actual_historical_training_overlap"],
                  "conditional_current_loader_label_mask": plan["conditional_current_loader_label_mask"],
                  "geometry_validity": plan["geometry_validity"], "physical_teacher_holdout": "unverified",
                  "claim": plan["claim"], "source_request_ledger_sha256": sha(output/"raw-surface/source-request-ledger.json")}
    write_json(output / "provenance.json", provenance)  # publication last
    print(json.dumps({"status": provenance["status"], "surface_sha256": provenance["array_sha256"], "mask_statistics": stats}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    proposal = sub.add_parser("propose")
    proposal.add_argument("--label-plan", type=Path, required=True)
    proposal.add_argument("--label-plan-sha256", required=True)
    proposal.add_argument("--selector-sha256", required=True)
    proposal.add_argument("--selection", type=Path, required=True)
    proposal.add_argument("--assembly-plan", type=Path, required=True)
    run = sub.add_parser("acquire")
    run.add_argument("--assembly-plan", type=Path, required=True)
    run.add_argument("--assembly-plan-sha256", required=True)
    run.add_argument("--assembler-sha256", required=True)
    args = parser.parse_args()
    propose(args) if args.phase == "propose" else acquire(args)


if __name__ == "__main__":
    main()
