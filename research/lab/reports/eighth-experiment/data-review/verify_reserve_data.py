"""Independent offline reconstruction of the eighth fresh-reference data.

This script does not import the acquisition, selection, or inference scripts.
It reconstructs the selected annotations from retained compressed objects and
the 21-layer surface by uint16 sums and integer ties-to-even division by four.
It never writes to an input directory or makes a network request.
"""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import numcodecs


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "byte_length": path.stat().st_size}


def run(source_plan_path, panel_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Independent review output must be a fresh directory")
    pins = {}

    def pin(record):
        actual = receipt(record["path"])
        assert actual["sha256"] == record["sha256"]
        assert actual["byte_length"] == record["byte_length"]
        old = pins.setdefault(actual["path"], actual)
        assert old == actual
        return Path(actual["path"])

    def read(path):
        path = pin(receipt(path))
        return json.loads(path.read_text())

    pin(receipt(__file__))
    source_plan_path = Path(source_plan_path).resolve()
    panel_path = Path(panel_path).resolve()
    plan = read(source_plan_path)
    panel = read(panel_path)
    assert plan["evaluation_participation"] == "supervision_mask"
    assert plan["source_response_body_cap_bytes"] == 40 * 1024 * 1024
    assert panel["source_plan_record"] == receipt(source_plan_path)
    assert panel["status"] == "panel_source_complete_no_inference"
    for record in plan["input_records"]:
        pin(record)
    for key in ("http_module_record", "source_code_record", "selection_record", "label_plan_record"):
        pin(plan[key])
    selection = read(plan["selection_record"]["path"])
    label_plan = read(plan["label_plan_record"]["path"])
    assert selection["status"] == "selection_complete_no_scans_or_inference"
    assert selection["plan_record"] == plan["label_plan_record"]
    selected = {c["crop_id"]: c for case in selection["cases"].values() for c in case["selected"]}
    assert set(selected) == {c["crop_id"] for c in plan["crops"]} == {c["crop_id"] for c in panel["crops"]}
    assert len(selected) == 2
    assert label_plan["case_order"] == ["pherc0814-46527"]
    assert label_plan["evaluation_participation"] == "supervision_mask"
    assert plan["old_used_contexts"] == label_plan["old_used_contexts"]
    assert plan["minimum_context_gap_px"] == label_plan["minimum_context_gap_px"] == 128
    descriptor = read(pin(panel["fresh_descriptor_record"]))
    assert descriptor["status"] == "fresh_reference_complete_no_inference"
    assert descriptor["evaluation_participation_role"] == "supplied_supervision"
    assert descriptor["source_plan_record"] == receipt(source_plan_path)
    descriptor_cases = {c["id"]: c for c in descriptor["cases"]}
    assert set(descriptor_cases) == set(selected)

    def rectangle_gap(a, b):
        # Center separations minus half widths give nonnegative axis gaps.
        dx = max(0, abs((a[0] + a[2]/2) - (b[0] + b[2]/2)) - (a[2]+b[2])/2)
        dy = max(0, abs((a[1] + a[3]/2) - (b[1] + b[3]/2)) - (a[3]+b[3])/2)
        return int(max(dx, dy))

    gap_checks = []
    for old in plan["old_used_contexts"]:
        old_provenance = read(pin(old["provenance_record"]))
        assert old_provenance["bbox_source_xywh"] == old["bbox_source_xywh"]
        for crop in plan["crops"]:
            gap = rectangle_gap(crop["bbox_source_xywh"], old["bbox_source_xywh"])
            assert gap >= 128
            gap_checks.append({"fresh": crop["crop_id"], "old": old["crop_id"], "source_xy_rectangle_gap_px": gap})
    mutual_gap = rectangle_gap(plan["crops"][0]["bbox_source_xywh"], plan["crops"][1]["bbox_source_xywh"])
    assert mutual_gap >= 128

    def check_ledger(record, cap, request_cap, plan_sha):
        path = pin(record)
        ledger = read(path)
        assert ledger["plan_sha256"] == plan_sha
        assert ledger["response_body_cap_bytes"] == cap
        assert ledger["maximum_http_request_attempts"] == request_cap
        assert ledger["body_bytes"] == sum(r["body_bytes"] for r in ledger["requests"])
        assert ledger["body_bytes"] <= cap and len(ledger["requests"]) <= request_cap
        artifacts = {}
        for a in ledger["artifacts"]:
            key = a["object_key"]
            assert key not in artifacts and a["verified"]
            pin({"path": str(path.parent / a["local_file"]), "sha256": a["sha256"], "byte_length": a["byte_length"]})
            responses = [r for r in ledger["requests"] if r["object_key"] == key]
            assert 1 <= len(responses) <= 6
            assert responses[-1]["status"] == 200
            assert responses[-1]["body_sha256"] == a["sha256"]
            assert responses[-1]["body_bytes"] == a["byte_length"]
            assert all(r["status"] in (301, 302, 303, 307, 308) for r in responses[:-1])
            artifacts[key] = a
        assert all(r["object_key"] in artifacts for r in ledger["requests"])
        return path.parent, ledger, artifacts

    label_raw, labels_ledger, label_artifacts = check_ledger(
        selection["label_request_ledger_record"], label_plan["label_response_body_cap_bytes"],
        label_plan["maximum_http_request_attempts"], digest(plan["label_plan_record"]["path"]))
    source_raw, sources_ledger, source_artifacts = check_ledger(
        panel["source_request_ledger_record"], plan["source_response_body_cap_bytes"],
        plan["maximum_http_request_attempts"], digest(source_plan_path))
    assert panel["response_body_bytes"] == sources_ledger["body_bytes"]
    assert panel["http_request_attempts"] == len(sources_ledger["requests"])
    source_by_key = {o["key"]: o for o in plan["objects"]}
    assert len(source_by_key) == len(plan["objects"])
    crop_records = {c["crop_id"]: c for c in panel["crops"]}
    results = []

    def verify_array(record, expected):
        path = pin(record)
        actual = np.load(path, allow_pickle=False)
        assert list(actual.shape) == record["shape"] == list(expected.shape)
        assert str(actual.dtype) == record["dtype"] == str(expected.dtype)
        assert np.array_equal(actual, expected), "Reconstructed array differs: " + path.name

    for crop in plan["crops"]:
        ident = crop["crop_id"]
        original = selected[ident]
        assert original["evaluation_participation"] == crop["evaluation_participation"] == "supervision_mask"
        x, y, width, height = crop["bbox_source_xywh"]
        assert width == height == 384 and x % 128 == y % 128 == 0
        assert crop["bbox_source_xywh"] == original["bbox_source_xywh"]
        assert crop["source_depth_slice"] == [13, 97] and crop["model_layer_slice"] == [2, 19]
        spec = label_plan["cases"][crop["case"]]
        label_objects = {o["key"]: o for o in spec["objects"]}
        codec = numcodecs.get_codec(json.loads(pin(spec["label_zarray_record"]).read_text())["compressor"])
        annotations = {role: np.zeros((384, 384), np.uint8) for role in ("inklabels", "supervision_mask", "validation_mask")}
        coverage = {role: np.zeros((384, 384), bool) for role in annotations}
        for obj_receipt in original["source_objects"]:
            obj = label_objects[obj_receipt["key"]]
            key = hashlib.sha256((crop["case"] + " " + obj["url"]).encode()).hexdigest()
            a = label_artifacts[key]
            assert a["sha256"] == obj_receipt["sha256"] and a["byte_length"] == obj["expected_byte_length"]
            assert a["etag"].strip('"') == obj["expected_etag"]
            decoded = np.frombuffer(codec.decode((label_raw / a["local_file"]).read_bytes()), np.uint8)
            assert decoded.size == 21 * 128 * 128
            decoded = decoded.reshape((21, 128, 128))
            assert set(np.unique(decoded)) <= {0, 1}
            assert np.count_nonzero(decoded[:10]) == np.count_nonzero(decoded[11:]) == 0
            yy, xx = obj["chunk_yx"]
            sy, sx = yy * 128 - y, xx * 128 - x
            assert 0 <= sy <= 256 and 0 <= sx <= 256
            role = obj["role"]
            assert not coverage[role][sy:sy+128, sx:sx+128].any()
            annotations[role][sy:sy+128, sx:sx+128] = decoded[10]
            coverage[role][sy:sy+128, sx:sx+128] = True
        assert all(c.all() for c in coverage.values())
        fine = np.zeros((109, 384, 384), np.uint8)
        observed = np.zeros((384, 384), bool)
        for key in crop["source_chunk_keys"]:
            obj = source_by_key[key]
            assert obj["url"] == spec["source_surface_url"] + "/2/" + "/".join(map(str, obj["chunk_zyx"]))
            cache_key = hashlib.sha256(("scan:" + crop["case"] + " " + obj["url"]).encode()).hexdigest()
            a = source_artifacts[cache_key]
            assert a["byte_length"] == obj["expected_byte_length"] == 109 * 128 * 128
            cube = np.frombuffer((source_raw / a["local_file"]).read_bytes(), np.uint8).reshape((109, 128, 128))
            zz, yy, xx = obj["chunk_zyx"]
            sy, sx = yy * 128 - y, xx * 128 - x
            assert zz == 0 and 0 <= sy <= 256 and 0 <= sx <= 256
            assert not observed[sy:sy+128, sx:sx+128].any()
            fine[:, sy:sy+128, sx:sx+128] = cube
            observed[sy:sy+128, sx:sx+128] = True
        assert observed.all()
        pooled = np.zeros((21, 384, 384), np.uint8)
        for z in range(21):
            total = np.zeros((384, 384), np.uint16)
            for source_z in range(13 + 4*z, 17 + 4*z):
                total += fine[source_z].astype(np.uint16)
            quotient, remainder = np.divmod(total, np.uint16(4))
            round_up = (remainder > 2) | ((remainder == 2) & ((quotient & 1) != 0))
            pooled[z] = (quotient + round_up.astype(np.uint16)).astype(np.uint8)
        val = annotations["validation_mask"].astype(bool)
        sup = annotations["supervision_mask"].astype(bool)
        ink = annotations["inklabels"].astype(bool)
        interior = np.zeros((384, 384), bool)
        interior[64:-64, 64:-64] = True
        primary = sup & observed & interior
        occupancy = np.logical_or.reduce(pooled[2:19] != 0)
        expected = {**annotations, "validation_membership": val, "supervision_membership": sup,
                    "current_loader_label_mask": sup & ~val, "primary_annotation_mask": primary,
                    "surface": pooled, "acquisition_observed_mask": observed,
                    "source_observed_nonzero_mask": occupancy,
                    "evaluation_participation_mask": sup & observed}
        for name, record in original["arrays"].items():
            verify_array(record, expected[name])
        provenance_record = crop_records[ident]["provenance_record"]
        provenance = read(pin(provenance_record))
        assert provenance["evaluation_participation"] == "supervision_mask"
        assert provenance["evaluation_participation_role"] == "supplied_supervision"
        assert set(provenance["arrays"]) == set(expected)
        for name, values in expected.items():
            verify_array(provenance["arrays"][name], values)
        counts = {"whole_validation": int(val.sum()), "whole_evaluation_participation": int(sup.sum()),
                  "whole_validation_positive": int((val & ink).sum()),
                  "whole_validation_negative": int((val & ~ink).sum()), "primary_annotation_pixels": int(primary.sum()),
                  "primary_positive": int((primary & ink).sum()), "primary_negative": int((primary & ~ink).sum()),
                  "supervision": int(sup.sum()), "raw_supervision_validation_overlap": int((sup & val).sum()),
                  "conditional_current_loader_training_pixels": int((sup & ~val).sum()),
                  "acquisition_observed_pixels": int(observed.sum()),
                  "source_observed_nonzero_pixels_selected17": int(occupancy.sum()),
                  "primary_observed_zero_profile_pixels": int((primary & ~occupancy).sum())}
        assert counts == provenance["counts"] == crop_records[ident]["counts"]
        assert min(counts["primary_positive"], counts["primary_negative"]) >= 256
        assert provenance["bbox_source_xywh"] == crop["bbox_source_xywh"]
        assert provenance["source_shape_zyx"] == spec["source_shape_zyx"]
        assert provenance["spacing_zyx_um"] == [9.596]*3
        fresh = descriptor_cases[ident]
        assert fresh["arrays"] == provenance["arrays"]
        assert fresh["counts"] == counts
        assert fresh["bbox_source_xywh"] == crop["bbox_source_xywh"]
        assert fresh["physical_group"] == "PHerc0814"
        assert fresh["evaluation_participation"] == "supervision_mask"
        assert fresh["provenance_record"] == provenance_record
        results.append({"crop_id": ident, "bbox_source_xywh": crop["bbox_source_xywh"], "counts": counts,
                        "annotation_arrays_equal": True, "surface_all_bytes_equal": True,
                        "observed_zeros_retained": True})

    assert {(c["stage"], c["case"], c["role"]) for c in panel["checked_metadata"]} == {
        (stage, o["case"], o["role"]) for stage in ("before", "after") for o in plan["metadata_objects"]}
    for checked in panel["checked_metadata"]:
        obj = next(o for o in plan["metadata_objects"] if o["case"] == checked["case"] and o["role"] == checked["role"])
        assert checked["sha256"] == obj["expected_sha256"]
        assert checked["byte_length"] == obj["expected_byte_length"]
        key = hashlib.sha256((checked["stage"] + ":" + checked["case"] + " " + obj["url"]).encode()).hexdigest()
        assert source_artifacts[key]["sha256"] == checked["sha256"]
    assert len(panel["checked_metadata"]) == 2 * len(plan["metadata_objects"])
    assert len(source_artifacts) == len(plan["objects"]) + 2 * len(plan["metadata_objects"])
    for value in pins.values():
        assert receipt(value["path"]) == value, "Input changed during independent review"
    output.mkdir(parents=True)
    result = {"schema_version": 1, "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "review_code": receipt(__file__), "source_plan": receipt(source_plan_path), "panel": receipt(panel_path),
              "input_receipts_before_and_after": list(pins.values()), "verified_unique_inputs": len(pins),
              "method": "Offline raw object reconstruction; uint16 four-plane sums and integer quotient/remainder ties-to-even division by4, without acquisition code imports",
              "network_requests": 0, "model_calls": 0, "source_body_bytes": sources_ledger["body_bytes"],
              "source_http_attempts": len(sources_ledger["requests"]), "source_chunk_count": len(plan["objects"]),
              "source_payload_bytes": sum(o["expected_byte_length"] for o in plan["objects"]),
              "label_body_bytes": labels_ledger["body_bytes"], "label_http_attempts": len(labels_ledger["requests"]),
              "source_xy_gap_checks": gap_checks, "fresh_mutual_source_xy_rectangle_gap_px": mutual_gap,
              "evaluation_participation": "supplied_supervision; historical validation retained separately",
              "fresh_descriptor": receipt(panel["fresh_descriptor_record"]["path"]),
              "crops": results, "limitations": ["Public reference annotation agreement, not authenticated ink accuracy",
              "Existing model training/online-validation universe includes these representations; historical pixel and teacher exposure unknown",
              "PHerc0814 is excluded from current PHerc0139 adaptation; fresh contexts are one specimen and sourceXY gap is not a native3D independence proof",
              "No geometry mask acquired; observed zero profiles are retained"]}
    target = output / "DATA_VERIFICATION.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "crops": len(results), "unique_inputs": len(pins),
                      "path": str(target), "sha256": digest(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--panel-provenance", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.source_plan, args.panel_provenance, args.output)
