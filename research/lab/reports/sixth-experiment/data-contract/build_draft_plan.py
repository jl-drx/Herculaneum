"""Build a reviewable metadata-only w029 acquisition proposal, without network."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASE = "pherc1667-w029"
SOURCE = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1667/segments/"
          "20251212185248-w029_20251212185248662_flatboi/surface-volumes/"
          "2.399um-0.22m-78keV-volume-20251217075048.zarr")
LABEL_ROOT = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w029"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ledger = json.loads((ROOT / "metadata-request-ledger.json").read_text())
    rows = {r["local_file"].split("-", 2)[2]: r for r in ledger["requests"] if r.get("local_file")}
    metadata = {}
    for name in ["w029-surface-zattrs.json", "w029-surface-level0-zarray.json", "w029-surface-level2-zarray.json"] + [
        f"w029-{kind}-{tail}.json" for kind in ["inklabels", "supervision_mask", "validation_mask"]
        for tail in ["zattrs", "level0-zarray"]
    ]:
        r = rows[name]
        metadata[name] = {"path": r["local_file"], "sha256": r["sha256"], "byte_length": r["body_bytes"], "url": r["url"]}
    inventories = {}
    inventory_files = []
    for kind in ["inklabels", "supervision_mask", "validation_mask"]:
        stem = "validation" if kind == "validation_mask" else kind
        entries = {}
        for f in sorted(ROOT.glob(f"*w029-{stem}-object-inventory-page-*.json")):
            inventory_files.append({"path": f.name, "sha256": digest(f)})
            for e in json.loads(f.read_text()):
                key = e["path"].rsplit("/", 1)[-1]
                if e["type"] == "file" and key.startswith("0."):
                    z, y, x = map(int, key.split("."))
                    if z != 0:
                        raise ValueError("Unexpected depth chunk")
                    entries[(y, x)] = e
        if len(entries) != 75 * 62:
            raise ValueError("Incomplete label inventory")
        inventories[kind] = entries
    # Size 78 is the common object-size baseline, used only as a ranking proxy.
    # It is never interpreted as a decoded label/mask value.
    potential = {k for k, e in inventories["validation_mask"].items() if e["size"] != 78}
    origins = {(y + dy, x + dx) for y, x in potential for dy in [-1, 0] for dx in [-1, 0]
               if 0 <= y + dy <= (9500 - 256) // 128 and 0 <= x + dx <= (7830 - 256) // 128}
    candidates = []
    for y, x in origins:
        keys = [(y + dy, x + dx) for dy in [0, 1] for dx in [0, 1]]
        candidates.append({"origin_yx": [y * 128, x * 128], "bbox_source_xywh": [x * 128, y * 128, 256, 256],
                           "proxy_validation_chunk_count": sum(k in potential for k in keys),
                           "proxy_possible_positive_chunk_count": sum(k in potential and inventories["inklabels"][k]["size"] != 78 for k in keys)})
    candidates.sort(key=lambda c: (-c["proxy_validation_chunk_count"], -c["proxy_possible_positive_chunk_count"], c["origin_yx"]))
    candidates = candidates[:16]
    union = {(c["origin_yx"][0] // 128 + dy, c["origin_yx"][1] // 128 + dx)
             for c in candidates for dy in [0, 1] for dx in [0, 1]}
    label_objects = []
    for kind in inventories:
        for y, x in sorted(union):
            e = inventories[kind][(y, x)]
            label_objects.append({"kind": kind, "chunk_zyx": [0, y, x], "url": LABEL_ROOT + f"/{CASE}_{kind}.zarr/0/0.{y}.{x}",
                                  "expected_byte_length": e["size"], "expected_etag": e["xetHash"],
                                  "upstream_xet_hash_is_not_plain_sha256": True, "upstream_mtime": e["mtime"], "upstream_uploaded_at": e["uploadedAt"]})
    plan = {"schema": "w029-data-plan-v1", "status": "draft_requires_parent_review_and_freeze", "selection_status": "no_label_chunks_read_no_crop_selected",
            "case": CASE, "sample_id": "PHerc1667", "target_volume_id": "20251217075048", "segment_id": "20251212185248",
            "segment_long_id": "20251212185248-w029_20251212185248662_flatboi", "source_surface_url": SOURCE,
            "source_level": 2, "source_shape_zyx": [109, 9500, 7830], "source_chunks_zyx": [109, 128, 128],
            "source_depth_slice": [13, 97], "pool_z": 4, "output_shape_zyx": [21, 256, 256], "model_layer_slice": [2, 19],
            "spacing_zyx_um": [9.596, 9.596, 9.596], "primary_margin_pixels": 64,
            "selection_rule": "Rank metadata-only chunk-aligned 256-square candidates by descending potential-validation chunk count, descending potential ink-label chunk count, then ascending source Y/X; retain first16. Sizes unequal to78 are ranking proxies only. Decode labels/masks in this fixed order and select first candidate with >=256 positive and >=256 negative pixels under validation>0 in fixed [64:192] square. No prediction/source-intensity selection, no threshold relaxation.",
            "minimum_primary_positive_pixels": 256, "minimum_primary_negative_pixels": 256, "candidates": candidates,
            "candidate_count_before_cap": len(origins), "label_objects": label_objects,
            "label_probe_success_body_bytes_bound": sum(o["expected_byte_length"] for o in label_objects),
            "label_probe_unique_xy_chunks": len(union), "label_probe_maximum_logical_requests": len(label_objects),
            "label_probe_response_body_cap_bytes": 512 * 1024, "source_acquisition_response_body_cap_bytes": 64 * 1024 * 1024,
            "selected_scan_chunk_count": 4, "selected_scan_success_body_bytes": 4 * 109 * 128 * 128,
            "automatic_retries": 0, "missing_object_policy": "stop; never zero-fill as observed evidence",
            "primary_domain": "validation_membership AND acquisition_observed AND fixed64pixel_interior; downstream intersect common evaluated/finite predictions; do not subtract raw supervision or nonzero occupancy",
            "actual_historical_training_overlap": {"status": "unknown", "mask_file": None},
            "conditional_current_loader_label_mask": "supervision>0 AND NOT(validation>0); not proof of historical label or input-context exposure",
            "geometry_validity": {"status": "not_acquired", "mask_file": None},
            "metadata_files": metadata, "inventory_files": inventory_files,
            "expected_outputs": ["surface.npy", "inklabels.npy", "supervision-mask.npy", "validation-mask.npy", "validation-membership.npy", "supervision-membership.npy", "current-loader-label-mask.npy", "acquisition-observed-mask.npy", "source-observed-nonzero-mask.npy", "primary-annotation-mask.npy", "provenance.json"],
            "source_nonzero_occupancy_definition": "np.any(surface[2:19] != 0, axis=0); diagnostic only, not geometry or metric-domain validity",
            "acquisition_script": "prepare_w029.py", "claim": "small mask-defined developmental annotation-agreement comparison; not independent physical/teacher holdout or authenticated text"}
    (ROOT / "data-plan.draft.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(candidates), "first_candidate": candidates[0], "label_objects": len(label_objects),
                      "label_success_bytes": plan["label_probe_success_body_bytes_bound"], "scan_success_bytes": plan["selected_scan_success_body_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
