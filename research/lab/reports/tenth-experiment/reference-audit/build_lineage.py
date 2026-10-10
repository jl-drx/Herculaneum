"""Build text/JSON receipts from retained reports and metadata; no tensor imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

lab = Path("/workspace/herculaneum-lab")
out = Path(__file__).resolve().parent
inputs = {}


def read(path):
    path = Path(path)
    body = path.read_bytes()
    inputs[str(path)] = {"path": str(path), "sha256": hashlib.sha256(body).hexdigest(), "byte_length": len(body)}
    return json.loads(body)


def receipt(path):
    path = Path(path)
    body = path.read_bytes()
    rec = {"path": str(path), "sha256": hashlib.sha256(body).hexdigest(), "byte_length": len(body)}
    inputs[str(path)] = rec
    return rec


reports = lab / "reports"
exposure_path = reports / "seventh-experiment/model-exposure-ledger.json"
exposure = read(exposure_path)
actual_models = read(exposure["source_path"])
assert inputs[exposure["source_path"]]["sha256"] == exposure["source_sha256"]
train = read(reports / "eighth-experiment/implementation/final-plan/training-plan.json")
preflight = read(reports / "eighth-experiment/implementation/final-plan/root-preflight.json")
old_panel = read(reports / "seventh-experiment/panel-sources/acquisition/panel-provenance.json")
fresh_panel = read(reports / "eighth-experiment/data-design/acquisition/panel-provenance.json")
fresh_descriptor = read(reports / "eighth-experiment/data-design/acquisition/fresh-descriptor.json")
w029_path = reports / "sixth-experiment/data-contract/selection-v2/provenance.json"
w029 = read(w029_path)
w013_path = lab / "data/experiments/known-ink-control/provenance.frozen.json"
w013 = read(w013_path)
source_reviews = [
    lab / "RESEARCH_CHARTER.md",
    reports / "ninth-experiment/NEXT_STUDY_DRAFT.md",
    reports / "ninth-experiment/PROTOCOL.md",
    reports / "eighth-experiment/PROTOCOL.md",
    reports / "seventh-experiment/panel-sources/SOURCE_REVIEW.md",
    reports / "seventh-experiment/panel-sources/SOURCE_RECEIPTS.json",
    reports / "eighth-experiment/data-design/DATA_DESIGN.md",
    reports / "eighth-experiment/data-design/SOURCE_RECEIPTS.json",
    reports / "sixth-experiment/data-contract/DATA_REVIEW.md",
    reports / "sixth-experiment/data-contract/DATA_VERIFICATION.json",
    reports / "sixth-experiment/data-contract/SOURCE_AUDIT.md",
    reports / "seventh-experiment/native-reference/NATIVE_FEASIBILITY.md",
    reports / "seventh-experiment/native-reference/candidate-contract.json",
    reports / "native-reference-preflight-20261008/source-preflight.md",
    reports / "fourth-experiment/source-terms-check.md",
    reports / "ninth-experiment/source-audit/LABEL_SMOOTHING.md",
    lab / "data/methods-review/tutorial5.md",
    lab / "data/methods-review/paper.txt",
    lab / "data/methods-review/sources.json",
    lab / "data/methods-review/ink_9um-README.md",
    lab / "data/experiments/known-ink-control/upstream-dataset-README.md",
    lab / "data/experiments/known-ink-control/upstream-public-data-terms.md",
    lab / "data/experiments/known-ink-control/upstream-legacy-data-LICENSE.txt",
]
for path in source_reviews:
    receipt(path)
candidate_panel = read(reports / "seventh-experiment/panel-sources/candidate-panel.json")
for rec in candidate_panel["retained_evidence_records"]:
    assert receipt(rec["path"])["sha256"] == rec["sha256"]


def known_exposure(segment, family="public_2p4_level2_zmean4"):
    result = {}
    for role, model in exposure["models"].items():
        matches = [r for r in model["representations"] if r["segment"] == segment and r["family"] == family]
        result[role] = {"configured_representation_present": bool(matches),
                        "raw_label_root_class": matches[0]["label_root_class"] if matches else None,
                        "historical_pixel_loss_exposure": "unknown",
                        "upstream_teacher_or_pretraining_exposure": "unknown"}
    return result


rows = [{"id": "pherc1667-w013-control", "physical_specimen": "PHerc1667",
         "bbox_source_xywh": w013["bbox_source_xywh"],
         "public_segment": w013["segment_long_id"], "source_surface_url": w013["source_surface_url"],
         "provenance_record": inputs[str(w013_path)], "current_role": "used_pipeline_positive_control",
         "new_adaptation_exposure": False, "new_calibration_exposure": False,
         "fresh_for_tenth": False, "annotation_origin": "supplied transferred public manual reference; human-only ancestry unverified",
         "configured_model_exposure": known_exposure("pherc1667-w013")}]
rows.append({"id": "pherc1667-w029", "physical_specimen": "PHerc1667",
             "bbox_source_xywh": w029["bbox_source_xywh"], "public_segment": w029["segment_long_id"],
             "source_surface_url": w029["source_surface_url"], "provenance_record": inputs[str(w029_path)],
             "arrays_from_retained_receipt_not_reloaded": w029["arrays"],
             "current_role": "reused_development_and_threshold_calibration",
             "participation": "validation_mask > 0; actual acquired and common finite/evaluated; fixed64 margin",
             "primary_counts_from_preflight": preflight["counts"]["pherc1667-w029"],
             "new_adaptation_exposure": False, "new_calibration_exposure": True,
             "fresh_for_tenth": False, "configured_model_exposure": known_exposure("pherc1667-w029")})
adaptation_ids = {r["id"] for r in train["adaptation"]}
exposure_by_id = {r["id"]: r for r in train["exposure"]}
for panel, is_eighth in [(old_panel, False), (fresh_panel, True)]:
    for crop in panel["crops"]:
        provenance_record = crop["provenance_record"]
        prov = read(provenance_record["path"])
        assert inputs[provenance_record["path"]]["sha256"] == provenance_record["sha256"]
        identifier = crop["crop_id"]
        adapting = identifier in adaptation_ids
        row = {"id": identifier, "physical_specimen": crop["specimen"],
               "bbox_source_xywh": crop["bbox_source_xywh"], "public_segment": prov["public_segment"],
               "source_surface_url": prov["source_surface_url"], "provenance_record": provenance_record,
               "arrays_from_retained_receipt_not_reloaded": prov["arrays"],
               "annotation_origin": prov["annotation_lineage"],
               "configured_model_exposure": known_exposure(prov["segment"]),
               "new_adaptation_exposure": adapting, "new_calibration_exposure": not adapting and not is_eighth,
               "fresh_for_tenth": False, "within_specimen_native3d_independence": "unverified",
               "primary_counts_from_preflight": preflight["counts"].get(identifier)}
        if adapting:
            row.update({"current_role": "adaptation_entire_context; historical_validation_retired",
                        "manual_participation": "S > 0 & V == 0 & acquired",
                        "pseudo_participation": "acquired & finite & seed43 evaluated & outside manual; includes retiredV",
                        "pseudo_target": "retained seed43 raw score >=0.5",
                        "scheduled_support_exposure_from_frozen_plan": exposure_by_id[identifier],
                        "retired_validation_inklabels_used_as_training_targets": False})
        elif is_eighth:
            row.update({"current_role": "eighth_reserved_from_new_adaptation_and_calibration; ninth_reused_development_retention_diagnostic",
                        "participation": "supplied supervision, acquired, fixed64 margin and common evaluated/finite support",
                        "historical_validation_retained_separately": True,
                        "minimum_source_xy_gap_px_in_eighth": 128,
                        "specimen_excluded_from_new_gradient_updates": True})
        else:
            row.update({"current_role": "seventh_reference_diagnostic; eighth/ninth_development_and_threshold_calibration",
                        "participation": "validation, acquired, fixed64 margin and common evaluated/finite support"})
        rows.append(row)

models = {}
for role, m in actual_models["models"].items():
    models[role] = {k: m[k] for k in ["model_repository", "model_revision", "repository_path", "sha256", "size_bytes", "step"]}
    models[role]["configured_representations"] = exposure["models"][role]["representations"]
    models[role]["configured_representation_count"] = exposure["models"][role]["representation_count"]
    models[role]["actual_historical_pixel_loss_exposure"] = "unknown"
    models[role]["upstream_teacher_or_pretraining_exposure"] = "unknown"
    # Record config receipt only; never open a checkpoint or deserialize tensors.
    config = json.dumps(m["embedded_config"], sort_keys=True, separators=(",", ":")).encode()
    models[role]["embedded_config_canonical_json_sha256"] = hashlib.sha256(config).hexdigest()

fragment_prov = read(out / "retained-fragment-provenance.json")
catalog_rec = fragment_prov["catalog_record"]
assert receipt(catalog_rec["path"])["sha256"] == catalog_rec["sha256"]
candidate_specs = []
for sample, item in fragment_prov["samples"].items():
    attrs_path = out / (sample + "-surface_attributes.json")
    array_path = out / (sample + "-surface_level2_array.json")
    label_path = out / (sample + "-labels_level2_metadata.json")
    attrs, array, label = read(attrs_path), read(array_path), read(label_path)
    assert array["shape"][1:] == label["shape"]
    assert array["chunks"][1:] == [128, 128] and array["dtype"] == "|u1" and array["compressor"] is None
    ci = item["label_package"]["creation_info"]
    path = item["label_package"]["origins"][0]["path"]
    candidate = {"sample_id": sample, "specimen_type": "fragment", "status": "metadata_only_candidate_not_acquired_or_scored",
                 "proposed_role": "reserved_from_current_adaptation_and_calibration_pending_explicit_contract",
                 "segment_long_id": item["segment_long_id"], "target_volume_id": ci["target"]["volume"],
                 "source_surface_url": ci["target"]["render"].replace("s3://vesuvius-challenge-open-data", "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com").rstrip("/"),
                 "source_level": 2, "source_shape_zyx": array["shape"], "source_chunk_zyx": array["chunks"],
                 "source_spacing_zyx_um": next(a["coordinateTransformations"][0]["scale"] for a in attrs["multiscales"][0]["datasets"] if a["path"] == "2"),
                 "source_scan_properties": item["source_volume"]["properties"],
                 "label_prefix_url": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/" + path,
                 "label_level2_shape_yx": label["shape"], "label_zarr_format": label["zarr_format"],
                 "label_level2_metadata_record": inputs[str(label_path)],
                 "label_codec": label["codecs"], "label_binary_encoding": "unverified; no arrays read",
                 "published_roles_claimed_in_catalog": item["label_package"]["properties"]["arrays"] + ["mapping_validity"],
                 "label_origin": {"annotated_on": item["label_package"]["properties"]["annotated_on"], "registration": item["label_package"]["properties"]["registration"],
                                  "review_method": ci["quality"]["validity_decision"]["review_method"], "package_commit": ci["package_commit"], "publication_tier": ci["publication"]["tier"]},
                 "declared_mapping_geometry": ci["geometry"], "declared_mapping_transfer": ci["transfer"],
                 "declared_mapping_conventions": ci["conventions"], "declared_mapping_quality": ci["quality"],
                 "archived_renderer_metadata": {"commit": item["surface"]["creation_info"]["provenance"]["parameters"].get("commit"),
                                                "num_slices": item["surface"]["creation_info"]["provenance"]["parameters"].get("num-slices"),
                                                "flip_normals": item["surface"]["creation_info"]["provenance"]["parameters"].get("flip-normals"),
                                                "source_geometry": item["surface"]["creation_info"].get("derived_from"),
                                                "label_depth_relation_verified": False},
                 "original_models_explicit_configured_training_roots": "absent from all four retained 29-representation configs; upstream exposure unknown",
                 "current_prototype_adaptation_or_calibration_exposure": False,
                 "candidate_arrays_or_predictions_read_in_tenth": False,
                 "bbox_source_xywh": None, "source_depth_selection": None, "pooling": None, "label_depth_center": None,
                 "missing_contract_evidence": ["explicit source-to-label transform and whether published canvas offset is already applied",
                                               "annotation plane/normal/depth relation; plane count alone is insufficient",
                                               "actual S/V/mapping-validity metadata and value encodings", "human/optical annotation lineage and upstream teacher/pretraining support"],
                 "future384_source_payload_if_fullZ_nine_chunks": 9 * array["chunks"][0] * 128 * 128,
                 "future_source_payload_condition": "384-square context with source XY origin aligned to128; full source Z, uncompressed nine chunks",
                 "future_acquisition_contract_frozen": False}
    candidate_specs.append(candidate)

network_ledgers = [read(out / "metadata-request-ledger.json"), read(out / "metadata-followup-ledger.json")]
network = {"request_attempts": sum(len(x["requests"]) for x in network_ledgers),
           "response_body_bytes": sum(x["response_body_bytes"] for x in network_ledgers),
           "http200_count": sum(y["status"] == 200 for x in network_ledgers for y in x["requests"]),
           "failures": sum(y["status"] != 200 or "error_type" in y for x in network_ledgers for y in x["requests"]),
           "redirects_followed": 0, "automatic_retries": 0,
           "body_budget_bytes": 524288, "request_budget": 8,
           "scope": "Two CT metadata files plus one prefix inventory and one label metadata file per fragment. No array bodies."}
assert network["request_attempts"] == 8 and network["response_body_bytes"] <= 524288

matrix = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "read_only_lineage_audit",
          "reference_rows": rows, "original_models": models,
          "source_contract_for_existing_curated_cases": {"shape": "21 × H × W uint8", "source_level": 2, "source_depth_half_open": [13,97],
              "pooling": "groups of 4 float32 means; nearest-even uint8", "model_layer_half_open": [2,19], "label_plane": 10,
              "pixel_spacing_um": 9.596, "not_transferable_to_new_fragments_without_verified_contract": True},
          "global_qualification": {"supplied_reference_labels": "transferred manual and iterative/model-assisted lineage; not proven optical/human-only truth",
              "S_vs_V": "Current ordinary loader S&~V; V eval override. Historical actual trainer pin and pixel/context support unverified.",
              "supervision_zero": "Unknown outside explicitly supplied participation, not automatic negative.",
              "observed_zero_ct": "Actually acquired zero profiles remain observed data; occupancy is separate.",
              "native3d_separation": "2D context gaps do not prove no intersecting depth/normal/interpolation support within a specimen.",
              "known2924_correction_overlap_scope": "author all-seven pseudo model, not any of our four selected arms",
              "current_fresh_independent_reference_count": 0,
              "PHerc0800_role": "unlabeled target diagnostics only; no adaptation/calibration label truth established"},
          "metadata_only_fragment_candidates": candidate_specs, "network_receipt": network,
          "execution": {"model_calls": 0, "tensor_calls": 0, "candidate_array_reads": 0, "scan_or_label_array_acquisition": 0}}
(out / "reference-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
reservation = {"status": "unexecuted_candidate_reservation_proposal", "future_acquisition_contract_frozen": False,
               "recommendation": "PHerc0009B first contract audit; PHerc0500P2 secondary. Keep both specimens outside current adaptation, teacher-target creation and calibration until root assigns roles.",
               "specimen_level_exclusion": "All available aliases/representations of either candidate; no crop selected.",
               "claims_allowed_after_current_only_reservation": "Agreement retention on a new specimen reserved from our current updates/calibration, conditional on public labels and unknown historical ancestry.",
               "claims_not_established": ["historical/global model holdout", "optical or human-only ground truth", "new native PHerc0800 ink accuracy", "decipherment"],
               "required_before_any_arrays": "Freeze exact transform/depth/normal/pooling/mask support and bounded label-only selection; do not borrow [13,97) from matching 109-plane shape.",
               "candidates": candidate_specs}
(out / "candidate-reservation-spec.json").write_text(json.dumps(reservation, indent=2) + "\n")

receipt(out / "metadata-lookup-plan.json")
receipt(out / "metadata-followup-plan.json")
receipt(out / "lookup_metadata.py")
receipt(out / "lookup_metadata_followup.py")
for p in out.glob("*label-prefix-inventory.xml"):
    receipt(p)
for path, rec in inputs.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == rec["sha256"], "Source changed during audit: " + path

receipts = {"status": "source_and_metadata_receipts", "network": network,
            "prior_sources_verified_unchanged_before_after_this_builder": True,
            "source_records": sorted(inputs.values(), key=lambda x: x["path"]),
            "primary_urls_and_pins": {
                "official_model_card": "https://huggingface.co/scrollprize/ink_9um/raw/7109667e2607db1b90c37c8b09cb876ea7fe7bb1/README.md",
                "official_curated_dataset": "https://huggingface.co/buckets/scrollprize/datasets/tree/ink_9um",
                "tutorial_label_origins": "https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/07_tutorial5.md",
                "Russo_author_code": "https://github.com/DomRusso2/ink9um-dense/tree/dd5bc8fb300c810c6ec042cb7e068be78829d086",
                "catalog": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json",
                "paper": "https://arxiv.org/abs/2606.29085"},
            "limits": "Embedded configs are read from prior verified JSON, not checkpoints. Array receipts are referenced, not reloaded. Catalog snapshot is retained; bounded current metadata does not independently verify its annotation authorship or upstream model history."}
(out / "SOURCE_RECEIPTS.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps({"reference_rows": len(rows), "source_records": len(inputs), "network": network}))
