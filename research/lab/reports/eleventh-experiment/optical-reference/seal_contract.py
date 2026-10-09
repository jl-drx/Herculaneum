"""Seal the completed text-only audit; writes only this new report directory."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin, unquote
import hashlib
import json
import re

root = Path(__file__).resolve().parent
lab = root.parents[2]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, obj):
    path = root / name
    if path.exists():
        raise RuntimeError(f"Preserve existing file: {path}")
    path.write_text(json.dumps(obj, indent=2) + "\n")

inputs = [
    lab / "RESEARCH_CHARTER.md",
    lab / "reports/tenth-experiment/NEXT_STUDY_DRAFT.md",
    lab / "reports/tenth-experiment/reference-audit/LINEAGE.md",
    lab / "reports/tenth-experiment/reference-audit/reference-matrix.json",
    lab / "reports/seventh-experiment/model-exposure-ledger.json",
    lab / "data/methods-review/tutorial5.md",
    lab / "data/experiments/pherc0800-native-ct-consistency/metadata.full.json.gz",
]
before = {str(p): {"sha256": digest(p), "byte_length": p.stat().st_size} for p in inputs}
ledger = json.loads((root / "request-ledger.json").read_text())
assert len(ledger["attempts"]) == 20
assert ledger["response_body_bytes"] == 38918
assert all(e["status"] == "complete" and e["http_status"] == 200 for e in ledger["attempts"])
assert all(digest(Path(e["saved_path"])) == e["sha256"] for e in ledger["attempts"])
assert before[str(inputs[-1])]["sha256"] == "7ec8dcaa3000b9716724c26a1991ee416491d8a6b3615238ca90ae11540abdd3"

assets = []
pattern = re.compile(r'<tr><td class="link"><a href="([^"]+)" title="([^"]+)">.*?</td><td class="size">(.*?)</td><td class="date">(.*?)</td></tr>', re.S)
for entry in ledger["attempts"]:
    if not entry["name"].endswith("index.html"):
        continue
    for href, title, size, date in pattern.findall(Path(entry["saved_path"]).read_text()):
        assets.append({"name": unquote(title), "url": urljoin(entry["url"], href),
                       "directory_reported_size": size,
                       "directory_reported_date": date,
                       "directory_evidence_path": entry["saved_path"],
                       "directory_sha256": entry["sha256"],
                       "payload_or_header_inspected": False,
                       "size_is_exact": False})
write("asset-inventory.json", {"assets": assets, "assets_are_directory_entries_not_payload_receipts": True})

base = "https://dl.ash2txt.org/fragments/Frag4/PHercParis1Fr39.volpkg/"
surface = base + "working/54keV_exposed_surface/PHercParis1Fr39_54keV_surface_volume/"
exposed = base + "working/54keV_exposed_surface/"
extras = exposed + "extras/"
model_ledger = json.loads(inputs[4].read_text())
models = {}
for name, model in model_ledger["models"].items():
    representations = model["representations"]
    matches = [r for r in representations if any(t in r["segment"].lower() for t in ["frag4", "fr39", "paris1"])]
    models[name] = {"representation_count": model["representation_count"],
                    "physical_groups": sorted({r["physical_key"].split(":")[0] for r in representations}),
                    "explicit_frag4_alias_matches": matches,
                    "configured_frag4_membership": False,
                    "upstream_teacher_and_pretraining_exposure": "unknown",
                    "historical_pixel_loss_exposure": "unknown"}

contract = {
    "status": "text_metadata_contract_only; no header/array acquisition or evaluation",
    "physical_specimen": "PHercParis1Fr39",
    "known_aliases": ["Frag4", "PHerc. Paris 1 Fr 39", "PHercParis1-Fg39"],
    "not_same_identity_as": ["PHercParis4", "PHerc1667/community Scroll 4"],
    "package_base_url": base,
    "modern_scan_segment_id": None,
    "model_exposure": models,
    "proposed_future_group_exclusion": "all aliases/representations outside new adaptation, teacher targets and calibration; not executed by this audit",
    "historical_kaggle_holdout_is_not_current_model_independence": True,
    "surface": {
        "url": surface, "metadata_url": surface + "meta.json",
        "metadata_local_path": str(root / "legacy-surface-meta.json"),
        "metadata_sha256": digest(root / "legacy-surface-meta.json"),
        "uuid": "20230301213512", "declared_shape_zyx": [65, 9791, 6064],
        "declared_voxelsize": 3.24,
        "declared_intensity_min_max": [0, 65535],
        "declared_description": "PHercParis1Fr39 54KeV merged, segmented and converted with vc_layers to surface volume",
        "tiff_urls": [surface + f"{i:02d}.tif" for i in range(65)],
        "directory_reported_size_per_tiff": "113.2 MiB (rounded)",
        "tiff_dtype_bitdepth_endianness_compression_strip_tile_layout": "unverified; min/max alone do not prove TIFF encoding",
        "index_names": "00.tif through 64.tif, exhaustive directory text; zero-padded filename index is not a physical depth convention",
        "label_reference_plane": None, "sample_depth_positions": None,
        "normal_orientation_vs_modern_training": "unverified",
        "vc_layers_command_and_revision": None,
    },
    "region_descriptor": {
        "url": extras + "PHercParis1Fr39_54keV_surface_volume.json",
        "sha256": digest(root / "legacy-surface-render.json"),
        "schema_version": "0.1", "type": "region", "invert_normals": False,
        "links_surface_mask_inklabels_ppm": True,
        "links_registered_ir_image_explicitly": False,
        "grid_bounds_as_published": [[0, 0, 6064, 4895], [0, 4895, 6064, 9790]],
        "grid_endpoint_semantics": "unknown; do not silently fix final row or use grid split as independent holdout",
    },
    "optical": {
        "original_ir_url": base + "working/reference/PHerc%20Paris%201%20Cornici%202-FRR%2039-01392-1000nmS40-FStop%208-Lens%3D.tif",
        "original_ir_directory_size": "6.0 MiB (rounded)",
        "wavelength_evidence": "1000nm in filename; no acquisition metadata inspected",
        "candidate_registered_ir_url": extras + "PHercParis1Fr39_54keV_ir.png",
        "candidate_registered_ir_directory_size": "11.5 MiB (rounded)",
        "registered_ir_dimensions": None,
        "alignment_project_url": extras + "PHercParis1Fr39_54keV_alignment.psd",
        "alignment_project_directory_size": "197.8 MiB (rounded)",
        "registration_transform_controls_residuals": "unverified",
        "image_payloads_and_headers_read": False,
    },
    "labels": {
        "ink_url": exposed + "PHercParis1Fr39_54keV_inklabels.png",
        "ink_directory_size": "96.1 KiB (rounded)",
        "mask_url": exposed + "PHercParis1Fr39_54keV_mask.png",
        "mask_directory_size": "53.2 KiB (rounded)",
        "png_canvas_color_bitdepth_encoding": "unverified",
        "annotation_author_and_independent_optical_derivation": "unverified",
        "mask_meaning_and_supervised_negative_participation": "unverified",
        "no_modern_supervision_validation_masks_identified": True,
        "eligible_claim_now": "declared asset association only; no metrics",
        "eligible_claim_after_scoped_array_read_if_provenance_remains_unknown": "agreement against supplied fragment reference, not independent ink accuracy",
    },
    "native_mapping": {
        "ppm_url": extras + "PHercParis1Fr39_54keV_surface_volume.ppm",
        "ppm_directory_size": "2.7 GiB (rounded)",
        "header_format_channels_coordinate_order_normals_units_frame": "unverified",
        "source_volume_identifier_and_merge_standardization_transform": "unverified",
        "ct54_zarr_url": base + "volumes_zarr/54keV_3.24um_.zarr/",
        "ct54_level0_shape_zyx": [9231, 1968, 5808],
        "ct54_level0_chunks_zyx": [128, 128, 128],
        "ct54_dtype": "<u2", "ct54_zarr_format": 2,
        "ct54_one_decoded_chunk_bytes": 4194304,
        "ct54_units_in_zattrs": None,
        "ct54_nominal_spacing_basis": "3.24um in directory name and separately rendered-surface voxelsize; no native CT physical-unit attribute",
        "ct88_zarr_url": base + "volumes_zarr/88keV_3.24um_.zarr/",
        "ct88_checked_scope": "directory existence only; do not substitute for54keV render",
    },
    "terms": {
        "url": "https://dl.ash2txt.org/LICENSE.txt",
        "sha256": digest(root / "legacy-LICENSE.current.txt"),
        "scope_text": "Scrolls1-4 and Fragments1-6 EduceLab-Scrolls; includesFrag4",
        "redistribution_requires_written_approval": True,
        "public_textual_revelation_and_related_code_restriction": "express written VesuviusChallenge/PapyrologyTeam consent outsideofficialDiscord",
        "citation_requirement": "EduceLab-Scrolls, Parsons etal2023 arXiv2304.02084; exact abstract/methods terms retained",
        "registration_or_terms_acceptance_performed": False,
    },
    "next_gate": {
        "not_authorized_by_current_text_only_budget": "image or PPM headers, any arrays, model runs",
        "recommended_first_stage": "root-authorized bounded structural-header probes; stop on non206 range response or any hardcap breach; noautomatic redirects/retries",
        "recommended_header_assets": [surface + "00.tif", base + "working/reference/PHerc%20Paris%201%20Cornici%202-FRR%2039-01392-1000nmS40-FStop%208-Lens%3D.tif", extras + "PHercParis1Fr39_54keV_ir.png", exposed + "PHercParis1Fr39_54keV_inklabels.png", exposed + "PHercParis1Fr39_54keV_mask.png", extras + "PHercParis1Fr39_54keV_surface_volume.ppm"],
        "bounded_full_object_candidate_sizes_from_directory": {"ink_png": "96.1KiB", "mask_png": "53.2KiB", "original_ir_tif": "6.0MiB", "registered_ir_png": "11.5MiB"},
        "bounded_full_object_gate": "after accessconditions and a distinct payload budget are resolved, retrieve only required small optical/label files under capped streaming; no full surface stack/PPM/PSD; hashes and exactbytes mustsupersede rounded directory sizes",
        "large_object_route": "range-select TIFF strips/tiles only afterlayoutchecked, or mapped nativeCTchunks after PPM format/frame/source relation verified; no route selectedyet",
        "remaining_scientific_gates": ["opticalwarp/residual and labelauthor/history", "maskpositives/negatives/unknownparticipation", "surface labelplane/depthstep/normal/referenceframe", "CT-to-PPM merge/standardization relation if native route used", "source-supported distinct3.24um input/resampling recipe", "historicalteacher/backboneexposure remainsqualified"],
        "evaluation": "rootfreezes distinctphysical group, label-only class/support selection and unchangedbaseline+retainedfinalarms before anypredictions; no newtraining immediately",
    },
    "scope": {"https_attempts": 20, "response_body_bytes": 38918,
              "all_http_status": 200, "failed_https_attempts": 0,
              "verified_tls": True, "automatic_redirects": False, "automatic_retries": 0,
              "images_imageheaders_ct_label_arrays_models_read": False,
              "paid_resources_outreach_publication_termsacceptance": False},
}
write("reference-contract.json", contract)
after = {str(p): digest(p) for p in inputs}
assert all(after[p] == record["sha256"] for p, record in before.items())
write("SOURCE_RECEIPTS.json", {
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "local_prior_sources": before,
    "prior_source_hashes_unchanged_during_report_construction": True,
    "read_prior_json_receipts_not_original_tensors_or_arrays": True,
    "tutorial_source_pin": {"commit": "e0bbb8b40a2db58b1d71864f286eb85717e59e64", "source_path": "scrollprize.org/docs/07_tutorial5.md", "relevant_lines": [50, 67, 101, 483, 593, 595], "illustrated_fragment_is_not_Frag4": "PHercParis2Fr47"},
    "requests": ledger,
    "local_nonnetwork_failure": {"command_kind": "systemPythonbounded_lookup.py invocation", "error": "ModuleNotFoundError: requests", "occurred_before_network_attempt_or_ledger_append": True, "resolution": "existing/workspace/.venvs/herculaneum/bin/python;noinstall/environmentchange"},
    "payloads_not_requested": ["imageheaders", "images", "CTchunks", "labelarrays", "PPM", "PSD", "checkpoints"],
    "network_budget_exhausted_by_attemptcount": True,
})
entries = [{"path": p.name, "byte_length": p.stat().st_size, "sha256": digest(p)}
           for p in sorted(root.iterdir()) if p.is_file() and p.name != "MANIFEST.json"]
write("MANIFEST.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "files": entries, "scope": "new optical-reference report only"})
print(json.dumps({"files_sealed": len(entries), "requests": len(ledger["attempts"]), "body_bytes": ledger["response_body_bytes"], "prior_source_hashes_unchanged": True}))
