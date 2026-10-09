"""Write a metadata-only staged proposal. Does not acquire or decode science."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT.parent.parent
BASE = "https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/"
SURFACE = BASE + "working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/"


def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    if (ROOT / "GEOMETRY_ACQUISITION_PLAN.json").exists():
        raise RuntimeError("Do not replace an existing proposal")
    source_names = [
        "nineteenth-experiment/optical-audit/CANDIDATE_CONTRACT.json",
        "nineteenth-experiment/optical-audit/OPTICAL_CANDIDATE_AUDIT.md",
        "nineteenth-experiment/optical-audit/SOURCE_PINS.json",
        "nineteenth-experiment/optical-audit/frag6-volume-20231121152933-meta.json",
        "nineteenth-experiment/optical-audit/frag6-volume-20231130112027-meta.json",
        "nineteenth-experiment/optical-audit/frag6-volume-20231201112849-meta.json",
        "nineteenth-experiment/optical-audit/frag6-volume-20231201120546-meta.json",
        "nineteenth-experiment/optical-audit/frag6-transform-53-70.json",
        "nineteenth-experiment/optical-audit/frag6-volume-zarr-index.html",
        "fifteenth-experiment/source-audit/frag6-surface-processing-index.html",
        "fifteenth-experiment/source-audit/frag6-surface-index.html",
        "seventeenth-experiment/source-review/SOURCE_AUDIT.json",
        "seventeenth-experiment/source-review/PPM_SERIALIZATION_EVIDENCE.json",
        "seventeenth-experiment/source-review/historical-inkid-ppm.py",
        "seventeenth-experiment/PPM_PREDECODE_AMENDMENT.md",
        "seventeenth-experiment/PPM_ZERO_MAPPING_CLARIFICATION.md",
        "seventeenth-experiment/acquisition/README.md",
        "seventeenth-experiment/acquisition/prepare_acquisition.py",
        "seventeenth-experiment/acquisition/regional_http.py",
        "seventeenth-experiment/implementation/regional_ppm.py",
        "eleventh-experiment/ct-acquisition/READER_CONTRACT.md",
        "eleventh-experiment/optical-depth/sampling-contract.json",
        "eleventh-experiment/optical-depth/LayerFromPPM.cpp",
        "eleventh-experiment/optical-depth/LineGenerator.cpp",
        "eleventh-experiment/optical-depth/LayerTexture.cpp",
    ]
    sources = [pin(REPORTS / name) for name in source_names]
    save("SOURCE_PINS.json", dict(schema="herculaneum.frag6-plan-sources.v1", sources=sources))
    ledger = json.loads((ROOT / "head-request-ledger.json").read_text())
    objects = []
    for entry in ledger["requests"]:
        assert entry["http_status"] == 200 and entry["response_body_read"] is False
        headers = {k.lower(): v for k, v in entry["headers"]}
        objects.append(dict(url=entry["url"], total_bytes=int(headers["content-length"]),
                            etag=headers["etag"], last_modified=headers["last-modified"],
                            accept_ranges=headers["accept-ranges"], observed_utc=entry["completed_utc"]))
    a_body = 1025 + 3 * (9 + 3 + 1543 + 65537)
    b_body = 243 * 144 + 243 * 768 + 1024
    native_header = 108 * (9 + 3 + 1543)
    native_ranges = 54 * 8 * 2 + 216 * 768 + 108 * 1024
    surface_ranges = 3 * (9 * 2 + 9 * 768 + 1024)
    c_body = 4 * 1024**2 + 1 + native_header + native_ranges + surface_ranges
    plan = dict(
        schema="herculaneum.frag6-staged-geometry-proposal.v1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="PROPOSAL_ONLY_NOT_AN_EXECUTION_GO",
        actual_work=dict(head_attempts=4, response_body_reads=0, retained_body_bytes=0,
                         scientific_arrays_loaded=0, model_calls=0, gpu_spend_gbp=0,
                         uploads=0, credential_access=False),
        candidate=dict(alias="Frag6", sample="PHerc51Cr4Fr8", parent="PHerc0051",
                       parent_alias_reservation="All PHerc51/PHerc0051 spelling, cradle, fragment, energy, resolution and representation variants; do not substring-match PHerc1451.",
                       primary_native_uuid="20231121152933", primary_native_shape_zyx=[8855, 2260, 6300],
                       primary_native_pitch_um=3.24,
                       other_native_uuids=["20231130112027", "20231201112849", "20231201120546"],
                       frame_binding="Unknown; primary UUID is a declared candidate, not yet the authenticated PPM source.",
                       role="Geometry/provenance work only. No gradient, teacher, calibration, normalization fitting, model/depth/crop/alignment selection.",
                       eventual_claim_ceiling="Prospectively locked agreement with supplied labels on a locally unscored parent, with unresolved upstream exposure and annotation lineage; not independent confirmation or authenticated text."),
        present_evidence=dict(objects=objects, canvas_shape=None, tiff_dtype=None,
                              ppm_byte_order=None, plane_offset_rule=None, interpolation=None,
                              physical_ink_depth=None, anatomical_normal=None,
                              listing_plane_count=65,
                              warning="No inference of center 32, one-voxel step, physical direction, or Paris [7,58)/factor-three pooling from 65 filenames."),
        transport_contract=dict(
            tls="Verified system TLS; exact HTTPS URLs on dl.ash2txt.org only.",
            authorization="Each stage requires root-frozen source/implementation/fixture/derived-request pins before execution; no automatic stage advance.",
            retry_count=0, redirects_followed=0, content_encoding="identity",
            headers="Reject duplicate singleton identity/framing headers, Transfer-Encoding, unexpected status, absent/weak ETag or changed total length. Every range uses If-Match from the retained HEAD/header pin.",
            range="HTTP 206, exact Content-Range/Content-Length and object total. Reject HTTP 200 without reading the body; never fall back to a full file.",
            multipart="Exact requested parts only; reject duplicate, missing, coalesced, overlapping or unrequested parts. Per-part framing cap 768 bytes, plus 1024 per response.",
            accounting="Durably reserve response-body cap before each attempt, retain failed reservation and partial bytes; no retries. These are response-body bounds, not total wire traffic.",
            stream_chunk_bytes=65536, connect_timeout_seconds=20, per_request_wall_seconds=60,
            process_rss_acceptance_bytes=268435456, process_address_space_limit_bytes=1073741824,
            resource_note="Sequential reads, no full PPM/TIFF/volume allocation. RSS acceptance is measured; RLIMIT_AS and an external controller provide separate process bounds."),
        stages=[
            dict(id="A_headers", ready_for_implementation=True, execute_now=False,
                 purpose="Establish PPM canvas/record extent and TIFF layout/metadata before any numerical geometry. Read any bounded ImageDescription that could contain actual source/render arguments.",
                 initial_requests=[dict(url=objects[0]["url"], method="GET", byte_range_inclusive=[0,1023], if_match=objects[0]["etag"], total_bytes=objects[0]["total_bytes"])] +
                                  [dict(url=o["url"], method="GET", byte_range_inclusive=[0,7], if_match=o["etag"], total_bytes=o["total_bytes"]) for o in objects[1:]],
                 tiff_expansion_rule="Per TIFF after its 8-byte classic header: exact 2-byte IFD count at the validated pointer; then exact 2+12*N+4-byte complete IFD at that same pointer, N<=128. If tag 270 ImageDescription is ASCII, external, nonoverlapping with image data and <=65536 bytes, one exact-range read. Inline ASCII is parsed from the IFD. No other out-of-line tag payload is acquired in this stage.",
                 tiff_checks=["Classic TIFF magic 42; parse each file's declared II/MM independently; reject BigTIFF at this frozen stage.",
                              "Width and height agree across the three TIFFs and result.ppm. No flip/transpose inferred from dimensions.",
                              "For the later sparse reader require one IFD, unsigned 16-bit, one channel, uncompressed, one contiguous strip, min-is-black, Orientation 1, compatible fill order/planar/sample-format/predictor. Explicit TIFF defaults may be recorded; unsupported layouts stop, never imply a new decoder.",
                              "Validate count/type, every pointer/span, strip byte count W*H*2 and total-object bounds before any pixel range. Reject duplicate tags and IFD cycles.",
                              "Treat header buffer bytes outside structural fields as uninterpreted; do not decode incidental pixel bytes."],
                 ppm_checks=["Exact LF UTF-8 header terminated by <>\\n within 1024 bytes; reject duplicate/unknown required format fields or malformed dimensions.",
                             "Require positive width/height >=11, ordered:true, dim:6, type:double, supported version and total size exactly data_offset+W*H*48.",
                             "Preserve header and comments verbatim. Comments may supply provenance candidates but do not authenticate an execution merely by naming a volume.",
                             "Do not numerically decode the PPM prefix or infer endianness from the TIFFs/current host."],
                 maximum_requests=13, maximum_reserved_response_body_bytes=a_body,
                 budget_formula="1025 + 3*(9 + 3 + 1543 + 65537): one conservative excess byte per single-range attempt.",
                 output="STRUCTURAL_CONTRACT.json plus exact raw headers/IFDs/description bodies, ledger and SHA256 pins. Stop here for review, including an explicit absent-render-metadata result if necessary."),
            dict(id="B_sparse_ppm", ready_for_implementation=False, execute_now=False,
                 requires="Passing A, reviewed conditional PPM numerical contract and a separately frozen exact span plan derived only from W,H,data_offset.",
                 lattice="For i,j=0..8: y=floor((i+1)*(H-1)/10), x=floor((j+1)*(W-1)/10). Keep all 81 centers in row-major order. Each contributes its 3x3 neighborhood; no labels, images, predictions, occupancy-driven replacement or regional ranking.",
                 range_formula="For each center and dy in {-1,0,1}: begin=data_offset+48*((y+dy)*W+x-1); end=begin+143. Exactly 243 parts, 729 logical records, 34992 useful bytes. Reject overlapping/duplicate scheduled spans if the observed canvas is too small.",
                 byte_order="Either authenticated exact producer byte order, or separately qualified phase17-style conditional invariant comparison of exactly little/big float64 over all 729 records. A candidate requires every nonsentinel record finite and normal norm within 1e-5 of 1; exact all-six-zero records remain unmapped. Exactly one candidate must pass and there must be mapped data. Both/neither passing stops. Never use CT fit, labels or sample display to choose byte order.",
                 checks="Record mapping coverage including unmapped sites; finite geometry; primary-volume bounds as a consistency test only; neighboring surface derivatives/Jacobian and normal-dot-tangent checks. Unit normals and bounds cannot select among same-shaped energy scans or authenticate physical anatomy.",
                 fixed_native_validation_sites="The nine lattice centers with i,j in {1,4,7}; no substitutions. At least six of these nine centers must have valid complete PPM neighborhoods, otherwise native comparison stops without replacement.",
                 maximum_requests=1, maximum_reserved_response_body_bytes=b_body,
                 budget_formula="34992 useful + 243*768 MIME framing + 1024 response framing = 222640.",
                 output="Conditional serialization/geometry evidence with all sentinel/invalid observations retained; exact immutable nine-site native schedule. No label acquisition or model inference."),
            dict(id="C_declared_native_reconstruction", ready_for_implementation=False, execute_now=False,
                 blocked_currently=True,
                 purpose="Numerically test an independently declared PPM-to-native render mapping on fixed sites; test whether the declared signed offsets are distinguishable from their sign-reversed counterfactual.",
                 preconditions=["Exact native UUID bridge for current result.ppm plus renderer version/arguments or equivalent publisher record; the directory name alone is insufficient.",
                                "An explicit surface pixel-to-PPM raster map, declared finite offsets for file indices 0,32,64, PPM normal handling, coordinate order/origin, native frame transform, interpolation, clipping and output quantization. Current values remain null.",
                                "Any resampling or affine operation, including voxel-vs-physical units, must be declared before native pixels. Do not silently apply the 53-to-70 keV transform or substitute 7.91 um data.",
                                "This budget supports trilinear sampling of eight uint16 neighbors in the declared primary native TIFF volume only; other kernels, projections, transforms requiring extra data or compressed/tiled TIFFs require a new plan, not fallback.",
                                "Independent synthetic fixtures must establish the numerical error bound before real pixels. At most one declared recipe and its exact offset-sign reversal; no searched shifts, axes, offsets, spacings, image warps, fitted gains or alternative source-volume selection."],
                 inventory_request=dict(url=BASE+"volumes/20231121152933/", method="GET", maximum_body_bytes=4194304,
                                        require="Complete retained directory inventory containing exact native slice filenames; MIME/UTF-8/list completeness checks. No guessed filename templates or trailing unchecked inventory. Failure/oversize stops."),
                 samples="At up to nine fixed valid sites, compare surface pixels from 00,32,64.tif with declared native interpolation at T(p+delta[k]*n). The counterfactual uses T(p-delta[k]*n); it is a sign diagnostic, not an anatomical alternative. Keep all fixed sites, marking invalid ones without replacement.",
                 native_request_derivation="Up to 9*3*2=54 native query coordinates. Trilinear neighborhoods require at most 108 distinct z-slice objects and 216 contiguous x-pair row spans (4 bytes each). Deduplicate exact spans only. Each native object uses 3 validated classic-TIFF header/IFD requests and at most one multipart pixel request. All requested objects/spans are frozen after metadata/PPM parsing and before any native pixel body is read.",
                 surface_request_derivation="Three existing TIFFs, one multipart request each with at most nine uint16 center pixels. Use A's offset/layout/ETag and exact byte spans; no crops or full rows.",
                 comparison_rule="Use the independently implemented declared interpolation/quantization. Retain every residual and validity bit. Numerical acceptance tolerance must follow source precision/rounding plus a synthetic error bound frozen before acquisition; absent sufficient source evidence stops, rather than inventing a tolerance from real residuals.",
                 sign_identifiability="Report signed order only when declared reconstruction satisfies its frozen error bound and the sign-reversed counterfactual is excluded beyond the combined bound at at least three fixed sites for each nonzero-offset endpoint. Constant/near-constant data, zero offsets, both passing or both failing mean sign unresolved. This describes signed offsets relative to stored normals, not inward/outward orientation or ink depth.",
                 volume_identity_limitation="A match validates sparse compatibility with the declared primary volume, not historical execution identity or exclusion of the other three scans. Same shape and bounds cannot disambiguate energies. Without independent identity/transform provenance, report compatible-with-candidate only and keep the CT evaluation gate closed.",
                 full_stack_limitation="Only indices 0,32,64 are numerically sampled. An authenticated renderer formula may declare all other offsets, but the other 62 TIFF headers/pixels remain unverified. Do not present three checked planes as a verified 65-plane model input contract.",
                 maximum_requests=436, maximum_reserved_response_body_bytes=c_body,
                 budget_components=dict(inventory_reserved_bytes=4194305,
                                        native_header_requests=324, native_header_reserved_bytes=native_header,
                                        native_pixel_requests=108, native_pixel_reserved_bytes=native_ranges,
                                        surface_pixel_requests=3, surface_pixel_reserved_bytes=surface_ranges),
                 output="Native compatibility and sign-identifiability report for exactly three planes and fixed sites. No label quality-control, model accuracy, crop/depth choice, or fresh-holdout admission."),
        ],
        total_future_ceiling=dict(http_attempts=450, reserved_response_body_bytes=a_body+b_body+c_body,
                                  retained_artifact_bytes=33554432, measured_peak_rss_bytes=268435456,
                                  process_address_space_limit_bytes=1073741824,
                                  aggregate_external_controller_wall_seconds=1800,
                                  network_requests_including_completed_head=454,
                                  gpu_spend_gbp=0, first_gpu_pilot_total_cap_gbp=10,
                                  qualification="Bounds describe this proposal if all gated stages execute. No future request has been executed or authorized by this artifact."),
        hard_stops=["Source pin/ETag/total-size drift; no silent refresh or mirrored-source substitution.",
                    "Unexpected HTTP status, redirect, retry, unsupported content encoding/framing/MIME, incomplete/truncated body, cap or deadline exceeded.",
                    "Malformed/unsupported PPM or TIFF structure, out-of-bounds spans or unsupported layouts; preserve observations then stop.",
                    "Ambiguous conditional PPM byte order, missing valid fixed sites, or unresolved normal units/volume frame/render recipe.",
                    "Native interpolation outside declared volume; do not clamp unless the source rule explicitly declares it, and report boundary policy.",
                    "Nonunique sign discrimination or unexplained residuals; no adaptive offset/crop/source search.",
                    "Any requested labels, optical image, mask, model execution, output-dependent tuning, paid resource, upload or publication lies outside this geometry plan."],
        immediate_recommendation="Implement and independently fixture-test stage A only; root may then freeze its <=13-request, <=202301-byte plan. Its result will resolve canvas/layout and either expose embedded render provenance or document the exact missing fields. Do not start stage C from the present metadata.",
        source_pins=pin(ROOT/"SOURCE_PINS.json"), head_receipt=pin(ROOT/"head-request-ledger.json"),
    )
    assert a_body == 202301 and b_body == 222640 and c_body == 4663451
    assert a_body+b_body+c_body == 5088392
    save("GEOMETRY_ACQUISITION_PLAN.json", plan)
    print(json.dumps(dict(stage_budgets=[(s["id"],s["maximum_requests"],s["maximum_reserved_response_body_bytes"]) for s in plan["stages"]], total=plan["total_future_ceiling"])))


if __name__ == "__main__":
    main()
