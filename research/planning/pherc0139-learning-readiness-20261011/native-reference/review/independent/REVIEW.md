# Independent native annotation intake check

11 October2026. **PASS for the retained annotation cohort and planned CT request set.** This review imported neither `intake.py` nor its replay helpers. It decoded all six retained shards with official Zarr3.4.0, recomputed mask selection and class counts with independent NumPy code, and checked source receipts/frozen input hashes. No network, CT voxel reading, model or training was used.

The recalculated output contains exactly265 frozen cores:109w035,54w039,102w044. Every ordered core ID and every per-core class count matches the saved CSVs. Full decoded-array SHA256 values match both retained decoder receipts. Labels and masks contain only0 and255.

| Winding | Eligible cores | Both classes | No ink | No non-ink | Supervised pixels in eligible cores | Ink / non-ink inside eligible supervision |
|---|---:|---:|---:|---:|---:|---:|
|w035|109|75|34|0|1,066,610|333,871 /732,739|
|w039|54|36|18|0|571,603|182,380 /389,223|
|w044|102|86|15|1|1,187,621|450,775 /736,846|

All single-class cores remain present. Full-array stored ink outside supervision is62,129/2,519/4,172pixels for w035/w039/w044; the corresponding eligible-core counts are11,086/2,370/4,153. These are correctly reported and excluded from supervised positive/negative counts. Unlabelled support is not silently promoted to negative labels.

## Source and timing evidence

All six source lengths, local SHA256 values and recorded ETags match the pinned source plan and acquisition ledger. The frozen protocol/code/metadata hashes match; every mask-stage file hash still matches its receipt. Mask requests occupy the first three ledger entries and label requests the last three.

Recorded freeze time:08:37:34.298485UTC. Mask-stage completion:08:37:37.191500UTC. First label retrieval completion:08:37:49.879956UTC. The frozen `labels()` source verifies the mask-stage receipt and all saved mask-stage hashes before invoking the first label fetch. Together the code and current receipts support the documented guarded execution order.

The ledger records successful retrieval *completion*, not separate HTTP request-start timestamps. This review does not claim independent external timestamp attestation or derive a stronger request-start ordering solely from those completion timestamps.

## Selection and scientific scope

The independent enumeration began at the first128-aligned core with a complete64-pixel context margin and retained every core with at least1,024supervised pixels. It read only supervision for eligibility and compared labels only afterwards. No threshold on ink counts, class balance or apparent model response appears in selection. No eligibility changes appeared after labels.

This is intentionally an annotation-coverage-selected cohort; it is not unbiased sampling of the scroll surface. Overlapping contexts do not turn265cores into265independent physical samples. The source annotations' provenance, upstream exposure and unresolved cross-winding input-context overlap remain limitations. Correct intake does not certify ink, true negatives, physical registration or pristine held-out evaluation.

## CT request-plan review

Independently mapping every actual pixel coordinate of the265full256-square contexts to128-square storage chunks gives exactly536unique objects:231w035,125w039,180w044. The plan contains this exact set without duplicates or extras, with all28planes retained. Every URL and local path matches its native array metadata and chunk coordinates; the metadata/source cohort hashes match.

The precise planned total is245,891,072bytes (`536 × 28 × 128 × 128`), leaving22,544,384bytes under the256MiBcap. The first-read identity policy, full-object length, no-missing-zero-fill rule and current-version—not historical volume identity—scope are explicit in the CT protocol. The request plan does not depend on ink-class counts and preserves the complete mask-frozen cohort.

This verifies CT **planning**, not successful acquisition, CT support counts or geometry. CT downloading was delegated separately and may have been running during this review. The review did not fetch or interpret its CT data. Later CT-support verification should remain a distinct receipt.

`recheck.py` regenerates `RESULTS.json` offline using only retained annotation arrays, metadata and plans. Its full per-core IDs and chunk lists make the assertions inspectable.
