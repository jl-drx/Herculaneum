# Reference panel source review — 8 October 2026

The primary proposal uses the existing PHerc1667-w029 result for development and annotation-selected contexts from **PHerc0139-w016** and **PHerc0814-46527** reserved from our subsequent tuning. These are different physical scroll specimens. They are not untouched model test cases: all four selected checkpoints include all three cases in their configuration, and the public masks were used for online validation. The appropriate result is agreement with public reference annotations across different specimens, with documented limits on historical training and teacher exposure.

The acquisition scripts, candidate order, public metadata and object inventories are now held unchanged. No new label chunks, source scan chunks, model weights or predictions were read by this source audit. Root owns the separately reviewed acquisition and benchmark execution.

## Available references and physical identities

The retained [official curated dataset README](https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/README.md), plaintext SHA-256 `c9a88ff970594fcef3788eb588bb75dbbd0e6436f4812f89074e26c0d9fef7a5`, documents 29 representations:

| Specimen | Pooled fine-scan annotation aliases | Native-family references |
| --- | --- | --- |
| PHerc0139 | w016, w017, w028, w029, w035, w039, w040, w041, w043 | w035, w039, w040, w041, w044 |
| PHerc1667 | w013, w018, w023, w028, w029, w031 | None in this package |
| PHercParis4 | w00, w01, w02, w03, w05, w06, w07, w09 | None in this package |
| PHerc0814 | 46527 | None in this package |

Only **pherc0139-w016, pherc1667-w029 and pherc0814-46527** have released validation masks. Other cases provide ink labels and supervision; zero ink labels outside a valid annotation mask do not establish negative truth. Paris4 and native PHerc0139 cases can therefore supply separately named training-reference or acquisition-transfer diagnostics, but not a new held-out accuracy claim.

Annotation aliases do not match public segment numbering automatically. PHerc0139-w016 is public segment **20250108000004-w029_2025010827**; it is neither PHerc1667-w029 nor the PHerc0139 annotation alias w029. The complete [candidate-panel.json](candidate-panel.json) supplies all 29 exact source URLs, scan volume identifiers, public segment IDs and representation families. Physical groups should use specimen and resolved public segment identity rather than annotation w-number alone.

Native w035, w039, w040 and w041 reuse public segment IDs of their fine-scan counterparts. Native w044 corresponds to aligned **pherc0139-w028**, the known cross-representation physical alias. These representations must stay in the same split group. A shared segment identity is a pairing lead; it does not prove pixel registration between independently rendered canvases.

## Verified primary contracts

| Field | PHerc0139-w016 | PHerc0814-46527 |
| --- | --- | --- |
| Public segment | `20250108000004-w029_2025010827` | `20260226000000-46527_2um_try2` |
| Source volume | `20260102150214` | `20260309142202` |
| Acquired source level | 2 | 2 |
| Source ZYX shape | 109 × 7,020 × 7,220 | 109 × 2,130 × 3,455 |
| Source dtype / chunk | uint8 / 109 × 128 × 128 | Same |
| Compression / order / separator | None / C / `/` | Same |
| Annotation ZYX shape | 21 × 7,020 × 7,220 | 21 × 2,130 × 3,455 |
| Annotation chunk / encoding | 21 × 128 × 128 / uint8, Blosc Zstd bitshuffle | Same |
| Declared annotation plane | 10 | 10 |
| Declared source Z range | [13,97) | [13,97) |
| Prepared spacing ZYX | 9.596 µm in each axis | Same |
| Complete inventory per annotation role | 3,135 chunks | 459 chunks |
| First metadata-ranked candidate XYWH | [1792,4864,384,384] | [256,1664,384,384] |

Exact public source roots:

- [PHerc0139-w016 source](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20250108000004-w029_2025010827/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr/.zattrs)
- [PHerc0814 source](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0814/segments/20260226000000-46527_2um_try2/surface-volumes/2.399um-0.22m-78keV-volume-20260309142202.zarr/.zattrs)
- [PHerc0139-w016 corrected annotation attributes](https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0139-w016/pherc0139-w016_validation_mask.zarr/.zattrs)
- [PHerc0814 corrected annotation attributes](https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/pherc0814-46527_validation_mask.zarr/.zattrs)

All three annotation attribute bodies agree within each case and name the same exact public source. All three annotation array descriptions agree within each case. Complete inventories were checked against every expected chunk coordinate; objects are pinned by recorded byte length and Xet ETag, with plaintext SHA-256 recorded when acquired. An ETag is not assumed to be plaintext SHA-256. Hugging Face buckets are mutable; these retained metadata hashes identify the observed version, rather than claiming a Git revision for a bucket.

The source metadata identifies 2.399 µm, 78 keV scan lineage and mean XY downsampling. Level 2 has spacing [2.399,9.596,9.596] µm. Preparation reads the published level-2 pixels and averages four consecutive source planes from [13,97), using float32 means, nearest-even rounding and uint8 conversion. The 21-plane result uses model slice [2,19), 17 planes. Its label plane 10 averages source [53,57), mean source index 54.5. This reproduces the retained [official preprocessing contract](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/preprocessing/prepare_9um_isotropic_input.py). No additional XY resampling, fitted Z reversal or CT normal-direction claim is introduced.

## Annotation lineage and exposure limits

Current attributes declare `annotation_source_results=results-fixed`, rebuilt aligned-public-21slice-v2 inputs, a scratch-path source manifest and relaxed mapping validity with strict-support neighborhood gating and cleanup. The complete historical alignment manifest is not published in these attributes. The underlying scroll labels combine transferred manual strokes and iterative model-assisted refinement; per-pixel human-only provenance is unverified. They are public reference annotations, not optical ink ground truth.

Current official loader semantics at [Villa e0bbb8b4](https://github.com/ScrollPrize/villa/tree/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection) remove validation membership from supervision for ordinary flat training patches and use validation as the override for evaluation patches. Thus raw supervision membership is not proof of actual training loss exposure. Retain S, V and labels separately; report S∩V. The derived current-loader mask S∩¬V is conditional evidence, not a reconstruction of every historical run. Input context can expose neighboring validation CT without exposing its labels to loss.

Root's [model-exposure-ledger.json](../model-exposure-ledger.json), SHA-256 **7773e56feb2add2eeb98ce06e63621c2ca47818eb54f5d5828c97893631f7320**, records actual embedded-config membership for all four arms. Baseline/control use manual-reference roots. Russo ex016 uses the manual-reference root for w016 and a pseudo-label replacement for PHerc0814 and PHerc1667-w029. All 29 representations remain included. Exact historical trainer revisions, teacher training support and complete loss-exposure masks remain unknown.

The [Russo correction report at dd5bc8fb](https://github.com/DomRusso2/ink9um-dense/blob/dd5bc8fb300c810c6ec042cb7e068be78829d086/results/label_version_diff.json) reports 2,924 w016 validation pixels overlapping the author's all-seven pseudo-label training. **That all-seven model is not one of this panel's four arms.** The field in this plan names that scope explicitly; it does not mark all four checkpoints as proven contaminated. The report says PHerc0814 masks were unchanged and its measured all-seven overlap was zero. Neither claim substitutes for a historical exposure audit of the four actual models.

## Fixed selection and acquisition bounds

The frozen [label-selection-plan.proposal.json](label-selection-plan.proposal.json) ranks exactly 16 candidates per case before labels or intensities: descending potential validation-chunk count, then potential ink-chunk count, then ascending Y/X. Inventory size unequal to the common 78-byte value is only a ranking proxy. Actual decoded masks determine the class gate.

For each case, accept the first candidate with at least 256 validation positives and 256 validation negatives in fixed local [64,320)². Then accept the next adequate candidate whose **entire 384 × 384 context** does not intersect the first. Record class failures, overlap exclusions and unexamined fixed candidates. Read no candidate outside the fixed set. Keep one case if only one adequate crop exists; require at least one accepted crop in each specimen before declaring a complete panel. Source-XY separation does not establish native-3D sampling independence within a specimen; different specimens establish physical separation across the two cases and development w029.

The union of candidates needs at most **276 unique annotation objects / 30,473 expected compressed payload bytes**. The shared annotation budget is **2 MiB response bodies / 1,656 HTTP attempts**, at most six requests per object including redirects. No automatic retries, missing-object fill or signed-query logging. Encoding {0,1} and center-plane-only annotations are asserted against actual decoded chunks; a mismatch stops selection.

Each accepted context needs nine uncompressed source chunks, **16,072,704 bytes = 15.328125 MiB**. Four contexts need at most **36 chunks / 64,290,816 bytes = 61.3125 MiB**. Fresh source attributes and level-2 array descriptions are compared to pins before and after acquisition. Eight metadata responses add **14,714 expected bytes**. Even five maximal 4,096-byte redirect bodies per object produce a successful-read upper bound of **65,206,650 body bytes**, below **64 MiB**. The generated source plan records the actual crop count and exact objects, with up to **264 HTTP attempts** at four crops. A non-200 source chunk stops; zero-valued acquired voxels remain observed data.

The separate [prepare_source_plan.py](prepare_source_plan.py) runs offline after successful label selection. Root then reviews and freezes that generated plan before [assemble_panel_sources.py](assemble_panel_sources.py) runs. Per-context provenance retains the same six canonical array roles as the sixth experiment, exact source/crop/pooling lineage, separate masks and source hashes. Acquisition observation is not geometry validity. Occupancy is `np.any(surface[2:19] != 0, axis=0)`, a diagnostic stratum; observed zero profiles stay in primary support. Benchmark common finite/evaluated support and class-count gates remain a later check.

## Native secondary lead and terms

The official native PHerc0139 package is a useful future transfer diagnostic. The native-specific audit verifies w035 on the separate 113 keV coarse-scan lineage: its delivered 9.362 µm representation derives from 4.681 µm binmean2 scan processing, not fine 2.399 µm / 78 keV downsampling. Native w035 is 28 × 5,820 × 5,240, whereas the fine source at level 2 is 109 × 5,660 × 5,100. No measured transform or registration residual has been established between them. Native labels have no released validation mask and contain transfer/cleanup provenance rather than proven human-only labels. **Do not acquire or score a paired native/fine pixel benchmark until those mapping and annotation-domain questions are resolved.** This does not delay the primary pooled panel.

Public CT terms default to CC-BY-NC 4.0. Retained curated documentation links legacy label terms; their exact reach across modern annotation versions remains unresolved. This is private noncommercial research, with no redistribution or public reading planned. Code licensing does not relicense source labels. No terms acceptance, submission, outreach, account action, installation or paid resource action occurred in this audit.

## Research receipts

New metadata research ended at **43 HTTP requests / 3,353,420 actual response-body bytes**, including explicitly followed HTTPS redirect bodies, under **80 requests / 8 MiB**. There were no automatic redirects. Signed redirect query strings and Location headers are omitted from retained ledgers. Headers/TLS are outside the response-body budget. The native-specific research has its own separately retained receipt and is not silently included in this number.

All 40 pinned selection-input records were checked offline for SHA-256 and length, every fixed bbox was checked against its array extent, and all 27 annotation keys per candidate match the complete inventories. Syntax parsing succeeded for all new scripts. These checks did not execute acquisition, decode new labels, read scans or run inference. Root retains the code/plan hashes and owns execution approval; no claim that unexecuted acquisition or benchmark behavior has passed is made here.
