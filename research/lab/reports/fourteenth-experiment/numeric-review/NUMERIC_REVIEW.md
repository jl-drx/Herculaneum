# Independent calibration-only verification

**PASS.** A single execution of the pre-frozen independent NumPy checker verified the completed calibration run and analysis. No failure or retry occurred. The [verification receipt](completed-review/independent-verification.json) is 212,692 bytes, SHA256 `a3da13aae862380353528a2c8e57fa34411e5780b126678b8d508bce8ce8399b`.

The checker rebuilt the original three development masks from public validation, acquisition support and the 64-pixel margin, preserving exactly **9,378 / 40,400 / 14,644 negatives**. It verified all nine regional selections: seed42 and both new states across the three cases. Unique-score multiplicities independently recover the crossing negative-score tie, exact allowable false-positive counts and next float32 cutoff. All primary/secondary cutoff bits match their locked artifacts; the baseline artifact remains literal and unchanged.

| Role | Primary cutoff | Secondary cutoff | Exact float32 bits |
| --- | ---: | ---: | ---: |
| frozen_seed42 | 0.7558947205543518 | 0.7558947205543518 | 1061257809 |
| A_preserve | 0.7450202703475952 | 0.7450202703475952 | 1061075366 |
| B_preserve | 0.7422624826431274 | 0.7422624826431274 | 1061029098 |

For these retained outputs, w029 determines the primary maximum for every role, so the predeclared secondary equals primary. No diagnostic outcome selected either cutoff. The original baseline lock and new lock state/map lineage passed independently. The frozen source and completed receipts support the lock-before-diagnostic-analysis ordering; this ordering does not establish prospective validation because the diagnostic evidence was already known.

All **six new maps, 118 raw tile logits and 118 normalized input hashes** passed. Mathematical float64 sigmoid/Hann calculations, rounded into FP32 accumulation, reproduce saved scores with maximum absolute error **2.980232238769531×10⁻⁷**, below the unchanged 2×10⁻⁶ limit. Evaluated masks match exactly. This is numerical tolerance rather than bitwise score identity across libraries. Normalized input hashes reproduce exactly from the three original pooled uint8 surfaces, with both states receiving identical inputs.

The four diagnostic masks and exact positive/negative counts were rebuilt from their original supplied-supervision or known-surface contracts. Every diagnostic score/coverage map was reused literally from thirteenth. AUC/AP agree with thirteenth for all four contexts and three roles. The checker also verified all 36 primary/secondary/common-cutoff confusion tables, operating rates, per-context differences, equal-context specimen macros, integer specificity checks, reduced rational sensitivity comparisons and sensitivity changes versus the common cutoff. It made **511 scalar comparisons**, with maximum error **1.1102230246251565×10⁻¹⁶**; counts, rational numerators/denominators, cutoff bits and screen booleans were exact.

| Arm | PHerc0814 recovery screen | Frag4 recovery screen |
| --- | --- | --- |
| A_preserve | Fail: mean sensitivity below baseline | Fail: mean sensitivity and context1 loss allowance |
| B_preserve | Pass | Fail: mean sensitivity and context1 loss allowance |

A's PHerc0814 mean sensitivity difference is exactly `-1749/20704670`; the frozen exact screen remains failed despite the small magnitude. B's corresponding difference is `8983/4140934`. Frag4 mean differences are `-30/1909` for A and `-139/7636` for B. Every candidate context meets the specificity requirement. The separately reported baseline floor correctly shows that frozen seed42 misses 95% specificity in `frag4-context2`. Improved sensitivity versus each state's common cutoff does not redefine either failed primary screen. Root owns the scientific interpretation and subsequent decision.

The [source freeze](code-freeze.json), [source review](SOURCE_REVIEW.json) and preparation fixtures were recorded before scientific execution. Preparation passed 27 numerical, three canonical-mask and 48 separate blend/normalization cases. The actual review bound the exact root-provided plan SHA256 `ca36e87f03ca3ac91caf8c3adc81c821d0fa67b2e3a2c711ece62a9fac190bb2`, locks-ready SHA256 `bdb9219ac818f2ac8ef76339b9595b689b679dd09be566e24c8395bc8f9c3ec2` and metrics SHA256 `7c9e12f7bbbe683fdaa1fff467b919768c67500255f6501fc63c593e8d9a58f0`. All **315 source/input pins** remained unchanged. Four checkpoint/weight records were handled as metadata and excluded from byte reads.

Independent claims concern saved numeric outputs, masks, normalized input identities, blending, thresholds, confusion counts, ranking integrity and rational screens. Semantic model state, strict schema/storage and absent gradients remain worker-reported checks: this reviewer checked their pinned consistency without loading weights or executing a model. Original calibration baselines lack archived normalized tile hashes; their linkage remains bounded by literal source/depth/normalizer/adapter pins and tile placement. Reused development and diagnostic contexts provide no independent performance, significance, model promotion or authenticated reading claim.

Reviewer operations were retained NumPy/JSON reads and calculations only: **zero Torch imports, model calls, checkpoint reads, network calls, new predictions or training**. The child blend task stopped after source/synthetic preparation and performed no real-output review; its helper ran within the single authorized checker pass. Earlier and author evidence remained unchanged. The final [review receipt](REVIEW_RECEIPT.json) and [manifest](MANIFEST.json) seal the package; no further edits are planned.
