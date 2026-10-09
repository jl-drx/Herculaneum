# Phase 18 retained-data audit

A bounded CPU depth-representation pilot is feasible with retained data, but it can provide **development evidence only**. No presently ready retained reference is fresh confirmatory material for a method motivated by phases 16–17. Paris2 is now evaluation-informed development material and remains excluded, with every alias, from gradients, teacher-target construction and cutoff fitting.

This audit reads prior reports/plans and array headers only. It made no network requests, decoded no scientific payload, scored no arrays, imported no model and performed no training. `AUDIT_SOURCE_PINS.json` hashes 14 source documents and records 92 inherited artifact identities. All 92 files exist with matching retained sizes; 88 NPY headers were read. Their payload hashes are inherited from pinned prior plans, **not independently recomputed here**. No first-seventeenth artifact or repository-tracked file was changed.

## Concrete available roles

| Role | Parent and cases | Full source XYWH / support size | Limits |
|---|---|---|---|
| Training | PHerc0139 w016 rank01 and rank02 | [1792,4864,384,384] and [3584,5120,384,384]; each retained surface 21×384×384 uint8, 3,096,704 file bytes | Already adapted in prior studies; historical validation explicitly retired. One parent, not two independent specimens. |
| Original calibration | PHerc1667 w029; PHerc0814 old rank01 and rank07 | [6912,3200,256,256], [256,1664,384,384], [640,1536,384,384]; respective calibration negatives 9,378, 40,400, 14,644 | Reused historical development. PHerc0814 calibration and diagnostics share a parent. |
| Scroll development diagnostics | PHerc0814 fresh-rank01 and fresh-rank14 | [1152,1152,384,384] and [128,1152,384,384]; primary positive/negative 19,545/29,488 and 4,767/16,561 | “fresh” is a historical filename; both have been repeatedly evaluated. Both supplied validation masks are empty; scoring participation is supplied supervision, not V. |
| Optical development diagnostics | Paris1, Frag4 / PHercParis1Fr39, two contexts | Each 17×384×384 float32, 10,027,136 file bytes; primary positive/negative 3,818/87,626 and 15,252/62,599 | Already evaluated. Excluded from gradients, teachers and calibration across aliases. |
| Optical calibration | Paris1, Frag3 / PHercParis1Fr34, two contexts | Same 17×384×384 float32 shape; primary positive/negative 9,434/82,951 and 5,771/83,490 | Permitted calibration material; shares Paris1 with Frag4. Phase17 inspected both contexts. |
| Optical development diagnostics only | Paris2, Frag1 / PHercParis2Fr47 and Frag2 / PHercParis2Fr143, four contexts | Same 17×384×384 float32 shape; positive/negative 19,400/69,159; 12,633/70,681; 19,049/61,660; 20,248/66,318 | All four must remain development-only. Two fragments are one parent. No fitting, selection of hard/easy subregions, depth/window tuning or alignment adjustment. |

The training manual participation is `M = (S > 0) & (V == 0) & acquired`. The frozen schedule covers six eligible 128×128 tiles per training context, batch size two, 32 updates, with retained D4 transforms. Unique scheduled manual pixels are 25,967 and 28,734, total 54,701. These are support pixels, not independent observations. The second context's full-context manual count 31,199 must not be substituted for scheduled exposure. Retired V labels must not be reintroduced as manual training labels or advertised as held-out validation.

The historical seed43 pseudo map has its own role and threshold 0.5; it is not the seed42 preservation teacher. Prior scheduled pseudo support contains 22,730/4,551 negative/positive pixels in rank01 and 18,187/2,231 in rank02. Prior seed42 teacher logits cover the exact 32 normalized batches and are bound to their input fingerprints. Reuse is justified only when the candidate's teacher branch sees those exact inputs; new perturbations cannot silently reuse mismatched cached teacher rows. A manual-only candidate need not introduce seed43 pseudo supervision.

## A leakage-controlled development split

For one new representation, a concrete minimal split is:

1. **Gradient and any preservation-teacher inputs:** only the two retained PHerc0139 contexts, using the literal prior support and schedule. Freeze whole-parent/all-alias exclusions, not random pixels or overlapping tiles.
2. **If a new cutoff is needed:** use only PHerc1667-w029's 9,378 retained calibration negatives and the existing negative-only 95%-specificity rule. This is the historical single-region sensitivity rule, so promoting it to primary would require an explicit pre-run declaration; it is not the unchanged phase14 three-region primary rule. Alternatively use the literal old cutoff and call its transfer descriptive.
3. **Post-lock development validation:** both retained PHerc0814 diagnostic contexts and both Frag4 contexts. These parents differ from PHerc0139 and PHerc1667. All results remain reused development evidence because original training and local method-development histories are known or unresolved. Paris2 can be reported separately as fixed development stress cases; no result there makes the study confirmatory.

This split avoids introducing a *new* PHerc0814 calibration/validation parent overlap. It cannot erase earlier PHerc0814 calibration or upstream exposure. Retaining the three-case historical calibration rule is also possible for comparability, but then the shared PHerc0814 parent must be disclosed. For an optical-domain calibration study, Frag3-only calibration and Paris2-only diagnostics likewise separate current fit/evaluation parents; they remain evaluation-informed development and Frag4 cannot be claimed parent-independent from Frag3.

The minimum scientifically useful CPU question is whether a pre-frozen depth branch preserves its raw-control initialization, learns on the declared PHerc0139 objective and changes the fixed development outputs under a matched budget. It does not require a newly acquired holdout to answer those limited questions. It does require fresh parent material before any subsequent confirmatory generalization claim. No architecture, optimizer, cutoff or stopping decision is chosen by this audit.

## Depth and checkpoint contracts

Training and scroll calibration inputs were pooled from published source Z [13,97) in groups of four, float32 mean then nearest-even rounding to uint8, producing 21 planes at nominal 9.596 µm. Model layers [2,19) select 17 planes, corresponding to source [21,89); declared pooled label plane10 becomes model plane8. Keeping these exact 17 planes permits a depth-index/axial-difference branch without acquiring more CT. The index is not authenticated inward/outward anatomy. Already rounded source intensity cannot reconstruct a finer native representation.

The optical inputs instead use native uint16 planes [7,58), mean groups of three in XYZ, yielding 17 float32 planes at nominal 9.72 µm with plane32 at the center. Preserve these separate acquisition contracts and the established per-tile raw normalization. A difference in index units and a difference per micrometer are different candidate definitions; freeze whichever is intended. Phase17 provides no basis to fit a reversal, shift or replacement depth window.

Available checkpoint file identities:

| Checkpoint | Bytes | SHA-256 |
|---|---:|---|
| Unchanged seed42 initializer | 138,360,039 | `e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab` |
| Retained phase13 A_preserve | 138,351,655 | `f91cef62694c51eab240fe6501cc6f49d1a41db6fa24be3307a5e705900f5bdb` |
| Retained phase13 B_preserve | 138,351,655 | `3b28cc5c7d511142afc557599e74a65f46e7d1992c94db4dc9f3a3355a35cdcb` |

Seed42 is `scrollprize/ink_9um` revision `7109667e2607db1b90c37c8b09cb876ea7fe7bb1`; filename step075000 contains recorded step74999. All four original compared models list the same 29 curated representations spanning PHerc0139, PHerc1667, PHerc0814 and PHercParis4. “w016 excluded” means excluded from pseudo-label replacement, not excluded from original training. Exact historical pixel-loss exposure, context halos and upstream teacher/pretraining support remain unresolved. Supplied scroll labels have transferred/manual-plus-iterative-pseudo lineage, not established human-only optical truth.

## Fresh material already identified in retained records

No candidate is selected here from performance, and none is acquired.

| Candidate | Retained status | Required before confirmatory use |
|---|---|---|
| PHerc0009B | Tenth-stage metadata-only reservation, absent from four explicit 29-root configs; all aliases reserved. CT level2 109×6340×5130, nominal spacing 2.401/9.604/9.604 µm. | Parent/alias relationship, actual source-to-label offset application, annotation depth, S/V/mapping encodings, human/teacher lineage and historical exposure. Metadata says annotations were realigned against predictions; this is not independent optical truth. |
| PHerc0500P2 | Same metadata-only reservation; CT level2 118×6190×3470, nominal spacing 2.215/8.86/8.86 µm. | Same gaps; its 118-plane renderer needs its own contract. Retained lineage notes a source-name/catalog energy discrepancy. |
| PHerc51Cr4Fr8 / Frag6 | Fifteenth-stage deferred optical candidate, an additional parent group; labels/IR/mask and 65-TIFF listings exist. | Specific annotation provenance, render depth/spacing and all-parent upstream/local exposure audit. File count alone does not establish depth geometry. |
| PHerc1667Cr1Fr3 / Frag5 | Deferred source contract, but already exposed PHerc1667 parent. | Not fresh-parent confirmation, even if its exact fragment is new. |

The prior tenth-stage preference to audit PHerc0009B first is a provenance/workflow recommendation, not a metric-based selection. Neither it nor PHerc0500P2 has a frozen crop, depth center or pooling rule. Preserve both reservations. PHercParis4/w00 and native PHerc0139/w035 may support further domain diagnostics but occur in the original configured training universe. PHerc0800 has no authenticated labels and cannot supply accuracy validation or training negatives.

Primary source entry points are pinned in `AUDIT_SOURCE_PINS.json`: phase13 training/retention plans; phase14 calibration plan; tenth reference `LINEAGE.md` and reservation spec; seventh model-exposure ledger; fifteenth source/benchmark contracts; sixteenth comparison plan; seventeenth results and next-study draft. Source files remain immutable.
