# Eighth study data design — 8 October 2026

The retained PHerc0139 contexts provide enough public manual-reference support for a small controlled continuation pilot. Two fresh PHerc0814 contexts are feasible at the metadata level with the agreed 128-pixel context gap, using **supplied supervision membership for reference evaluation**. Actual fresh class counts remain subject to the frozen annotation gate. PHerc0814 is excluded in its entirety from our new adaptation and teacher-target generation. It was used historically by both initializer and teacher, so this reserve is held out from our new adaptation and tuning, not globally model-held-out.

## Retained adaptation support

Training support is exactly `supervision>0 & validation==0 & acquisition_observed`. It does not infer negatives from unannotated zero labels. Labels are public manual-reference annotations with transferred and iterative model-assisted history, not proven human-only or optical truth.

| Retained PHerc0139 context | Unique participating pixels | Positive | Negative | Eligible 128² tiles |
| --- | ---: | ---: | ---: | ---: |
| rank01, source origin [1792,4864] | 25,967 | 14,221 | 11,746 | 6 |
| rank02, source origin [3584,5120] | 31,199 | 6,775 | 24,424 | 6 |

Tiles use a fixed stride of 64 over the complete 384² crop. Eligibility requires at least 256 participating positives and 256 negatives; all 25 tile counts per context are retained in [data-proposal.json](data-proposal.json). Tiles overlap, so summed tile counts are not unique source-pixel counts. The training contract must account for repeated exposure separately from region/class loss weighting. These eligibility decisions used annotations only, without reading teacher scores. Pseudo-target support must still respect the fixed teacher's evaluated and finite mask; teacher agreement does not redefine manual labels.

The teacher proposal is retained seed43 predictions on the retained PHerc0139 contexts. This audit did not inspect those values, run inference, select a teacher cutoff, build targets or train a model. Root owns those separate frozen contracts.

## Fresh reserve and exclusions

The entire PHerc0814 specimen, all its representations and teacher targets are excluded from our new adaptation. Earlier PHerc0814 contexts are development evidence and are excluded from the fresh reserve:

- rank01: source XYWH **[256,1664,384,384]**.
- rank07: source XYWH **[640,1536,384,384]**.

Every proposed fresh full context must have `max(horizontal rectangle gap, vertical rectangle gap) >= 128` from each old context and each accepted fresh context. A touching boundary has gap zero. This conservative source-XY exclusion protects complete model context as well as primary pixels. It does not establish separation of every native-3D rendered sampling support or exclude nearby/colliding papyrus faces. Distinct PHerc0139 and PHerc0814 specimens establish physical separation between our adaptation and reserve. The two fresh reserve contexts still form **one** specimen group.

Retained complete inventories yield 35 gap-qualified annotation-proxy-supported 384² origins. None intersects a validation chunk with a size differing from the common 78-byte value. These are metadata proxies, not a newly decoded proof that every validation voxel is zero. Root consequently authorized **supervision-only evaluation participation** while preserving the actual validation arrays separately.

The fixed top 16 candidates rank by descending non-78-byte validation chunk count, then supervision count, then ink count, then ascending source Y/X. The first is **[1152,1152,384,384]**. If it passes, the first later metadata-ranked context satisfying the pairwise gap is rank06, **[512,1024,384,384]**. Actual selected ranks may differ only through the predefined annotation gate and gap exclusions. No source intensity or prediction informs candidate selection.

[label-selection-plan.json](label-selection-plan.json) is the executable contract. It traverses only the fixed 16 candidates and accepts the first two with at least 256 supplied-supervision positives and 256 supplied-supervision negatives in fixed local **[64,320)²**. Gap failures are recorded without annotation retrieval; class failures and unexamined candidates remain recorded. Exactly two adequate contexts are required before a complete reserve is claimed. There is no geometry/class relaxation or outside-candidate search. The earlier data-proposal records an unexecuted validation-first fallback proposal; the frozen executable plan explicitly supersedes that policy with root's supervision-only instruction.

## Source, masks and bounded acquisition

The source remains the verified public PHerc0814 segment `20260226000000-46527_2um_try2`, volume `20260309142202`:

[Public source metadata](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0814/segments/20260226000000-46527_2um_try2/surface-volumes/2.399um-0.22m-78keV-volume-20260309142202.zarr/.zattrs).

Read level 2, uint8 **[109,2130,3455]**, uncompressed C-order **[109,128,128]** chunks with `/` separators. Read published source Z **[13,97)** and pool consecutive groups of four with float32 means, nearest-even rounding and uint8 conversion. Prepared surface shape is **[21,384,384]**, nominal spacing **9.596 µm** in all axes; model planes are **[2,19)** and the annotation plane is 10. Increasing published source Z is retained without fitted reversal, offset or CT normal claim.

Current corrected annotation masks declare the same source, pooling and center channel. The source, annotation metadata and inventories are reused byte-for-byte from the seventh audit. Root's source execution compares attributes and the level-2 array description before and after retrieval, preserves each actual HTTP 200 source object and SHA-256, and stops on a missing/non-200 object. No zero fill, automatic retry or signed-query logging is introduced. The bounded HTTP helper is an unchanged copy of the independently reviewed prior implementation, SHA-256 `b731d7bdb9ec9677bf47641409054d70636f2725e56f4f6b21b414292bfffce7`. Its earlier documented limitations remain; this package does not claim a newly audited general transfer framework.

| Stage | Exact maximum/frozen payload | Shared response-body cap | HTTP-attempt cap |
| --- | ---: | ---: | ---: |
| Candidate annotations | 150 unique objects; 14,332 expected compressed bytes | 2 MiB | 900 |
| Two accepted source contexts | 18 chunks; 32,145,408 bytes | 40 MiB | 132 including metadata |
| Source metadata before/after | 7,346 expected bytes | Included above | Included above |

With five maximal 4,096-byte redirect bodies per source/metadata object, the successful source-read upper bound is **32,603,314 bytes**, below 40 MiB. Headers/TLS are outside these body budgets. Root first executes annotation selection, then generates and independently reviews/freezes the exact source plan before source retrieval. No network, new label/source acquisition, weights or inference occurred in this audit.

Each source provenance retains raw ink labels, supervision, validation, their memberships and the conditional current-loader `S & ~V` mask. `evaluation_participation='supervision_mask'` and `evaluation_participation_role='supplied_supervision'` name the evaluation domain explicitly. Its primary mask is **S ∩ actually acquired ∩ fixed 64-pixel interior**, later intersected with common finite/evaluated model support and the same class-count gate. Acquired zero values stay observed; `np.any(surface[2:19] != 0, axis=0)` remains a diagnostic occupancy stratum, not an automatic exclusion. No geometry-validity mask is claimed.

The assembler emits per-context provenance and `acquisition/fresh-descriptor.json` with canonical arrays, source/crop/depth/spacing, class counts and explicit participation. IDs use `pherc0814-46527-fresh-rankNN` to distinguish fresh contexts from seventh-study ranks. The parent worker consumes this descriptor only after acquisition and independent data reconstruction pass.

## Limits and receipts

Existing student/teacher configurations include PHerc0814. Complete historical pixel/loss exposure, teacher ancestry, annotation authorship and inter-face 3D independence remain unverified. An improvement here would concern agreement with public reference labels under a controlled adaptation experiment. It would not authenticate ink, establish native PHerc0800 transfer, decipher letters or demonstrate a novel method by itself.

Private noncommercial research is retained. Public CT defaults to CC-BY-NC 4.0; the reach of legacy annotation terms across current versions remains unresolved before any release. No terms acceptance, publication, provider action, paid GPU, R2 write or new checkpoint occurred in this audit. Prior scientific artifacts and tracked repository files were untouched.

Offline preparation verified 23 pinned input records, derived both-class counts and all 50 candidate training-tile counts, checked source-XY exclusions and parsed all new scripts. These checks did not execute acquisition or training. Executable plans/scripts/proposals are held unchanged; [SOURCE_RECEIPTS.json](SOURCE_RECEIPTS.json) records their final hashes. Any later correction must be additive and separately reviewed.
