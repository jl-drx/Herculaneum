# Plinius47: fixed-grid sensitivity confirmed; return to target preparation

The bounded CPU comparison completed at the same exposed PHerc343 ROI with all
four predetermined origins. The unchanged(0,0) baseline reproduced all262,144
saved seed42 plain pixels exactly. Three shifted grids then ran once. No model,
depth, threshold, region or phase was selected using the results.

| Offset against(0,0), row/column pixels | Pearson | MAE, uint8 levels | Top10% response Dice | Selected responses within8px, pooled |
| --- | ---: | ---: | ---: | ---: |
| (0,32) | 0.819505 | 9.404213 | 0.626129 | 0.780955 |
| (32,0) | 0.781892 | 9.210874 | 0.609504 | 0.805962 |
| (32,32) | 0.678020 | 11.220278 | 0.479684 | 0.651451 |

There are261,919 valid pixels. All six pair comparisons, all5/10/20% selections,
both directions of nearest-response distances, selected fractions and source/grid
phase bins are retained in ANALYSIS.json. Across all pairs correlation ranges
0.6514–0.8195 and MAE7.6783–12.2099. Top10% boundary ties increase actual selection
fractions to10.027–10.298%; no coordinate tie-breaking was applied. Differences
are calculated at unchanged source positions. Distances are not tracked feature
motion, physical ink error or independent biological evidence.

Independent addition of saved per-batch probability tiles reproduces all four
float32 weighted sums, weight sums and uint8 maps exactly. All18 pair/rank
bidirectional distance arrays agree between EDT and a separate KD-tree method.
Every frozen input/source hash was rechecked. The fixed-scale figure was inspected
in full: some high-response locations persist, while prominence, extent and
connectivity vary. Its difference scale clips at64 for display; full maxima up
to106 are retained numerically and were not hidden from the assessment.

## Interpretation and decision

This establishes operational sensitivity to the inference grid in this fixed
region, with depth, checkpoint, raw CT and runtime held fixed. It supports Sam's
recommendation to account for phase uncertainty. It does not identify whether
patchwise normalization, model context, blending or their combination dominates;
nor which response is correct. Reindexing half-stride grid-phase bins is an
algebraic shift, not independent evidence that features physically follow a grid.
No causal grid-locking claim is made from those bins.

The four-origin study is complete. Retain the author's current four-model recipe
and ownership merge as the baseline. Do not search more origins, average for a
better-looking image or reopen training on this exposed crop. For target work,
record phase-specific evidence and avoid treating unstable strong responses as
authenticated characters. A prospective change to inference needs its own
evaluation. The primary programme now returns to PHerc0800.

## Target preparation

Twelve retained provenance records were reviewed:10 surface acquisitions collapse
to three maximal exposed extents; two earlier low-resolution CT cube inspections
remain separately listed. A KD-tree screen compares samples every20 native
surface pixels from all five larger meshes against3,260,832 valid old surface
centres in the same CT coordinate frame. All source identities match the previous
cumulative seal or the earlier Git workspace snapshot.

| Mesh | Sampled valid positions | Within64 CT voxels of old surface centres | Largest separated component, sampled positions |
| --- | ---: | ---: | ---: |
| 20251028213516 | 5,783 | 87.98% | 271 |
| 20251028220042 | 5,377 | 91.52% | 245 |
| 20251028220955 | 5,169 | 82.22% | 237 |
| 20251028222030 | 7,899 | 76.69% | 485 |
| 20251028225813 | 6,210 | 83.64% | 623 |

All five mesh counts were independently recomputed, and30 predetermined
nearest-distance queries matched a brute-force scan of all3,260,832 prior centres
exactly. This verifies the calculation, not the physical interpretation.

Different mesh identifiers therefore cannot be treated as evidence of new tissue.
This centre-distance screen is not a physical-sheet match and does not certify
that a region is fresh. The largest sampled separated component is on25813, with
native bounds x600:1561,y20:501. It is a geometry lead only; its bounding rectangle
contains holes and pixels outside the component, and it lies near a render edge.
Full model context, slab/normal/interpolation support, and all older CT exposures
still require checking. No target field was admitted and no target ink map was
read or generated in this step.

Original geometry remains the baseline. The phase45 M7 medians(0.5,1,1voxel) did
not justify blanket snapping; local outliers, two tied profiles and neighboring
sheets remain the geometry factors. See NEXT_STUDY_DRAFT.md for a concrete
geometry/context admission task before target predictions.

## Execution record and limitations

The first runner's source inspection found a keyword-only API call error before
inference; additive v2 corrected it with the original script/GO retained. Its
first baseline then failed because the new recorder called an in-place sigmoid
on live logits, causing the main loop to apply sigmoid twice. That gate stopped
all shifted inference. The failed108 tile forwards, arrays and receipt remain.
V3 records from a detached clone, preserves the same scientific protocol and
repeats the exact baseline gate in a new directory. It passed with zero differing
pixels, then completed456 tile forwards across the four origins. Total new tile
forwards564; no training or Hecate execution. The phase46 result was unaffected.

An initial target metadata check stopped because some early provenance files are
absent from the cumulative seal. The additive v2 checks those files against their
existing private Git workspace manifest; no identity requirement was waived.
No target pixels were inferred. Logs retain both attempts. Matplotlib used a
temporary font cache after a nonfatal home-directory permission warning; later
commands use writable cache locations.

GPU spend£0. No paid resources, new CT downloads, R2 upload, external messages,
new reading or translation. All maps, tiles, failures and review records are
preserved; selected source/reports and a separate replay bundle are saved privately.
