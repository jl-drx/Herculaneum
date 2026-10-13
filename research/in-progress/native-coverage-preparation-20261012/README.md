# Matched native coverage: source and training preparation

12 October 2026. The controlled training pair now has a concrete source proposal,
a geometry-based support reservation, and a passing real-model accumulation and
save/reload check. **The full training population is not yet admitted and no GPU
has been rented.** These are preparation results, not a learning improvement.

## The contrast we can actually isolate

The [catalogue audit](COHORT_CANDIDATES.json) binds all 56 native-v2 representations
in the released Reader configuration. They are already in Reader's published
training list; they are not 56 independent subjects or newly unseen examples.
The five proposed original-stream sources are PHerc0139's direct native-frame
annotations on w035, w039, w040, w041 and w044. Their annotation ancestry does
not make them independently authenticated ink.

| Added-source option | Native observation | Teacher observation | Decision for this pair |
|---|---|---|---|
| PHerc0814, 8 segments | 9.362 µm, 113 keV | 2.399 µm, 78 keV; model 20260417190342 | Same acquisition/teacher contract as the PHerc0139 native-v2 sources |
| PHerc0500P2, 18 | 9.362 µm, 113 keV | 2.215 µm, 111 keV | Separate acquisition/data-package contrast |
| PHerc0343P, 6 | 8.64 µm, 116 keV | 2.215 µm, 111 keV | Separate acquisition/data-package contrast |
| PHerc0172, 4 | 7.91 µm, 53 keV | Different TimeSformer predictions | Separate acquisition/data-package contrast |

The other 20 native-v2 representations are PHerc0139. Their teacher targets are
distinct from the proposed original stream's native annotations.

**Both arms use the same eight PHerc0814 segments and the same per-segment
quotas in their additional slots.** A repeats restricted regional support; B
uses broader support. Original PHerc0139 draws, augmentation keys and loss
coefficient remain identical. Additional slots never repeat the original
PHerc0139 stream, which would otherwise change its total exposure. This first
pair tests regional coverage within a fixed parent mixture. It does not isolate
cross-parent diversity, add a new teacher, or constitute a novel model method.
The broader programme and acquisition-diversity experiments remain open.

Retain dense-native as a mandatory frozen comparator and Hecate as the different
architecture control, following the completed comparison. Dense-native was used
for the mechanics check here. Freeze the actual initializer with the final
cohort; do not keep searching the exposed comparison panel for one.

## Actual source contracts and fixed smoke locations

Eight TIFF header checks consumed 115,403 bytes without reading image payloads;
the complete compressed maps total 641,596,866 bytes. Native CT is uint8,
28 planes, uncompressed 28×128×128 chunks. Fully materialising all eight native
canvases would take 26,547,315,200 bytes, so use verified selective chunks.

Before acquiring image bodies, one 128-square crop was fixed at each native
canvas centre, aligned to the source chunk grid. Seven were readable. The
20250926165636 centre returned HTTP 404 and stays in the results as unsupported;
no substitute crop or zero-filled input was used. Its other geometry-qualified
regions are not rejected merely because this centre is absent.

The [source qualification](SOURCE_QUALIFICATION.json) accounts for all eight
cases: 23 bounded body requests, **5,095,839 bytes**, seven decoded crops,
**82,706 supervised pixels and 18,454 teacher-positive pixels**. Explicit zlib
and tifffile decoding agree for every needed teacher tile. Global INTER_AREA
resize phase is preserved; three sparse/full known-image controls pass and an
independent area integral differs by at most 0.499972 uint8 levels, consistent
with rounding. A crop-local resize with a different phase is not substituted.

The published hard-target recipe is teacher ≥128 positive, teacher ≤64 negative,
with native centre-plane CT >0 required for supervision. Intermediate teacher
values and invalid CT remain ignored. This is **weak teacher supervision**:
reproducing the resize does not verify fine/native physical registration. The
six earlier failed transfer sites remain unadmitted as precise labels.

All eight locations were viewed in [the fixed 0–255 panel](SOURCE_REVIEW.png).
No fitted contrast, letter reading, or reference accuracy follows. These smoke
crops qualify bounded decoding; they do not stand in for the full corpus.

## Real model loss and checkpoint mechanics

The [frozen check](TRAINING_GO.json) used four actual native PHerc0139 patches,
chosen for negative/two-class mask variation, and the full **34,546,498-parameter
dense-native model**. It did not train on PHerc0814 or evaluate held-out maps.

The objective is a fixed weighted sum of independently normalized streams.
Within each stream, BCE divides by total supervised pixels while per-image
Dice divides by image count. Accumulation must preserve both denominators.
The retained BCE smoothing is 0.5 and Dice weight is 0.25. The demonstration
stream coefficient 0.5 is not yet the frozen study hyperparameter.

[Qualification passed](TRAINING_QUALIFICATION.json):

- The separate loss and logit gradients match the retained objective. Naively
  averaging unequal-mask microbatch losses fails the deliberate negative control.
- Full batch versus four single-image microbatches gives relative gradient L2
  difference **2.58×10⁻⁵**, within the predeclared 10⁻⁴ tolerance.
- All 3,200 original-slot checks retain their draws and augmentation keys when
  the added pool changes. The model uses sample-wise InstanceNorm; actual
  BatchNorm/dropout/running-stat couplings are rejected by this check.
- After two SGD updates, reloading the first checkpoint and replaying the
  second update reproduces model values and optimizer momentum **bitwise**.

There were 17 model forwards across the gradient comparisons, two updates and
one update replay. This is a CPU save/reload test, not a process-crash test,
accuracy study, CUDA/AMP benchmark or test of adequate training exposure.
The released checkpoint was unchanged. The complete PHerc0814 loader, final
source quotas, production augmentation and GPU delivery still need qualification.

## Reserved support and its physical limits

[Version 2](support-reservation-v2/RULE.json) fixes the geometry rule before
population-wide teacher decoding: 128-square patches at stride 96, valid stored
XYZ throughout the footprint plus two mesh nodes of halo, and whole-patch
containment within 512-square blocks. Both arms use that same eligibility rule.
A keeps a deterministic quarter of the blocks by a fixed segment/block hash;
B includes every eligible block. No ink values choose these blocks.

The [reservation](support-reservation-v2/RESULTS.json) contains **2,886 A patches
and 11,887 B patches**, with all eight segments represented. Patch count is
4.12× larger in B, but the union of padded XYZ context boxes occupies only
**1.60×** as many 64-voxel cells and **1.46×** as many 128-voxel cells. These
are coarse input-neighbourhood proxies, not unique sheet area. The 866,898
valid stored mesh vertices contain no exact duplicated XYZ records, but six
segment pairs have rounded-voxel coincidences and all bounding boxes overlap.
This is not a certificate of independent physical tissue.

A [separate arithmetic implementation](SUPPORT_VERIFICATION.json) reproduces
all candidate membership, padded boxes and union counts using integer coordinate
conversion, summed-area masks and dense occupancy arrays. It does not create
independent physical annotation. Eight smoke locations had already been viewed;
this is development training material, not an untouched evaluation reserve.

## Preserved failures and amendments

The first source preparation stopped on the missing central chunk before a GO
record or image download. It was revised to account for the unsupported case
without replacement. The first training attempt stopped before any forward
because the retained loss constructor required `patch_overlap`; the corrected
configuration retained the same fixture, objective, budget and tolerances.

Support version 1 applied the within-block rule only to A. Version 2 applies it
to both arms, removing that eligibility confound before population labels or
training. The hash, geometry rule and original outputs remain unchanged and
preserved; this was not a reroll to obtain a preferred result.

## Next executable step and decision boundary

1. On this frozen reservation, make an exact bounded request plan and check
   native source existence/validity and teacher supervision across the population.
   Keep all failed cases, use identical eligibility in both arms, and report
   per-segment positive/negative/unknown exposure and physical-context overlap.
   Do not silently replace missing cases or redraw the narrow blocks.
2. Admit the original five native-annotation sources under the same explicit
   mask and exposure policy. Freeze schedules, quotas, loss coefficients,
   initializer, effective batch, full optimization budget, development endpoints
   and checkpoint rules. A class/support mismatch remains a data-package
   limitation unless a declared control separates it; do not fit an unreported
   balancing rule after observing model performance.
3. Then present a concrete capped GPU timing/training proposal. A 20,000-update
   pair is a candidate, not a runtime promise. Pin CUDA, artifact delivery,
   current rates and an independent shutdown; obtain approval before renting.
   The total first-pilot cap remains £10 including retries and overheads.

The pilot's practical improvement/retention/negative-control gates remain in
[the matched design](../frontier-intake-20261012/models/MATCHED_COVERAGE.md).
Do not lower them because this preparation passed. Insufficient supervision
or an inadequate physical contrast makes this particular pair uninformative;
it does not close broader native learning or justify more tiny CPU continuations.
Final improvement claims require fresh physically separated, independently
supported evaluation. PHerc0175 A/B and PHerc1203 stay unopened; PHerc0841 and
the whole PHerc0009 family stay excluded from new training. Completed native0139
fits and common-support comparison outputs remain immutable.

The parallel [CT corridor follow-up](../geometry-corridor-preparation-20261012/README.md)
has two shorter provisional paths, not a qualified surface-recovery benchmark.
See [PORTABILITY.md](PORTABILITY.md) for the private input/evidence bundle.
