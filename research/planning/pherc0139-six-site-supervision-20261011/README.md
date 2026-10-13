# Six-site teacher-supervision sensitivity

**Completed: the remaining CT alignment uncertainty changes the teacher targets.
Soft scores do not remove the problem, and uncertainty averaging is numerically
almost the same as matched stationary smoothing in these regions.** This is a
supporting diagnostic of target construction, not a learned-model result.

Protocol and implementation were frozen and pushed in private commit `96ec2f6`
before reading these target crops. All six scorable sites from the eight-site CT
pilot were retained. Their four fixed checking patches supplied 657 distinct
patch-specific CT alternatives, evaluated at eight depths with two operators:
384 cases, with 24 primary plane 14 patches per operator. The two previous
source-support failures remain failed and were not replaced. All six strict
CT-registration screens still fail their original criteria.

## What the data show

All six expanded teacher-input footprints pass geometry, source extent, all 332
reserved native-context exclusions and the conditional fine-reference envelopes.
This clears bounded target inspection, not paired-training or accuracy admission.
The 15 conditional TIFF image requests transferred **1,752,709 bytes** (1.672MiB),
plus 18,953 structural metadata bytes. No new CT, inference or GPU was needed.

Teacher scores are nonzero at all 13,824 native-eligible primary output pixels.
That is a coverage convention, not evidence of ink: **16 of 24 patches have no
above-threshold teacher pixels**, and six more have too few common negatives.
Only two meet the predeclared ≥100 positive-union and ≥100 common-negative counts.

| Primary patch, pooled operator | Alternatives | Positive union / common negative | Variable positive fraction | Worst alternative soft MAE on positive union |
|---|---:|---:|---:|---:|
| w040-x6720-y4672, check5 | 27 | 356 /220 | **53.65%** | **0.05896** (15.03/255) |
| w041-x7488-y3136, check5 | 7 | 435 /141 | **12.64%** | **0.06437** (16.41/255) |

Both cross the frozen hard-target variation gate of 5% and soft-score MAE gate
of 0.05. The latter is the largest **single alternative's mean error** against
the fitted target over the whole finite-set positive union; it is not mean
per-pixel worst error, teacher error against truth, or a calibrated uncertainty.
Even the closest structural site, w041-x7488-y3136, is sensitive on its informative
checking patch. High CT NCC did not guarantee interchangeable supervision.

Across all 24 primary patches, six have hard-target sensitivity and three have
soft-score sensitivity; four and one of these, respectively, lack the required
class balance. No patch has teacher-nonzero-support sensitivity. Repeating every
depths 10–17 gives the same classification counts at every depth. The unpooled
operator control preserves those conclusions. Depths are repeated measurements
of the same regions, not independent replicates or verified ink faces.

The mean target over CT alternatives closely matches the frozen stationary
displacement-kernel control. With the primary operator at plane 14, the largest
patch MAE across common supervision is **0.01146 uint8 levels**, and the largest
individual-pixel difference is 0.04511 levels. Across all eight depths the largest
patch MAE is 0.04071 levels. Sampling only at the mean coordinate differs more
(up to 1.8382 levels of patch MAE at plane 14). The averaging effect here is
explained by ordinary smoothing to high numerical precision. A future method
must beat a matched smoothing baseline; this operation alone has no demonstrated
distinct benefit.

## Verification and interpretation

Fourteen numerical fixtures passed before acquisition. A second implementation
reconstructed the six crops from the 15 saved compressed source tiles, selected
all alternatives from the saved CT landscapes, independently sampled **6,054,912
alternative target values**, and recomputed all 384 case metrics, controls and
aggregate results. Maximum transfer difference was 5.69e-14 uint8 levels; maximum
control difference was 2.48e-11. No threshold decisions or counts changed under
alternate arithmetic. This was different code in the same runtime, not a
separate human/team replication. See `RESULT_VERIFICATION.json`.

`VISUAL_REVIEW.md` records inspection of all 24 fixed primary patches. The packet
includes both operators' fixed-scale panels, all intermediate targets and all
depth results. No panel or crop was selected for looking promising.

These are current canonical v42 teacher scores at declared fine-raster
coordinates. They are not sparse manual annotations, exact historical merged
targets, independent ground truth, authenticated letters, or evidence that the
teacher's training excluded our reference annotations. w040/w041 manual overrides
were not reconstructed; w043 has no native annotation release in this catalog.
No pixel AP/AUC, ink accuracy, learning improvement or new physical depth is
claimed. See `PROVENANCE.json`, `PROTOCOL.md` and `NEXT_DECISION.md`.

The native-learning study continues unchanged. Its held-out predictions remain
sealed until all nine endpoints are complete. This diagnostic does not alter its
training data, stopping rules or evaluation.

## Portable evidence

`PORTABILITY.md` gives the private selective LFS download and self-contained CPU
replay. `BUNDLE.json` binds every archived file by length and SHA256. The frozen
protocol/request plan is unchanged. Ordinary Git contains the report and code;
the bounded private LFS archive carries binary inputs and outputs. No new PR,
Actions run or spare-VM task was created.
