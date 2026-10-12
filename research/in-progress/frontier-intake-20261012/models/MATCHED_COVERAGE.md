# Native coverage comparison: implementation specification

This is the next training design, not a launch or a frozen data cohort. It is
independent of whether the current small native0139 model reaches its gate.
The existing native study and its endpoint selection remain unchanged.

## Question and strong controls

Does adding broader native training support improve cross-parent annotation
agreement and supported-negative behaviour, when original-source exposure,
architecture, initialization and compute are controlled?

Reader v2 is strictly executable in the retained runtime. The
[completed common-support comparison](../../native-common-comparison-20261012/completion/README.md)
puts dense-native highest on equal-parent AP, with Hecate leading PHerc0841 and
trailing on PHerc0009B. Reader leads neither parent; released seed variation is
large and is not a controlled noise-floor estimate. Retain dense-native as a
mandatory frozen comparator and Hecate as an architectural control. Reader's
training recipe remains useful without being the automatic initializer. Choose
and freeze one initializer with the actual training cohort. The fixed five-model
mean did not beat dense-native's aggregate; do not fit ensemble weights on this
panel, and account for all constituent compute in any future ensemble comparison.
There is no general best-model or fresh confirmation claim. Respect each model's
actual sampling contract:
the hybrid models use 17×128×128 inputs and robust-MAD normalization; Hecate uses
16×64×64 inputs sampled at 9.6 µm in XYZ and divides by 255. Retain full source
depth so fixed model windows can be reproduced; do not reopen the completed
28-plane native study or silently force every model to the same incompatible tensor.

## Paired experiment

Use one released hybrid architecture, a single pinned initializer, one objective
and the same effective batch and schedule. A 20,000-update continuation is a pilot
candidate, not a CPU run or an estimate of achievable GPU training time. Reader's
published full recipe is 40,000 at batch 64; count actual patch exposures.

For each step, freeze two lists of batch slots:

- **Original-source slots:** exactly the same physical patch IDs, inputs, label
  support, weights and augmentation seeds in both arms. Their cumulative draws
  are equal; adding a crop must not halve the original-source exposure again.
- **Additional slots:** arm A repeats a fixed restricted development corpus;
  arm B draws broader native regions. Slot count, augmentation policy and target
  provenance tier are matched. Compare unique physical support and cumulative
  positive/negative/unknown supervised exposure explicitly.

The old and additional slot counts are frozen only once the admitted corpus is
known. They are not silently inherited from Reader's eight-source quotas. The
same additional support must not receive more manual labels only in one arm.
If teacher family, annotation policy, intensity processing or acquisition changes
cannot be matched, name the intervention as the resulting data package and add
the needed control before claiming a pure physical-coverage mechanism.

Generate patch schedules independently from optimizer RNG; preserve old-source
draw and augmentation identity across arms and paired seeds. Keep optimizer,
LR, warmup, effective batch, objective scale, label masks, checkpoint rule and
retention exposure fixed. A final loader needs the counterfactual excluded-label
test: excluded labels must not alter membership, targets, inputs or gradients.
Do not infer correctness from an upstream loader's name or config alone.

**Fixed draws also require fixed loss accounting.** The
[retained-objective counterfactual](../../native-training-preflight-20261012/LOSS_ACCOUNTING.json)
shows why: increasing only added supervision from 64 to 256 pixels changed a
mixed-batch BCE denominator from 128 to 320 and reduced the unchanged original
patch's BCE gradient to 0.40×. Use a fixed coefficient times the independently
normalized original-stream objective plus its complement times the added-stream
objective, with that same definition in both arms and all loss/deep-supervision
terms covered. Freeze the coefficient with the cohort; the probe's 0.5 is not a
final hyperparameter choice. Verify conditional original-logit gradient identity
when only added support density changes. This controls direct dilution, not the
intended downstream change in model weights, or other data-package differences.
If batch-dependent normalization is present, account for that separate coupling.

The [12 October synthetic loader preflight](../../native-training-preflight-20261012/README.md)
passes that check for the retained zero-coverage-threshold flat recipe, including
default augmentation and all five jitter offsets. Two negative controls require
care: unmasked label-coverage filtering can change patch membership, and flat
depth projection can import excluded values from another plane. Preserve a
separate ignore mask and canonicalize a training-only label view before projection;
do not overwrite source references. Reader's own central-plane label writer leaves
other planes zero. This bounded preflight does not qualify a new real training
cohort or the eventual CUDA loader.

## Data admission and scoring

The [real-source preparation](../../native-coverage-preparation-20261012/README.md)
audited every one of the 56 native-v2 representations. For the first pair, keep
the same eight PHerc0814 segments, per-segment quotas, native acquisition and
teacher in both additional streams. A repeats geometry-reserved regional
support; B uses broader support on those same segments. Neither additional
stream reuses original PHerc0139 patches, which would change its total exposure.
This is a regional-coverage contrast under a fixed parent mixture; broader
parent/acquisition diversity is a subsequent, separately named intervention.

The v2 reservation fixes equal block-boundary eligibility and retains 2,886 A
versus 11,887 B candidates. Its coarse input-neighbourhood ratio is only
1.46–1.60×; do not equate UV patch count with independent physical tissue.
Check all admitted population CT and supervision before freezing schedules.
The five direct-native PHerc0139 annotation sources are proposed for the fixed
original stream. Completed native0139 fits and evaluation remain immutable.

Seven fixed PHerc0814 smoke crops reproduce source decoding and global teacher
resize phase; one missing centre is retained without replacement. The full
34.5M-parameter model passes stream-wise gradient accumulation and a bitwise
two-update save/reload check on real PHerc0139 inputs. This is mechanics only,
not full-cohort admission, CUDA qualification or a learning/power result. The
upstream teacher masks remain weak training supervision, not independent truth.

`READER_TRAIN_EXPOSURE.csv` records 99 released representations across eight
declared source groups. These are not 99 physical subjects. Native/fine aliases,
legacy w028/native w044 and potentially related fragments stay together.
PHerc0343P shares the conservative PHerc0343 family: the old PHerc343 reproduction
is useful development evidence, not a new external Reader test.

Use current native annotations only under their supervision masks and declared
provenance. Reader's prediction/teacher correlation is not CT registration;
the six failed correspondence sites remain unadmitted as precise transferred
labels. Hard unknown-to-negative conversion is forbidden. Broad area-reference
AP alone is not stroke recovery; prepare separately supported development detail
and negative endpoints, without fitting thresholds on the exposed four-segment
comparison. No new reserve parent or completed native0139 endpoint may select
the cohort, model or threshold.

Primary draft pilot endpoint: equal-parent AP gain >=0.02 over matched A, no
parent loss worse than 0.02, and no >2 percentage-point increase in independently
supported negative FPR at a development-fixed recall threshold. Report every
parent, seed, calibration and supervision tier. These margins are practical
drafts, not a power calculation; freeze the final margins/cohort after development
precision and label uncertainty are measured, before confirmation. One pair is
exploratory. Replication requires multiple paired seeds and additional parents.

## Compute and stopping

The [population intake](../../native-coverage-admission-20261012/README.md) is
currently incomplete because public S3 requests encounter proxy-tunnel HTTP 503
errors. All five original annotation sources decode, but only the 481 candidates
on three cached windings have CT support; the other 395 remain unknown. Most
added-stream dependencies are unreceived. Keep the intended five/eight-source
design; do not qualify a smaller availability-selected pair or interpret this
operational pause as a learning result. The exact request lists and partial
caches are portable, and resumption reuses byte-verified receipts.

Prepare exact input/model hashes and a restartable container workload before a
GPU proposal. First measure effective updates/second and peak memory on the
actual device under a separately approved capped timing segment. A5000/3090-class
24 GB hardware is a candidate, not an established fit or speed guarantee. Runtime
must include loading, evaluation and artifact delivery; use an independent
shutdown mechanism. £10 is the all-in first-pilot ceiling, not a promise that a
full arm pair fits inside it. No new paid compute is authorized by this file.

Do not select whichever partially trained arm looks better when time expires.
Use matched completed budgets or report incomplete optimization. Still-improving
training at the cap does not close the broader-learning hypothesis. A gain only
against the generating teacher, from changed old-source exposure, or from losing
hard evaluation support fails the intended mechanism test. Broad architecture
search and from-scratch volumetric pretraining remain deferred.
