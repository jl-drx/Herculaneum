# Segment weights need an explicit loss contract

12 October 2026. The existing separate original/added objectives protect the
original component. Within the added stream, however, **equal segment draw quotas
do not also fix segment BCE coefficients**. A pooled masked BCE weights each
segment by its share of supervised batch pixels; the retained per-image Dice
term weights it by its share of images. This is a known normalization property,
not a novel objective or an observed explanation for earlier learning failures.

The offline audit quantifies this distinction and qualifies a small proposed
per-segment loss wrapper. It does not change the qualified training harness,
historical results, eight-segment reservation, or evaluation material. No model
forward, optimizer update or paid compute occurred. The full pair remains
unadmitted while source acquisition is blocked.

## Evidence from saved populations

With equal unaugmented draws from the five original PHerc0139 windings, the
large-batch limit of each source's pooled BCE coefficient is:

| Winding | Patch quota / Dice coefficient | Pooled BCE coefficient |
| --- | ---: | ---: |
| w035 | 20% | 19.370% |
| w039 | 20% | 19.249% |
| w040 | 20% | 18.419% |
| w041 | 20% | 20.820% |
| w044 | 20% | 22.142% |

These are coefficients of per-source mean BCE, not gradient magnitudes or ink
prevalence. They do not imply an A/B confound in the original stream: its exact
draws, masks and objective will be held identical between arms. No change to that
already qualified stream is needed on this evidence.

All **292,393** cross-winding patch pairs were also enumerated, covering each of
the ten source pairs with one draw from each. For example, w040's expected BCE
coefficient against w044 is **44.792%**, despite a 50% patch quota. This finite
two-source example is not the five-source training schedule or the large-batch
limit above. Equal expected counts and expected ratios are different quantities.

For the one complete added segment, `20250925161630`:

| Descriptive quantity | Restricted A | Broad B |
| --- | ---: | ---: |
| Mean supervised pixels per draw | 14,571.92 | 14,652.63 |
| Zero-positive teacher patch probability | 13.333% | 19.302% |
| Mean positive share, one patch | 23.190% | 20.706% |
| Expected positive share, two independent patch draws | 22.661% | 20.182% |
| Positive share, large-batch limit | 22.135% | 19.665% |

Mean supervised area differs by only **0.554%** in this case. That does not
establish a material full-cohort normalization confound: seven added segments
are unresolved. The roughly **2.47 percentage-point** teacher-positive difference
and increased zero-positive draw probability survive this bookkeeping. They are
data-package differences that source normalization cannot remove. Do not fit
balancing weights from the first available segment or call teacher negatives
authenticated blanks.

The two-patch calculation enumerates **1,763,325** ordered draws, including
repeated patches. Counts precede BCE target smoothing, Dice remains nonlinear
in predictions, and future augmentation can change support. A separate
standard-library implementation recomputes the table arithmetic to 1e-12;
see [INDEPENDENT_CENSUS.json](INDEPENDENT_CENSUS.json).

## Proposed addition to the future paired study

Preserve the existing original-stream objective. For the added stream, use a
fixed weighted mean of separately normalized segment objectives:

`L = alpha * L_original + (1-alpha) * sum_s beta_s * L_added_segment_s`

Each segment retains masked BCE plus the same per-image Dice term. Both arms
use identical positive coefficients `beta_s` summing to one, in addition to
identical segment draw quotas. Freeze their actual values, `alpha`, batch sizes
and schedules only with the full population. The probe's 0.5/0.5 coefficients
are demonstrative, not a selected training setting. This definition differs
from pooling every added pixel into one denominator and must apply to both arms
before any new performance feedback.

The full optimizer accumulation window must include every configured source.
Use its **post-augmentation** supervised-pixel and image totals; never recompute
denominators per microbatch, drop an empty source or silently redistribute its
weight. Slot generation, augmentation, every loss/deep-supervision term and
complete consumption of the window still require final loader qualification.
This wrapper is not that production loader.

[The prototype](source_objective.py) was checked on the first four saved added
patches, divided into two mathematical probe groups. They are all from one real
segment, not evidence of two independent physical sources. Reducing only the
second group's mask to 1,024 previously supervised pixels per patch:

- Scales the unchanged group's pooled BCE gradient by **1.87944×**, as predicted
  by the denominator. This deliberate stress change is not the observed A/B
  population difference.
- Leaves its proposed per-group BCE-plus-Dice logit gradients **bitwise equal**.
- Gives **zero measured relative gradient difference** for full versus two
  microbatch partitions. Scalar loss differences are below 6e-8.
- Rejects a naive microbatch mean (2.985% relative gradient difference), a missing
  configured source, invalid coefficient sums, empty support and unknown sources.

The first inter-stream dilution test already established the mechanism. This
follow-up measures within-stream source weighting and checks the proposed
source-level implementation; it makes no training-improvement or novelty claim.
It does not preserve within-source class mixture, annotation quality, physical
independence, or future model trajectories. Full-population accounting and
supported development/evaluation references remain required before GPU approval.

## Reproduction and operational status

[FREEZE.json](FREEZE.json) binds the plan, code, prior receipts and inputs before
this calculation. [RESULTS.json](RESULTS.json) contains every source-pair result,
class summary and conditional probe. Table-only verification needs no archive,
PyTorch, NumPy, scan or model download:

```bash
python research/in-progress/native-source-accounting-20261012/verify_census.py \
  --repo /path/to/Herculaneum \
  --result /path/to/Herculaneum/research/in-progress/native-source-accounting-20261012/RESULTS.json \
  --output /new/INDEPENDENT_CENSUS.json
```

The complete loss probe needs the source qualification arrays restored by the
[latest input checkpoint instructions](../native-coverage-retry-20261012-2318/PORTABILITY.md),
NumPy 2.4.4 and CPU PyTorch 2.13.0:

```bash
python research/in-progress/native-source-accounting-20261012/audit_accounting.py \
  --repo /path/to/Herculaneum \
  --qualification /restored/work/added-qualification-20261012 \
  --output /new/source-accounting
```

The [23:43 transport retry](TRANSPORT_RETRY.json) failed its one-byte source
preflight with HTTP 503, before bulk acquisition. No new source object was
retained: **3,515 present; 14,055 pending**. The GitHub metadata read and normal
push also failed with HTTP 503. Runtime status reported running/connected with
no provider failure; these observations do not identify the transport's cause.
No acquisition process remains active. The latest 361 MiB input/qualification
archive is restore-tested locally but not confirmed remotely; the earlier
checkpoint was privately published and downloaded successfully at `2369f0d`.

This report is locally committed; see [PUBLICATION.json](PUBLICATION.json) for
its final publication attempt. Continue the same source requests when transport
recovers. Do not replace the missing cohort, repeat completed learning studies,
or use this mechanics pass as a GPU learning gate.
