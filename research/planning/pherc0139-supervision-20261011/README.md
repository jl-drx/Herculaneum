# PHerc0139 site1: supervision sensitivity to CT correspondence

11 October2026. **The retained alignment alternatives materially change hard
pseudo-labels on three of the four fixed supports; continuous scores change much
less.** This makes ordinary soft supervision an essential baseline before claiming
that a more elaborate geometry-aware method is needed. The fourth support has
no teacher-positive pixels at the frozen threshold and cannot establish stability.
No new model inference, training, GPU resource or CT acquisition was used.

This is a teacher-only diagnostic on inspected w040 training tissue. The source
is a2D model prediction, not ground truth. The original strict CT screen remains
failed. Nothing here certifies ink, an ink face, precise3D transfer or new letters.

## What was fixed and verified

Before target pixels, we froze the four original32-square checking supports,
the same common CT fit(0,−1,0), all11/6/22/4 descriptive CT alternatives,
threshold>=48 from the released **w040 own-calibration** report, two sampling
operators, primary native index14 and all eight checking planes10:18. Alternative
sets were not fit or weighted using teacher agreement and are not confidence sets.

The provenance audit tied the exact teacher filename to fine78 volume20260102150214
and the published v42 inference path. That path preserves full-raster XY without
an extra crop or translation; the TIFF header agrees. It aggregates fine layers
[23,85) and does not identify an ink depth. Current-object ranges are pinned by
ETag, length and SHA256; no historical whole-object checksum is available.

The live manual-label count differs from the historical builder report, so we
measured pseudo-targets **before manual overrides**. These are not the actual
historical merged labels and cannot diagnose the released model's learned weights.

Exactly4 compressed TIFF tiles/428,618bytes were fetched with strict206/range/ETag
checks, within the frozen8MiB cap. All compressed and decoded hashes are retained.
The780×788 crop is x9084:9872,y7088:7868 in the fine raster. Reconstructed native
eligibility uses all28 retained CT planes; no new CT was fetched. The complete
common pseudo-supervision support is1,024pixels per check for every tested case.
There were no supervision-support changes hidden by common-mask restriction.

## Primary result: native index14

Primary operator: unrounded4×4 teacher mean followed by bilinear interpolation
at the previous geometry queries. This is a declared approximation, not exact
integration over a curved native pixel and not released uint8 INTER_AREA replay.

“Variable positive” means a pixel above threshold under at least one retained
alternative and below it under another. It is an envelope over the finite set,
not the fraction changed by every individual shift or an estimated error rate.

| Original support | Alternatives | Positive union | Variable positives | Fraction of positive union | Worst single-candidate positive-union score MAE,0–255 |
|---|---:|---:|---:|---:|---:|
|4|11|0|0|Not defined|Not defined|
|5|6|102|33|32.35%|6.15|
|6|22|451|146|32.37%|6.94|
|7|4|144|32|22.22%|2.42|

The largest individual fit-versus-alternative threshold disagreements are23.9%,
17.1%,15.8% of their respective pairwise positive unions. Do not confuse those
pairwise denominators with the larger set-envelope denominator above.

The three informative supports exceed the prospectively chosen5% label-change
routing tolerance. Their score MAEs are0.0241,0.0272,0.0095 normalized; all remain
below the separate0.05 continuous-score routing tolerance. These operational
cutoffs were declared before target access and are not calibrated to training AP.
Binary thresholding makes modest displacement consequential for hard labels.
For BCE, changing a soft target changes the logit gradient by exactly that score
change; changing a hard label changes it by one. This observation does not predict
an eventual model's accuracy or make these scores calibrated probabilities.

## Controls and limits

Point-bilinear sampling without4×4 pre-pooling gives variable-positive fractions
29.81%,32.37%,22.22% and MAEs6.07,6.95,2.44. It reaches the same operational
conclusion; the result is not explained away by our additional pre-pooling.
Both still interpolate an already aggregated teacher prediction.

Across every fixed native plane10,...,17, the primary-operator variable-positive
fractions remain30.0–33.3%,32.0–32.9%,21.9–22.4%. All three retain adequate frozen
class counts and are operationally sensitive under both operators. Support4 has
zero threshold-positive pixels throughout and remains inconclusive. No support
was replaced or searched after looking at targets.

The separate same-operator naive-canvas versus geometry-zero contrast changes
more labels on these informative supports:48.3%,35.8%,90.2% of pairwise positive
union. These figures include source-frame/inverse-geometry differences and are
**not** residual registration uncertainty, an exact reproduction of the public
INTER_AREA calculation, or demonstrated historical label error. Geometry-zero
versus the one-pixel CT fit is separately retained. Neither contrast was mixed
into the headline CT-alternative envelope.

All calculations project a2D teacher onto query XY. Five support6 candidates
have dz=1; their projected changes do not measure a wrong-depth/face target error.
Depth uncertainty remains unresolved even where projected values barely change.
The other alternatives also do not certify physical ink-surface identity.
Four neighboring supports in one training winding do not establish population
statistics, generalization, target accuracy or a pristine evaluation benchmark.

## Independent checks

12 prospective arithmetic/support fixtures and a synthetic end-to-end check
passed. An independent implementation imported none of the study calculator:
it decoded the4 source tiles, rebuilt native support, and recomputed all880
transfers/901,120pixel values, with maximum score difference2.84e−14. It recomputed
all64 finite-set summaries and816 pair comparisons from retained arrays exactly.

One exact48-valued source plateau gives47.99999999999999 in the frozen four-product
sum and48 in a nested interpolation sum. The same unique UV occurs in six
support6 alternatives at j11, producing six threshold occurrences with different
rounding; none is at primary j14. All64 envelope counts and routing decisions
remain unchanged under the alternate arithmetic. Frozen outputs/thresholds were
preserved, and every discrepancy is retained in the independent review. Do not
claim every thresholded candidate bit is implementation-order invariant.

The four fixed0–255 primary panels were viewed. They show smooth score gradients,
not authenticated character shapes. No fitted contrast was used. Their limited
appearance is consistent with the numerical outcome; panels do not replace it.

## Next research decision

Do not jump from hard-label sensitivity to a new architecture. First freeze a
matched-physical-region comparison with an ordinary **soft-target baseline**;
then require any CT-informed target averaging to beat that baseline and a
support- and smoothing-matched control. Candidate sets are not probabilities, so
uniform averaging would be a declared treatment, not posterior marginalization.
With BCE it is still training against an averaged target, not by itself a novel
latent-geometry model.

A useful training design must fix acquisition, initial weights, exposure, patch
sequence, target decoding and prospective physical evaluation regions. Corrected
source coordinates and residualCT alternatives are separate interventions. Do
not train only these tiny checked supports, reuse inspected regions as an untouched
test, or measure success solely by agreement with the same teacher used in training.
The current evidence prioritizes that controlled baseline; it does not yet supply
independent ink-accuracy supervision or justify a GPU launch. No extra depth sweep
or crop search follows from this audit.

See PROTOCOL.md, REQUEST_PLAN.json, provenance/REPORT.md, results/RESULTS.json,
review/independent/ and PORTABILITY.md. This supporting result does not advance
the formal study50/registerv46 cumulative seal. The separate completed PHerc0800
study52 retains its own final review. Everything is saved privately; no PR,
Actions run, spare-VM task or paid compute was used in this turn.
