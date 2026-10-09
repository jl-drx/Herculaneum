# Fourth experiment: new PHerc0800 screening region

**The frozen CPU screen completed and found two main shared detector responses. No coherent source-supported letter sequence, transcription or translation is established.** These are localized leads for source/geometry inspection, not authenticated ink.

Open the [visual evidence](analysis/review.html), [whole-field source and score panels](analysis/screen-overview.png), or [fixed depth comparisons](analysis/condition-sensitivity.png). The source images use unchanged 0–255 grayscale; raw-score displays use fixed 0.25–0.75 limits. The labeled maximum projection combines depth slices and is not a single observed layer.

## Region selected before viewing its intensities

The frozen geometry-only rule evaluated 1,124 chunk-aligned positions on five retained surfaces. There were 228 fully valid candidates outside previously acquired same-source extents; 29 passed the fixed distance rule. The selected 384×384 crop on segment `20251028220955` starts at source X/Y `(1024,384)` of PHerc0800 volume `20250521135224`.

All 147,456 crop pixels have finite valid geometry. Exact nearest-neighbor comparison against 2,693,760 prior acquired geometry-valid base coordinates gives a minimum separation of 52.8981 native voxels, approximately 457.04 µm. No points were subsampled. Neighboring coordinate steps are approximately one voxel with no detected degenerate finite-difference cross products. This does not reconstruct complete renderer sampling support or prove independent papyrus layers. See the [geometry review](GEOMETRY_REVIEW.md) and [immutable selection](geometry-selection/selected-plan.json).

The full canvas is approximately 3.32 mm square; its fixed 64-pixel-margin analysis interior is 2.21 mm square. This limited context cannot establish a readable passage. The whole crop is geometry-valid but not uniformly 64 pixels from the original global geometry boundary; the fixed interior lies within a completely valid crop and all contributing tiles remain inside that crop.

## Source acquisition and execution

All nine requested native source chunks returned genuine observed payloads. No missing-object fill was needed. Source payload totaled 4,571,136 bytes; metadata brought total acquisition response-body traffic to 4,574,804 bytes, below the 6 MiB cap. The separate current source/license check used 81,035 bytes; combined recorded response-body traffic was 4,655,839 bytes. Neither tally includes protocol overhead. No whole volume or new model weights were downloaded.

Both pinned final-step checkpoints passed SHA-256/size checks and strict state matching. The unchanged 9 µm adapter used CPU FP32, four threads, native intensity data, upstream patchwise normalization, 128×128 windows, stride 64 and Hann blending. The predeclared runs were center `7:24` with both seeds, then seed42 forward `5:22`, forward `9:26` and reversed `7:24`. All 125 scheduled tiles were scored. Actual model inference totaled approximately 10.65 seconds; the four subprocesses together took 23.70 seconds including startup and checks. No GPU, paid provider operation, training or fitted preprocessing was used.

The analysis verified the frozen plan, original source chunks and geometry crops, assembled-array alignment, observation masks, checkpoint/source pins, execution manifest, all raw score arrays and mask hashes. Primary and stricter support domains are identical: 65,536 pixels, with no bad contributing tile and no missing, invalid or all-zero depth profile in that domain.

## What the detector measurements show

| Fixed interior diagnostic | Measurement |
| --- | ---: |
| Both center raw scores above 0.5 | 3,019 / 65,536 pixels (4.61%) |
| Center seed correlation | 0.510 |
| Above-midpoint seed Jaccard overlap | 0.302 |
| Center vs minus-two depth correlation | 0.939 |
| Center vs plus-two depth correlation | 0.878 |
| Center vs reversed-depth correlation | 0.250 |

The midpoint is a fixed behavior diagnostic for smoothed training targets, not calibrated ink confidence. Both seeds share training data; their agreement does not independently authenticate ink. Each two-slice shift shares 15 of 17 slices with the center window, so their similarity is not independent replication. Reversing depth tests model sensitivity, not a scientifically established ink-negative condition.

Six connected diagnostic components were retained, including three single-pixel components. Two dominate the shared response: 2,583 pixels at source bbox `(1247,471,64,55)`, peak `(1278,504)`; and 407 pixels at `(1095,610,23,32)`, peak `(1100,628)`. All components, exact CT coordinates, raw scores and masks remain in [summary.json](analysis/summary.json). These detector-selected locations are post hoc leads, not a fresh validation set.

Root visual inspection shows broad rounded detector responses and textured, fissured source structure, without a coherent stroke sequence supported across the observed image. Small depth shifts retain much of the response; reversal changes it substantially. These observations warrant source/geometry diagnosis but do not support naming a letter, language, author or text. The [separate assistant visual review](VISUAL_REVIEW.md) reached the same conclusion after viewing both image panels; neither review is specialist papyrology.

## Decision and retained evidence

Keep these results as unresolved leads. Before treating either response as ink or using it for training, check its mesh placement and depth profile against native CT. A later larger surrounding crop can test spatial context, but must be documented as follow-up to detector-selected locations. Do not choose a preprocessing variant or crop because it resembles an expected letter. No target pseudo-labeling or fine-tuning is justified by this screen alone.

Original source, immutable plans, code, raw outputs, logs, geometry diagnostics and hashed manifests are retained privately under this experiment and `data/experiments/pherc0800-screen384-20261008`. No novel result was published or submitted. R2 access is verified in the local Windows runtime; these new cloud results have **not** been uploaded to R2. The £10 TOTAL first-GPU-pilot allowance remains unused by this experiment, and all paid-launch prerequisites remain in force.
