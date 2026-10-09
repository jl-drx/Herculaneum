# Plinius phase 20: depth-feature fusion did not preserve baseline performance

**The prespecified study is complete. The explicit-depth adapter beat its matched raw-input control, but failed five baseline-retention criteria. Retain unchanged seed42; do not promote either adapter or scale this recipe to a GPU.** Independent training, development and metric verification passed. This is a negative developmental result, not a failed execution.

## Main result

The table uses the frozen equal-fragment Paris2 aggregation. Average precision (AP) measures ranking agreement with the supplied ink labels; it is not transcription accuracy.

| Model | Paris2 AUROC | Paris2 AP |
| --- | ---: | ---: |
| Unchanged seed42 | 0.657642 | 0.384286 |
| Raw feature control | 0.628518 | 0.304992 |
| Explicit depth features | 0.647074 | 0.313037 |

Explicit depth improved AUROC by **0.018557** and AP by **0.008045** against the raw control, exceeding both prespecified 0.005 margins. Against seed42, however, it lost **0.010568 AUROC and 0.071248 AP**. The full 17-component screen passed 12 components and failed five. The relative-to-control gains do not establish an improvement over the existing model.

All six regions are reused DEVELOPMENT. Paris2 and its aliases were excluded from gradients, teacher inputs and cutoff fitting. Neither these results nor their numerical verification establish independent confirmation, frontier performance or newly readable text.

![Training risk and all-region development differences](figures/depth-fusion-study.png)

[SVG figure](figures/depth-fusion-study.svg) · [Exact metrics and every gate](analysis/metrics.json) · [Independent verification](independent-review/RESULTS.md)

## What was tested

The [frozen protocol](RESEARCH_DESIGN.md) compared two freshly initialized, capacity-matched 5,296-parameter branches inserted into seed42's 32-channel depth-fusion features before its frozen 2D U-Net. The raw arm received previous/center/next intensity planes. The explicit arm received intensity, axial differences and nominal depth index. Differences and coordinates were changed together; the study does not identify their separate contributions. Nominal index is not physical ink depth.

Both arms used the identical retained PHerc0139 schedule: 32 batch-two updates, 17 planes, unchanged normalization and XY augmentation, manual smoothed BCE plus unit-weight seed42 KL on the same manual pixels. The original 34,546,498 parameters remained frozen and in evaluation mode. The decoder retained gradients with respect to its input, allowing the feature adapter to learn. Frozen parameter values alone do not preserve predictions when the features entering that decoder change.

Before updates, both adapters with zero-initialized projections reproduced the original outputs bit for bit on all 32 scheduled inputs. Their stems were randomly initialized. The original outputs also matched their retained teacher caches exactly. Both final adapters were then evaluated on all six prescribed regions and all 25 tiles per region, with original-model cache equality checked on every tile. The cutoff remained **0.7558947205543518**; no threshold, checkpoint, learning rate or seed was selected from these outcomes.

## Training result

Both arms started at smoothed BCE/full risk **0.5970771741**, with zero KL. Final fixed-schedule measurements were:

| Measurement | Raw control | Explicit depth |
| --- | ---: | ---: |
| Smoothed manual BCE | 0.5920359623 | 0.5905926935 |
| Manual teacher KL | 0.0020616669 | 0.0032397444 |
| Full objective | 0.5940976292 | 0.5938324379 |
| Hard-label BCE, diagnostic | 0.3154183514 | 0.3163772486 |
| Mean absolute manual probability drift | 0.0120784395 | 0.0165220329 |

The explicit arm improved its own full risk by 0.0032447362 and finished 0.0002651913 below raw control, passing the 0.0001 margin. It had higher KL and probability drift, and slightly worse hard-label BCE than raw control. Loss improvement therefore does not imply better preservation of the reference. Fixed-schedule risk was nonmonotone; the raw arm at update 16 exceeded its initial full risk. All scheduled measurements were retained and only update 32 was evaluated on development data.

These are equal-batch empirical risks over repeated and overlapping training exposures, not population estimates. The manual objective covered 54,701 unique scheduled pixels. No historical pseudo-label array supplied targets or loss; such arrays participated only in historical input-identity checks.

## Regional results

Each cell reports AUROC / AP. All regions are included, including improvements and regressions.

| Region | Seed42 | Raw control | Explicit depth |
| --- | ---: | ---: | ---: |
| frag1-context1, Paris2 | 0.446052 / 0.185973 | 0.492388 / 0.195863 | 0.485090 / 0.196248 |
| frag1-context2, Paris2 | 0.903678 / 0.500096 | 0.810680 / 0.371352 | 0.804515 / 0.298378 |
| frag2-context1, Paris2 | 0.501298 / 0.269839 | 0.480877 / 0.242757 | 0.515983 / 0.264768 |
| frag2-context2, Paris2 | 0.779541 / 0.581235 | 0.730126 / 0.409998 | 0.782710 / 0.492756 |
| frag3-context1, Paris1 | 0.870064 / 0.298078 | 0.865420 / 0.273716 | 0.828191 / 0.246072 |
| frag3-context2, Paris1 | 0.699512 / 0.182887 | 0.669590 / 0.140104 | 0.709984 / 0.203192 |

The candidate failed the maximum 0.02 loss versus seed42 in five components: frag1-context2 AUROC (-0.099163) and AP (-0.201718), frag2-context2 AP (-0.088480), and frag3-context1 AUROC (-0.041873) and AP (-0.052006). These are substantial failures rather than rounding ambiguities. Candidate AUROC improved in four regions and AP in two; the result is heterogeneous, not a universal decline.

Paris1 equal-context AUROC/AP were seed42 **0.784788 / 0.240483**, raw **0.767505 / 0.206910**, and depth **0.769087 / 0.224632**. At the frozen cutoff, Paris2 sensitivity/specificity were seed42 **0.025981 / 0.985120**, raw **0.022909 / 0.985793**, and depth **0.018330 / 0.984367**. The low sensitivities matter to practical use. All confusion counts and other cutoff metrics are retained. Undefined precision for raw control in a Paris1 region remains undefined in the macro summary; it was not converted to zero.

## Verification and resources

The frozen implementation passed 30 author synthetic checks and 44 independent technical checks before scientific execution. The separate numerical checker verified losses, manual logit gradients, clipping, AdamW transitions, state continuity, probability blending, rank metrics, cutoff arithmetic, aggregation and all 17 screen decisions. It made 2,812 loss/summary comparisons, 1,355,776 optimizer comparisons and 985,056 exact adjacent-state scalar comparisons. Maximum loss error was below 9.81e-8 against a 2e-7 tolerance; maximum manual-gradient error was below 1.04e-11 against 2e-10. Maximum blend error was 2.98e-7 against 2e-6. No decision boundary was ambiguous and no post-output tolerance or checker source was changed.

Independent checking did not rerun the convolutional model, independently replay autograd, or decode the checkpoint. Hidden feature execution and original/cache equality rely on reviewed frozen code and its bound runtime receipts; arithmetic agreement is not an independent proof of every upstream operation. The original annotation/teacher ancestry also remains unresolved.

Scientific execution used Python 3.12.14, NumPy 2.5.3 and Torch 2.12.0+cpu, four intraop threads and one interop thread. Training controller time was **55.72 seconds** and development controller time **52.21 seconds**, excluding implementation and review. Training worker peak RSS was about **1.33 GB**. Both final adapter files are 23,705 bytes; no new full-backbone checkpoint or intermediate learned checkpoint was saved.

Across four workers there were four retained-backbone loads, 64 optimizer updates/backwards, 64 decoder-input backwards, 364 original full-model calls, 364 fusion calls, 1,112 original 2D-network calls and 748 adapter calls. Full-model and component counts overlap and must not be summed as independent forwards. Scientific workers ran once each, with no failures or retries.

One root prelaunch metadata check incorrectly expected PASS where the frozen preflight reported READY. Its attempted runner invocation then stopped before worker creation, array loading or model calls because no GO file existed. Only the root assertion was corrected; the scientific protocol and frozen runner/checker stayed unchanged. A separate pre-execution synthetic review-oracle dtype failure was also retained and corrected before freezing. Neither event was a scientific-result retry. See [prelaunch receipt](ROOT_PRELAUNCH_STOP.json) and [technical review](technical-review/TECHNICAL_REVIEW.md).

## Frag6 and the next decision

The parallel [Frag6 plan](frag6-plan/README.md) made four public HEAD requests, all successful, and read zero response-body bytes. It proposes a first structural stage bounded to 13 requests / 202,301 response-body bytes for PPM and three TIFF headers/IFDs. No CT pixels, PPM numerical records, labels or model predictions were acquired. Advertised range support is not yet tested. That stage needs an implemented, fixture-tested and frozen reader before execution; later native reconstruction remains blocked by missing declared rendering metadata.

The [next-study draft](NEXT_STUDY_DRAFT.md) prioritizes this source-contract work and a training-only preservation-support audit. The present result is consistent with harmful transfer under this specific recipe. It does not establish that explicit depth is intrinsically unhelpful, nor isolate optimizer scale, support coverage, geometry or domain shift as its cause. Another architecture or GPU run would not resolve that uncertainty by itself.

**Paid GPU spend remains £0; the first paid pilot retains its £10 total cap.** No promotion, translation, submission, upload, external publication or new credential action occurred. Tracked repository files and all first-nineteenth evidence remain unchanged. Local retention and saved environment instructions are not an external backup or verified fresh-task restoration.
