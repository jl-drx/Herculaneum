# Exposure-matched training-coverage experiment

**The fixed 512-update coverage arm completed. The predeclared improvement-and-retention screen did not pass.**

## Interpretation and decision

**Keep unchanged seed42 as the baseline and retain the full input contract. The coverage arm is a useful retention result, but it did not earn model promotion, a longer run or GPU work.**

The mixed schedule doubled unique supervision while matching every update's positive/negative counts. It improved equal-parent Paris AP by 0.003813 over seed42, below the required 0.01. It was also 0.001147 below the narrow PHerc0139 control, rather than the required 0.01 above it. The primary hypothesis therefore failed under the frozen rule.

The informative secondary finding is regional retention: all eight regions stay within the allowed 0.02 AP/AUROC loss against seed42. The largest AP loss is only 0.002464, compared with 0.025821 for the old control. This improvement is uneven. Relative to A512, frag2-context2 recovers 0.023357 AP, but frag3-context2 gives up 0.022153 AP and 0.025041 AUROC. It is not uniformly better than the control. Changing the aggregation weights, reporting only Paris2, or lowering the success margin would hide that tradeoff.

On exactly the original PHerc0139 training support, the mixture's final objective is 0.585542 versus A512's 0.583281, and hard-label BCE is 0.312357 versus 0.306838. Both beat the original hard BCE of 0.325393, but the mixture fits the original training region less tightly. Together with phase37, this supports investigating source-dependent transfer and retention rather than adding more identical updates. It does not identify gradient conflict, label error, optical misregistration or weak representation as the mechanism.

One additional confound matters before crediting coverage: the mixture halves the number of PHerc0139 patches, so improved retention may partly reflect less adaptation toward that source. Matching total labeled-pixel counts does not remove that change in source-specific exposure. Neither the common-support risk nor the regional tradeoff establishes that added diversity itself caused the improvement.

The next useful comparison is therefore a cheap, fixed control using the retained predictions: the midpoint of seed42 and A512 scores, `S_half = 0.5*S_seed42 + 0.5*S_A512`. Fix the coefficient before computing its metrics; do not tune it on this panel. Compare its complete regional AP/AUROC and aggregate with coverage512. This tests whether simply reducing movement from seed42 offers a comparable tradeoff. It is a prediction comparator, not proof about a shorter training trajectory or an optimizer mechanism. If it matches or improves coverage512's AP and retention, there is less reason to pursue this particular training mixture. If coverage512 has a distinct advantage, a separately selected source-region replication becomes more useful. Neither outcome qualifies independent ink accuracy or GPU work. This control has not been computed in this phase.

## Primary development comparison

| State | Paris1 AP | Paris2 AP | Equal-parent Paris AP | Paris AUROC |
|---|---:|---:|---:|---:|
| seed42 | 0.240483 | 0.384286 | 0.312384 | 0.721215 |
| A_control-512 | 0.250887 | 0.383802 | 0.317344 | 0.732030 |
| C_coverage-512 | 0.241296 | 0.391099 | 0.316197 | 0.727733 |

Mixed-arm AP gain versus seed42: **+0.003813**. Gain versus phase37 A512: **-0.001147**. The frozen criterion requires at least +0.01 against each, and no region losing more than 0.02 AP or AUROC against seed42. These are practical development margins, not statistical significance.

Region–metric retention failures: **0**. Exact failure records: `[]`.

## Every evaluation region

| Region | Seed42 AP/AUROC | A512 AP/AUROC | Coverage512 AP/AUROC |
|---|---:|---:|---:|
| frag3-context1 | 0.298078/0.870064 | 0.295032/0.873256 | 0.298005/0.872697 |
| frag3-context2 | 0.182887/0.699512 | 0.206741/0.733937 | 0.184587/0.708896 |
| frag1-context1 | 0.185973/0.446052 | 0.182171/0.440002 | 0.184362/0.443530 |
| frag1-context2 | 0.500096/0.903678 | 0.522583/0.911131 | 0.527024/0.912649 |
| frag2-context1 | 0.269839/0.501298 | 0.275038/0.529384 | 0.274238/0.521425 |
| frag2-context2 | 0.581235/0.779541 | 0.555414/0.761336 | 0.578771/0.781074 |
| pherc0814-46527-fresh-rank01 | 0.999824/0.999882 | 0.999811/0.999874 | 0.999810/0.999873 |
| pherc0814-46527-fresh-rank14 | 0.997196/0.999334 | 0.996383/0.999266 | 0.997422/0.999399 |

All six Paris regions and both PHerc0814 regions are reused development. The historical word “fresh” in two filenames does not make them fresh tests. All original scoring support is retained. No checkpoint, cutoff, region or mask was selected from the results. METRICS.json includes every regional delta against both references and separate confusion counts at 0.5 and the old 0.7558947205543518 cutoff.

## Fitting on fixed support

| Mixed updates | Own BCE + KL | Hard BCE | Negative Dice | Manual KL |
|---|---:|---:|---:|---:|
| 0 | 0.602349082 | 0.350165829 | -0.559231654 | 0.000000000 |
| 32 | 0.595896664 | 0.337052420 | -0.561915028 | 0.002792103 |
| 128 | 0.589602061 | 0.326063571 | -0.564086369 | 0.006098162 |
| 256 | 0.588138816 | 0.321990424 | -0.566211268 | 0.007514746 |
| 384 | 0.586989788 | 0.321079819 | -0.566191693 | 0.007554354 |
| 512 | 0.586505938 | 0.320880178 | -0.568090811 | 0.007251778 |

These rows use the mixed schedule. They cannot be compared directly with the old control’s objective values on a different schedule. The following comparison uses exactly the original PHerc0139 schedule for all three states.

| State on original PHerc0139 support | BCE + KL | Hard BCE | Negative Dice | Manual KL |
|---|---:|---:|---:|---:|
| seed42 | 0.597077174 | 0.325393246 | -0.583743866 | 0.000000000 |
| A512 | 0.583281090 | 0.306838395 | -0.580453521 | 0.006525980 |
| Coverage512 | 0.585541631 | 0.312356710 | -0.577738816 | 0.005258589 |

## What the comparison controls

Both recipes use original seed42, the full network, unchanged 17-plane input contract, AdamW at 1e-6, the same BCE smoothing and unit-weight seed42 KL, clipping, 512 updates and deterministic runtime. The additional crop is retained PHerc1667/w013; the old two PHerc0139 regions remain in the mixed schedule. One patch per batch is replaced, alternating slots. New supported labels are deterministically subsampled to match the removed patch’s positive and negative counts exactly. Unknown and retired-validation labels remain excluded.

**Unique supervised pixels increase from 54,701 to 110,014 while per-update class exposure stays identical.** Each 32-update cycle has 230,340 positive and 387,868 negative visits in either recipe. All original unique PHerc0139 support remains represented, plus 55,313 w013 pixels. The schedule repeats 16 times. This controls update budget and labeled-pixel exposure; source identity, spatial arrangement, repeated exposure distribution and teacher confidence necessarily change. It is one particular mixture, not an isolation of every mechanism behind data diversity.

w013 was admitted because it is retained operationally supervised material outside the excluded evaluation/reserve parents; candidate model performance was not used to select its patches. Its inputs and masks were reconstructed exactly from delivered raw chunks. The cached w029 crop has zero training-supervision pixels and was rejected. No source or annotation was downloaded for this study. The w013 crop was previously used development material and is now explicitly training-exposed.

Supplied annotations have incomplete human/teacher ancestry. “Manual” remains an operational mask definition, not certified human-only truth. Source pooling fidelity does not authenticate optical alignment or physical ink depth. No Paris/PHerc0814 evaluation labels, Frag6 or reserved-parent material entered gradients or teacher-cache inputs. PHerc0009B, PHerc0500P2 and PHerc0841 remain reserved. No representation, depth or fusion adapter was introduced.

## Verification and accounting

Seven pre-run fixtures passed. The optimized scalar and its gradients exactly matched the historical control across all 32 retained batches. Separate data reconstruction verified the 32 replacement masks, source coordinates, normalizer, augmentations and class-exposure matching. The teacher remained original seed42; all 32 retained PHerc0139 patch outputs exactly matched the historical teacher cache.

Independent NumPy arithmetic checked 768 loss rows and 4,608 scalar values, including final original-support risk. Maximum errors by term: `{"dice_loss": 1.122787711693718e-07, "full": 1.0514931336746969e-07, "hard_base": 7.928146666946745e-08, "loss_D_M": 1.3877787807814457e-17, "loss_U": 1.0514931336746969e-07, "probability_drift_manual": 1.3877787807814457e-17}`. It did not replay neural autograd or optimizer updates. Worker checks enforce finite gradients/parameters, zero unsupported output gradients, all 264 continuous optimizer step counters, measurement isolation and exact safe state reloads.

Independent evaluation checked 48 AP/AUROC values, all canonical supports, parent means, regional contrasts and the decision. It reconstructed 2,359,296 score pixels from 400 retained tiles, including the old control. Maximum rank discrepancy 1.11e-16; maximum sigmoid/Hann blend discrepancy 2.79e-07, below the frozen 2e-6 tolerance. No model calls were added by these numerical checks.

Scientific controller time: 591.144 seconds, below 1,800. Actual counts: 512 optimizer updates/backwards; 705 batch-two student forwards; 32 batch-two frozen-teacher forwards; 200 single-tile development forwards. The original control and baseline were reused without new inference. Peak training RSS: 1,625,214,976 bytes, below 4 GiB. No automatic retry, threshold fitting or budget extension.

The first unexecuted freeze copied obsolete descriptive arm/count/endpoint fields from phase37. Preflight caught this before any model call. PLAN-V2.json and GO-V2.json correct those fields, with identical scientific recipe, data and tolerances; original files remain retained. Active execution used run_experiment_v2.py and its pinned v2 worker copies. This was a prelaunch metadata correction, not a failed scientific run or result-driven change.

GPU spend remains £0; the £10 total future-pilot cap is unchanged. No network acquisition, R2 upload, outreach, publication or submission occurred. Repository source/tests/locks are unchanged. Local retention and saved restart instructions do not establish an external backup or new snapshot restoration.

[Mixed training curves](figures/learning-curves.png) and [all regional deltas](figures/all-regional-deltas.png).
