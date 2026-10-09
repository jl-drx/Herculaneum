# CPU budget × objective experiment

**All 1,024 updates and all four fixed development evaluations completed.** **No endpoint passed the predeclared baseline-improvement screen.**

This study compares longer full-network continuation with one objective change. It does not test a depth/fusion adapter, perform input recentering, or establish independent ink accuracy. Every evaluation region was already inspected development material.

## Interpretation and decision

**Retain unchanged seed42 and the existing full 17-plane input contract. Do not extend either trajectory, promote either checkpoint, or rent a GPU from this result.**

The control's update-32 weights and all fixed-risk outputs exactly reproduce the historical run. Extending that same optimizer trajectory to 512 updates lowers its own training objective from 0.590246 to 0.583281. Paris AP rises by 0.005191 versus update 32, but only 0.004960 versus unchanged seed42, below the predeclared 0.01 margin. The frag2-context2 AP loss of 0.025821 also breaches regional retention. There is a modest budget effect, not the material, retained improvement needed to change our decision.

The Dice arm provides the sharper counterexample to interpreting training progress as transfer. From update 32 to 512 its hard-label training BCE falls from 0.306468 to 0.250895 and soft Dice rises from 0.602102 to 0.649006. Yet Paris AP changes by only +0.000038, staying 0.005194 below seed42. At update 512 it is 0.010154 below the matched control and breaches retention in four Paris regions (five region–metric failures). Both PHerc0814 regions remain within the retention margin; the main failure is uneven Paris transfer. AP is a ranking measure, so retuning a single score threshold cannot repair this result.

This reduces the case that the earlier negative results arose simply from stopping at 32 updates. It does not establish convergence at 512, rule out every optimizer/objective, or identify weak representation as the cause. We tested one initializer, one fixed learning rate, one Dice weight and repeated exposures from two regions of one training parent. Domain coverage, annotation quality, teacher ancestry and unresolved reference registration remain plausible constraints. The changed gradient scale is part of the Dice recipe tested, not independently separated here.

My next recommendation is a CPU comparison of training-data coverage at a fixed budget and objective, conditional on identifying additional defensible manual training material. It should keep the original two-region control, the existing evaluation panel and the full input contract. It must not relabel the already inspected Paris panel as an untouched test or consume the reserved parents. The candidate design and its data-eligibility gate are in NEXT_STUDY_DRAFT.md; no further experiment is frozen or launched.

[Training curves](figures/learning-curves.png) and [all regional deltas](figures/all-regional-deltas.png) show both the fitting gains and the retention failures.

## Primary development result

| State | Paris1 AP | Paris2 AP | Equal-parent Paris AP | Gain vs seed42 | Region–metric failures | Pass |
|---|---:|---:|---:|---:|---:|---|
| seed42 | 0.240483 | 0.384286 | 0.312384 | +0.000000 | 0 | Baseline |
| A_control-32 | 0.243952 | 0.380353 | 0.312153 | -0.000231 | 0 | False |
| A_control-512 | 0.250887 | 0.383802 | 0.317344 | +0.004960 | 1 | False |
| B_dice-32 | 0.236118 | 0.378187 | 0.307153 | -0.005231 | 1 | False |
| B_dice-512 | 0.246150 | 0.368230 | 0.307190 | -0.005194 | 5 | False |

The screen requires at least +0.01 absolute equal-parent Paris AP versus seed42 and no one of eight development regions losing more than 0.02 AP or AUROC. These are fixed practical margins, not statistical significance. PHerc0814 is an additional retention diagnostic, not an improvement target. No region, threshold or checkpoint was selected from the outcomes.

| Matched contrast | Paris AP change | At least +0.01 |
|---|---:|---|
| budget_A: A_control-512 minus A_control-32 | +0.005191 | False |
| budget_B: B_dice-512 minus B_dice-32 | +0.000038 | False |
| objective32: B_dice-32 minus A_control-32 | -0.005000 | False |
| objective512: B_dice-512 minus A_control-512 | -0.010154 | False |

## Every region

| Region | Seed42 AP/AUROC | A32 | A512 | B32 | B512 |
|---|---:|---:|---:|---:|---:|
| frag3-context1 | 0.298078/0.870064 | 0.295781/0.869129 | 0.295032/0.873256 | 0.295437/0.864269 | 0.271130/0.843504 |
| frag3-context2 | 0.182887/0.699512 | 0.192123/0.713854 | 0.206741/0.733937 | 0.176799/0.680165 | 0.221171/0.707293 |
| frag1-context1 | 0.185973/0.446052 | 0.184922/0.443266 | 0.182171/0.440002 | 0.188398/0.450476 | 0.184665/0.439901 |
| frag1-context2 | 0.500096/0.903678 | 0.490315/0.900825 | 0.522583/0.911131 | 0.474507/0.893671 | 0.469000/0.891635 |
| frag2-context1 | 0.269839/0.501298 | 0.275617/0.521068 | 0.275038/0.529384 | 0.271019/0.499190 | 0.275297/0.480095 |
| frag2-context2 | 0.581235/0.779541 | 0.570560/0.781489 | 0.555414/0.761336 | 0.578826/0.789677 | 0.543958/0.791157 |
| pherc0814-46527-fresh-rank01 | 0.999824/0.999882 | 0.999803/0.999868 | 0.999811/0.999874 | 0.999784/0.999855 | 0.999748/0.999831 |
| pherc0814-46527-fresh-rank14 | 0.997196/0.999334 | 0.997490/0.999374 | 0.996383/0.999266 | 0.997279/0.999331 | 0.996264/0.999143 |

All fixed support pixels remain included. Exact score-tie AP/AUROC, 0.5 confusion counts and separately labeled old-cutoff 0.7558947205543518 confusion counts are in METRICS.json. Neither threshold was fitted. Score-scale changes and ranking are distinct; old-cutoff sensitivity alone does not decide this experiment.

## Learning curves on the same fixed training exposures

| Arm | Updates | Own objective | Common hard BCE | Common smoothed BCE | Negative Dice | Manual KL |
|---|---:|---:|---:|---:|---:|---:|
| A_control | 0 | 0.597077174 | 0.325393246 | 0.597077174 | -0.583743866 | 0.000000000 |
| A_control | 32 | 0.590246351 | 0.311774724 | 0.586598447 | -0.585068817 | 0.003647904 |
| A_control | 128 | 0.585231781 | 0.307727635 | 0.579932207 | -0.581071893 | 0.005299573 |
| A_control | 256 | 0.584187384 | 0.306528558 | 0.578007422 | -0.581685615 | 0.006179962 |
| A_control | 384 | 0.583656459 | 0.307123407 | 0.577162361 | -0.580055452 | 0.006494099 |
| A_control | 512 | 0.583281090 | 0.306838395 | 0.576755110 | -0.580453521 | 0.006525980 |
| B_dice | 0 | 0.013333309 | 0.325393246 | 0.597077174 | -0.583743866 | 0.000000000 |
| B_dice | 32 | -0.005946020 | 0.306468254 | 0.592000358 | -0.602101836 | 0.004155458 |
| B_dice | 128 | -0.029892917 | 0.271045118 | 0.585244967 | -0.628629009 | 0.013491126 |
| B_dice | 256 | -0.037393606 | 0.258300285 | 0.585652215 | -0.641237024 | 0.018191203 |
| B_dice | 384 | -0.040316936 | 0.253730228 | 0.585923882 | -0.646122361 | 0.019881543 |
| B_dice | 512 | -0.042384682 | 0.250894694 | 0.585819852 | -0.649006394 | 0.020801860 |

A and B optimize different total objectives; their absolute totals are not directly comparable. Common terms are reported separately. These are equal-batch empirical risks over 32 frozen exposures repeated 16 times, not independent population measurements. Lower negative Dice means greater soft overlap. Reaching 512 updates is not a convergence certificate.

## What changed and what stayed matched

Both arms started from original seed42 and trained the full 34,546,498-parameter network. A exactly retained phase 13 manual smoothed-BCE plus unit-weight teacher-preservation KL. B added only unit-weight masked soft Dice, with hard supplied targets and the pinned upstream non-batch reduction/smoothing constant 1. This restores one upstream objective component, not the full pretraining recipe. Learning rate 1e-6, AdamW state continuity, clipping, input planes, normalization, schedule, XY augmentations and all support were matched. Objective changes necessarily alter gradients; this is a comparison of recipes under that fixed optimizer, not a proof about an objective at its optimum.

**The A control reproduced both the historical first-step state/logits and the complete phase 13 update-32 state and fixed-risk logits exactly.** The optimizer was not reset at 32. Thus the new A32-to-A512 comparison extends the same actual trajectory. Both 32- and 512-update states were saved; intermediate states were measured for training risk without development-driven selection.

Supervision is the original PHerc0139 supplied-supervision support excluding retired validation and unknown pixels. It contains 54,701 unique scheduled manual pixels and 618,208 visits per 32-step cycle. Annotation/teacher ancestry remains incomplete; “manual” is operational, not proven human-only truth. Historical seed43 arrays were read for unchanged identity checks and loss diagnostics only, not optimized. Seed42 KL operates only on manual support and is a regularizer, not independent truth. Paris data never entered gradients, teacher inputs or cutoff fitting. Frag6 and the other reserve parents were not accessed for this experiment.

The unchanged 17-plane input contract is retained; no claim is made that every available native plane is supplied. The unresolved optical–CT registration residual and physical ink-bearing depth remain unresolved. They do not invalidate this matched computational comparison, but they limit interpretation of reference agreement and especially any attribution of weak transfer to representation.

## Verification and resource accounting

Seven pre-execution numerical fixtures passed. Independent NumPy arithmetic checked 1408 training/baseline/fixed-risk rows and 11264 loss values, without replaying model autograd or optimizer transitions. Maximum per-term differences: `{"base": 1.197034763533722e-07, "control_objective": 1.197034763533722e-07, "dice_loss": 1.3523409403415343e-07, "full": 1.773932146731383e-07, "hard_base": 5.763880811526789e-08, "loss_D_M": 1.3877787807814457e-17, "loss_U": 1.197034763533722e-07, "probability_drift_manual": 2.7755575615628914e-17}`. Actual optimizer continuity, unsupported output gradients, finite parameters, inference state isolation and safe exact checkpoint loading are frozen worker checks. Historical update-32 state/logit equality is an additional strong control, not a proof of the new arm's physical validity.

Independent evaluation rebuilt eight canonical supports, checked 80 AP/AUROC values, every screen and matched contrast, and reconstructed 4718592 score pixels from 800 raw tiles. Maximum ranking discrepancy 1.11e-16; maximum independent sigmoid/Hann blend discrepancy 3.19e-07, below the predeclared 2e-6 tolerance. Ranking checks use the retained FP32 scores, so reconstructed floating-point differences are not silently used to alter ties.

Controller time 1195.120 seconds under 1,800 seconds. Training: 1,346 batch-two forwards, 1,024 backwards/updates, zero teacher forwards. Development: 800 single-tile forwards, zero baseline reruns. Counts refer to scientific calls and exclude tensor-only fixtures. Maximum training worker RSS 1573707776 bytes, under 4 GiB. All scientific workers completed once; no scientific retry or budget expansion.

The independent evaluation checker received a pre-output ordering correction: JSON sorts context dictionaries while failure lists retain source order. Version 2 compares identical failure records irrespective of list order, with unchanged numerical tolerances. Original checker and its freeze remain retained. This was identified before any checker execution or development result; no scientific code or score changed.

The scientific workers made no network requests. No paid GPU resource, R2 upload, publication, submission or outreach occurred. GPU spend remains £0 and the £10 total first-pilot cap remains unchanged. Repository source/tests/locks are unchanged. This local retention and saved startup configuration do not establish an external backup or a newly published/restored snapshot.
