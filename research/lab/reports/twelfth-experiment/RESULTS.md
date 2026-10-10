# Twelfth study: a finite first-step overshoot is demonstrated

The historical learning rate of **1e-4 increases both continuation objectives on their very first training batch**. At the predeclared 1e-6 diagnostic rate, the same initial gradients decrease both the smoothed base loss and the complete prediction-preservation loss. The optimizer moves in the correct first-order direction and matches the analytical fresh AdamW update within the frozen FP32 allowance. This identifies a concrete finite-step problem on that batch; it does not establish a better trained model or explain all 32-step transfer failures.

The saved-input audit also found no collapsed normalization scale or large scalar-conversion discrepancy. Retain the original seed42 model. No candidate checkpoint was saved or promoted.

## Same-batch causal comparison

Both objective families start from the identical published seed42 weights and exact historical first input batch. A uses smoothed manual BCE; B adds 0.25 times the separately normalized hard-pseudo BCE. Smoothing maps participating 0/1 targets to 0.25/0.75. The complete preservation objective adds corresponding KL(initializer teacher || student) terms with coefficient 1. At initialization this KL and its gradient are zero, so the first step is shared by the earlier smoothing and preservation variants.

Each rate has a fresh AdamW state. All parameters are reset and checked against the initializer before reusing the unchanged initial clipped gradient. There are no sequential updates across rates. Input/target/mask fingerprint and initial logits match the historical batch exactly; initial hard/smoothed losses and gradient norms reproduce their archived checks.

Loss changes below compare the actual post-update logits against the initializer on the **same batch**. Negative means improvement. These are absolute loss changes, not percentages or accuracy.

| One-step learning rate | A: base change | A: full change | B: base change | B: full change |
|---|---:|---:|---:|---:|
|1e-6|−0.006745|−0.005870|−0.007344|−0.006496|
|1e-5|−0.018163|+0.024002|−0.021086|+0.026476|
|1e-4, historical|+0.014596|+0.118165|+0.020080|+0.145220|

![Same-batch first-step loss changes](objective-audit/step-probe-figures/first-step-loss-change.png)

The middle rate illustrates why monitoring only the base loss would be inadequate for preservation: it improves that term while worsening the complete objective. At 1e-4, even the base loss worsens immediately. At 1e-6, both decrease. The six observations are not a learning-rate search winner, an optimality bound or a generalizing training recipe.

Every checked parameter moves with or orthogonal to the negative clipped gradient, and aggregate gradient-dot-update is negative in all six branches. Actual parameter-update norms are approximately 0.005864,0.05864 and0.5864 at the three rates. Global gradient clipping does not directly bound the resulting fresh Adam step: apart from epsilon, its coordinatewise first update is proportional to the sign of the clipped gradient. Thus a correctly implemented clip does not guarantee a suitably small prediction change.

See [step-probe/PROPOSAL.md](step-probe/PROPOSAL.md), the root's pre-execution [decision](STEP_PROBE_DECISION.md), frozen [plan](step-probe/plan.ready.json), and [run receipt](step-probe/run/run-results.json). There were exactly 8 actual model forwards, 2 backwards and 6 independent optimizer steps, taking 16.083 seconds of controller wall time. Peak worker RSS was 1,220,005,888 bytes. No additional teacher forward, checkpoint output, holdout inference, calibration, scientific retry or network operation occurred.

## What the retained trajectory establishes

The independent source and synthetic-mathematics review found no loss-sign, masking, support-denominator or update-order defect. The field `original_objective` already meant the same epsilon 0.5 smoothed base BCE optimized by the model. It was never hard-target BCE in this comparison. A hard-versus-smoothed target explanation therefore does not remove the earlier worsening.

The NumPy audit reconstructs all 32 saved pre-update batches from each preservation arm, their manual/pseudo supports, hard-class and region counts, and all five predeclared windows. For teacher probability q, smoothed target s and logit displacement dz:

`base change = KL(q || p) + mean[(q − s) * dz]`

`full change = 2 * KL(q || p) + mean[(q − s) * dz]`

These use separate support means and B's unchanged 0.25 pseudo coefficient. In every eight-step window, the signed linear term is negative but positive curvature outweighs it. Across all 32 snapshots, mean full-loss changes versus the same-input initializer are +0.068098 for A and +0.065789 for B. A has 31 positive full deltas; B has 30; both first snapshots are zero. In the last eight snapshots, base/full deltas are +0.005913/+0.052498 for A and +0.003659/+0.057529 for B.

Those observations describe an evolving model on a changing batch schedule. They are neither final-model risk nor immediate within-step effects. The new same-batch probe supplies only the missing first-step measurement. It cannot distinguish all later minibatch interference, momentum or representation effects. The mathematical free-output optimum `(s+q)/2` remains an oracle diagnostic; a shared network is not assumed able to realize it.

See [objective-audit/RESULTS.md](objective-audit/RESULTS.md). Independent reconstruction checks all 64 snapshots, 512 region/class strata and every fixed window using separately implemented algebra and D4 indexing. Ninth's original scalar losses remain receipt-based because its all-step logits were never retained.

## Input-domain evidence

The fixed seven-context audit covers both Frag4 regions, both PHerc0139 adaptation regions and three original calibration regions. All 159 stride64 tiles have positive robust-MAD scales with no fallback or entirely empty tile. The implementation's normalization trace matches the pinned normalizer exactly. All 50 Frag4 tiles also receive the predeclared 257-fold scalar check: maximum absolute normalized-input difference is 4.2915344e-6, without a model-score invariance claim.

This weighs against a large scalar `/257` or collapsed-MAD explanation on these inputs. It does not erase actual acquisition differences: 54 versus 78 keV, fractional versus integer pooled intensities, slightly different nominal spacing, boundary/support occupancy and depth/XY texture. The first Frag4 context includes 10.58% zero input values over the whole patch domain, versus 0.165% in its fixed interior and 0.956% over supplied surface support; its MAD does not collapse. Such descriptive differences do not establish their causal effect on predictions or validate optical registration.

The 209 actual normalization calls took 7.364 seconds, with 171,716,608 bytes peak RSS. No ink-label values, model scores or checkpoints entered this input audit. See [input-audit/RESULTS.md](input-audit/RESULTS.md) and its compact seven-context table. Overlapping tiles and multiple contexts from the same specimen are not independent samples.

## Independent verification

The final [independent trajectory reconstruction](numeric-review/completed-review-v2/independent-verification.json) passes 14,841 scalar comparisons across64 batches, 512 region/class strata and all fixed windows. It uses independent log-probability/entropy algebra and coordinate-index augmentation, with explicit FP32 bit checks. Maximum discrepancy against the original stored scalar objectives is 1.247e-7; the float64 KL reconstruction differs by at most 2.776e-17. An additive checker revision strengthened signed-zero bit identity; the first checker and receipt remain preserved, with unchanged scientific numbers.

The [independent first-step review](loss-review/STEP_OUTPUT_REVIEW.md) reconstructs all six branches from raw saved logits/masks and checks 1,584 parameter-summary/probe rows. Maximum BCE/scalar difference is 1.021e-7 and stable-change error 2.776e-17. Full-coordinate Adam agreement is established by the reviewed worker's checks; the reviewer independently verifies summaries and one retained coordinate per tensor, not unavailable full gradient/parameter snapshots. Its initial receipt-serialization failure and corrected receipt are preserved; no scientific execution or tolerance was changed.

Root inspected both numerical figures for readable axes, legends and interpretation. The plots use retained numbers only, with no model calls or chosen optimum. Prior first–eleventh evidence remains immutable; a separate final retained-state manifest and verifier cover the completed phase.

## Scientific decision and scope

Do not scale the old 1e-4 continuation recipe to a GPU or add more loss coefficients to compensate for its immediate overshoot. The next useful control must measure a fixed training objective across updates and establish continuation stability before transfer or novelty claims. A small first step alone does not settle long-run optimization; proposed follow-up is in [NEXT_STUDY_DRAFT.md](NEXT_STUDY_DRAFT.md).

Frag4 is seen diagnostic evidence and remains excluded from training, pseudo targets and calibration across aliases. Its supplied optical-derived labels still carry the registration and historical-exposure qualifications from the eleventh study. The original PHerc0814 regions remain reused development diagnostics. No independent accuracy, first-in-class model, authentic letter, transcription or translation has been established.

All previous frozen evidence and tracked repository files are preserved. Work used local retained data and CPU only: paid GPU spend remains £0, with the £10 TOTAL pilot cap unchanged. No R2 upload, public release, prize submission, outreach, credential import or terms acceptance occurred. Saving environment guidance remains a draft update, not a backup or verified restoration.
