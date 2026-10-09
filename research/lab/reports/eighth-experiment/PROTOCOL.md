# Eighth study: bounded continuation and regional operating-point transfer

Frozen scientific choices before fresh reference acquisition or any adaptation, 8 October 2026. The lead owns high-level review, following the user's clarification; specialist agents check data and implementation independently. The user authorized proceeding with the recommended research. This protocol authorizes the described unpaid CPU work only and preserves all earlier studies.

## Question and scope

Can a small continuation step preserve teacher-derived ranking while improving the transfer of a negative-score cutoff? Compare three matched training arms and the unchanged seed42 initializer. This is a one-seed, two-training-region feasibility/mechanism pilot, not a full reproduction of Russo's training, proof of a new method, independent ink-accuracy estimate or novel reading. A null result completes the experiment and does not authorize checkpoint or hyperparameter shopping.

## Physical roles and annotation identity

Adaptation uses only the two retained PHerc0139/w016 contexts from the seventh study. Both whole contexts are now adaptation support. Their historical public validation membership is retained in the audit but explicitly retired as a holdout for this study. Manual anchors use supplied supervision minus supplied validation, intersected with acquired source support. These public references have incomplete human/teacher ancestry; "manual" is the operational comparison name, not independently authenticated human-only truth.

Fixed pseudo labels come only from the retained seed43 score maps at raw score ≥0.5. Pseudo participation is acquired, finite evaluated source outside manual participation, including the retired validation area. Historical validation ink labels are not used as training targets or for calibration. This choice is declared before acquisition: excluding historical validation leaves only 72 teacher positives among 98,230 eligible pixels, whereas including it yields 31,953 positives among 237,746. The latter is a meaningful dense teacher-imitation pilot; it still does not authenticate teacher positives.

All PHerc0814 inputs are excluded from gradient updates. Calibration uses the previously used PHerc1667/w029 primary crop and the two old PHerc0814 primary regions. Fresh evaluation will use two PHerc0814 384×384 contexts, each with a fixed 64-pixel margin and at least a 128-pixel source-XY gap from the full old contexts and from the other fresh context. Exact selection is frozen separately from retained metadata. No qualifying public-validation proxy support exists under that gap rule, so evaluation participation uses supplied supervision explicitly; it is not relabeled public validation. Each fixed interior must contain at least 256 supplied positives and negatives before prediction.

Fresh contexts are withheld from our gradient updates, calibration and choices for this study. They remain within the initializer/teacher's known historical development corpus. Different source-XY contexts do not prove complete 3D sampling independence. Both new contexts are on one specimen, so no independent-specimen statistical inference is warranted. Earlier PHerc0814 use informed the hypothesis and calibration; the role ledger must make this visible.

## Matched continuation

Use the unchanged strict 34,546,498-parameter factory and retained seed42 checkpoint for every arm. Seed42 is selected for continuity, not seventh-panel rank. Teacher seed43 remains fixed and separate. Input remains 17×128×128 from planes [2,19) of the pooled fine scan, with unchanged reference robust-MAD normalization. No depth shift or native-domain comparison is introduced.

Use 32 updates per arm, batch size two with one patch from each adaptation region. The common eligible grid is 128-pixel tiles at stride 64, requiring at least 256 manual positives and 256 manual negatives in each tile. There are six eligible tiles per region. Freeze seeded tile permutations/cycles and XY dihedral augmentations once and reuse exactly across arms. Thus every arm gets the same 64 input patches, manual exposure, augmentation and update count. No empty manual steps are silently skipped.

Let U be mean BCE over all manual-participation pixels in the batch, P mean BCE over eligible teacher pixels, and R the equal mean of the four region×class manual BCE means. Use unsmoothed hard targets in all terms:

- A, manual-only continuation: U.
- B, hard-pseudo continuation: U + 0.25 P.
- C, region/class-balanced anchors: 0.5 U + 0.5 R + 0.25 P.

Each arm has total manual coefficient one; B/C have the same teacher targets, participation and coefficient. Equal coefficients/exposure do not imply equal gradients. This compares a candidate package; it does not isolate region balancing from class balancing. Unknown pixels never acquire background labels from zero fill. Teacher-negative targets are explicit model outputs, not supplied negatives.

Optimizer, learning rate, clipping, schedule bytes, code hashes, input hashes and exact CPU/output limits will be fixed in the execution plan before any update. No early stopping, best-checkpoint selection or tuning against fresh results. Retain only each arm's final checkpoint and the planned predictions. A failed run is preserved; no automatic retries.

## Calibration and evaluation

The primary calibration comparator is a single conservative global cutoff: for each of the three fixed development regions, select the lowest float32 cutoff satisfying empirical specificity ≥95%, excluding the complete crossing score tie; use the maximum of those three cutoffs. Apply the same rule separately to frozen seed42 and each final adapted model. This guarantees the bound on each development region only; evaluate achieved specificity on every fresh region. Retain the original PHerc1667-only rule as a secondary comparator.

This compares changed calibration support, not a monotonic remapping of scores. A strictly increasing global score transform followed by the same quantile rule cannot improve ranking or change threshold decisions, so it would not be an informative calibration baseline. The multi-region rule may coincide with the original cutoff; retain that outcome rather than forcing a different comparator.

Use the same frozen fresh primary masks for all four models, after intersecting acquired support and common finite evaluated predictions. Preserve whole-context outputs and diagnostics. Report per-context ROC AUC, average precision, class prevalence, raw score distributions, confusion counts, sensitivity and achieved specificity under both calibration rules. Aggregate contexts equally and report no pixel confidence intervals. Report calibration-region metrics separately from evaluation.

Predeclared signal for follow-up: C must achieve at least 95% specificity on both fresh contexts under the primary rule, improve equal-context sensitivity over B by at least 0.02, lose no more than 0.02 sensitivity on either context, and lose no more than 0.01 AUC or 0.02 AP on either context. Compare C with A and frozen seed42 under the same constraints; an outcome obtainable by manual-only continuation or frozen-model calibration does not establish value from the candidate package. The small step budget may yield little measurable separation. A promising signal requires later seeds, more physical regions and factorial controls; failure does not justify adapting this gate after viewing results.

## Resources and execution ownership

Reuse retained weights, teacher predictions and adaptation/development scan arrays. Acquire at most two new contexts with ≤40 MiB new scan response bodies and ≤2 MiB label response bodies under a frozen bounded request plan. Preserve source metadata checks, exact bytes/SHA receipts and known partial-read accounting limitations. No new dependencies or external accounts are needed.

Only the root executes acquisition, training and inference after receiving a no-further-edits freeze. Agents may add reviews without modifying executing inputs. Code receives meaningful mask, loss, schedule and cutoff checks before real training; no broad unrelated suites are required. Independent reconstruction must verify final source/mask assembly and reported evaluation arithmetic.

No GPU or R2 upload is required for this CPU pilot. Paid spend remains £0. The £10 total first-GPU cap, verified scoped worker delivery, pinned CUDA environment, current pricing and independent shutdown still govern any later paid launch. No publication, prize submission or outreach is part of this study. Actual source licenses remain binding while prizes remain secondary to research.
