# Plinius working series: conservative continuation control

Recorded before new scientific array evaluation or model execution on 8 October 2026. The user authorized further CPU tests before renting a GPU and suggested **Plinius** as the model name. Use Plinius as a provisional research-series name; these controls remain adaptations of the credited upstream seed42 model, not a newly demonstrated architecture or novel model. Historical study IDs and artifact paths stay unchanged.

The restored twelfth manifest passed: SHA256 `6a212fe4a2c43962c06c140f49380591ea1c6900b7de6a93217efa639deb7d46`, 1,986 records and 3,397,420,887 bytes. Tracked repository commit remains `49fabf3e890d5b880f391b27d581f1e7a1ebdbcc`, with only the preserved untracked `.codex/`. All earlier frozen studies remain immutable.

## Question and matched controls

Does reducing the finite AdamW step resolve immediate overshoot and improve the fixed adaptation objective after 32 updates? If an arm improves that objective, does its final state preserve the reference predictions sufficiently to justify further research?

Run A_preserve and B_preserve from identical seed42 initialization, using the exact tenth inputs, two PHerc0139 regions, hard targets, manual/pseudo participation, normalization, ordered 32-batch schedule, D4 transforms and objective. The sole training hyperparameter change is learning rate 1e-4 to **1e-6**, motivated by the completed first-step diagnostic. This is a predeclared conservative control, not an optimal rate or an open search. No alternative rates, warmup, scheduler, extra seeds or loss coefficients are allowed in this study.

The base objective is epsilon 0.5 smoothed manual BCE U for A, and U+0.25P for B, with separate participant means. Full objectives add temperature-one KL(initializer teacher || student), coefficient 1, on the matching supports. Pseudo targets remain the original hard seed43 map, distinct from the seed42 preservation probabilities. Unknown output locations never become negatives. Full preservation gradients operate on all later updates; the zero first-step preservation gradient does not justify optimizing base BCE alone thereafter.

Keep fresh AdamW per arm with betas 0.9/0.999, epsilon 1e-8, weight decay 0, global gradient clipping 1 and exactly 32 sequential updates. No early stopping on scientific results. Finite/runtime/identity failures stop the run and remain preserved; no automatic retry or relaxed tolerance. Save only each final raw state dictionary, validating finite tensors, exact values/aliases and strict safe reload. No selected intermediate state or saved optimizer.

## Cached teacher and measurements

Reuse the retained tenth teacher logits on all 32 exact batches. Both historical arms have identical teacher values, and their step logs retain normalized input/target/mask hashes for every batch. Recreate and check each batch against both logs before using its cache row. Bind the teacher cache to its original plan, unchanged initializer and runtime. Baseline fixed risk is explicitly **cache-derived initializer risk**, not a new model forward. Check the first student logits and gradient against historical evidence and the first post-update logits/state against the twelfth 1e-6 branch.

Record pre-update and immediate post-update student logits on each update's own batch. At completed updates 1, 8, 16, 32, evaluate all 32 frozen schedule batches on the same current state. Average per-batch supported losses equally; overlapping tiles and repeated exposures retain their actual multiplicities. This defines an empirical adaptation risk, not an independent population sample or a pooled-pixel objective. Record U, P, weighted KL, base/full loss, hard BCE as secondary, and prediction drift. Report every checkpoint and update; do not choose the best checkpoint.

Use no_grad for measurements, with unchanged deterministic train-mode behavior, no mutable/running buffers or active dropout. Verify state and RNG are unchanged by fixed-risk evaluation and that gradient/optimizer state is not mutated. The fixed-risk row corresponding to the current update must reproduce its immediate post-update output. Retain all raw logits and batch fingerprints for independent NumPy reconstruction. Gradient norms, update norms and state fingerprints accompany the losses.

Each arm performs 32 pre-update, 32 immediate-post and 128 fixed-risk batch-two forwards: **192 forwards and 32 backwards/updates**. Both arms total 384 forwards, 64 backwards and 64 updates, with no actual teacher forwards. The redundant current-batch fixed-risk call is intentional for a simple independent consistency check. Four CPU threads, Python 3.12.14, torch 2.12.0+cpu and NumPy 2.5.3 remain pinned. Cap each worker at 180 seconds, controller 420 seconds, CPU 600 seconds per worker, RSS 4 GiB, address space 12 GiB and combined outputs 450 MiB. No network or paid compute.

Compare old and new *online pre-update* trajectories on the same schedule where appropriate. The old 1e-4 study did not retain final fixed-risk measurements; do not invent them or treat its last online window as final risk. The primary new fixed-risk comparison is against the same cache-derived initializer.

## Conditional retention stage

An arm is eligible only if its final step 32 fixed **base and full** risks each decrease by at least 0.0001 versus their respective initializer risks. This is a frozen computational screen above rounding noise, not a statistical or biological significance threshold. Record both pass and fail arms. Scientific failure is a result; it does not trigger another rate or altered cutoff.

After independent training-output verification, root may freeze the retention plan for eligible final states only. Use the two unchanged PHerc0814 diagnostic contexts and two unchanged Frag4/PHercParis1Fr39 contexts, exact forward layer selection, masks and score blending from their original studies. At most 2 states × 4 contexts × 25 tiles = 200 additional tile forwards. No new data, calibration, depth reversal, orientation choice, baseline execution or checkpoint selection. Reuse retained baseline and matched old 1e-4 final-score maps for comparisons. Use the original seed42 cutoff 0.7558947205543518 uniformly for secondary threshold metrics; never fit a new cutoff here.

Primary retention results are per-context AUC/AP and equal-context means separately for each specimen/domain, with comparisons to the frozen baseline and old matched continuation. For a descriptive stability screen, require each domain's mean AUC and AP to be no more than 0.01 below baseline, and each context's AUC and AP to be no more than 0.02 below baseline. Apply absolute metric differences without statistical-significance claims. These pragmatic tolerances are declared before new predictions and cannot be relaxed. Passing does not automatically promote a model or establish external generalization.

PHerc0814 and Frag4 are already seen development evidence, not prospective holdouts. Frag4 remains excluded from gradients, teacher targets and calibration across aliases. Preserve the optical-registration, annotation and historical-exposure qualifications from prior studies. Four contexts from two specimens are not four independent specimens. Do not infer authenticated ink, letters or translation from model-score agreement.

## Review and scope

Root owns scientific review and executes actual model work only after code/plan review. Agents may prepare code, source contracts and synthetic fixtures, then independently reconstruct saved numerical outputs when authorized. No duplicate model training by reviewers. Retention uses a separate frozen plan because checkpoint identities and eligibility are not available before training; its question, cases, measures and criteria are already fixed here.

Paid GPU spend remains £0 and the first-pilot cap remains £10 TOTAL including failures/retries and attributable overhead. No R2 upload, publication, submission, outreach, credential import or terms acceptance. CPU work needs no credentials. A future paid job still requires a useful frozen workload, pinned CUDA environment, scoped worker delivery, current all-in pricing and independent shutdown. Record outcomes and provenance even if the control fails; research progress is evidence, not a promise of a first-in-class model.
