# Independent review tools for the native-learning evaluation

**Ready for the eventual sealed results; no live held-out evaluation has been
read or verified yet.** The native worker continues its unchanged nine-fit
study. These separate tools do not amend its data, optimizer, stopping rules,
endpoint selection, metrics or practical advancement thresholds.

`verify_evaluation.py` is read-only with respect to the run and source packet.
It imports neither the training implementation nor Torch, deserializes no model
checkpoint, performs no neural inference, and makes no network requests. The
CLI is bound to the existing frozen GO SHA256:

`2c20b4491a2c94fdc4c479a84a6ab85f309673a113cfdd8088b30fb7df6bf61c`

## What it checks

Before any held-out map can be opened, the verifier checks the frozen code and
plan, every completed fit's training history, the earliest minimum-BCE endpoint,
learning-rate schedule, stopping classification, checkpoint hashes and bindings.
All nine completion records must match the joint ENDPOINTS seal. The producer
must also have finished RESULTS.json and all fifteen map receipts. Incomplete
training or evaluation returns an explicit waiting status with zero map bodies
opened; inconsistent or corrupted artifacts raise an error.

After that boundary it verifies all source-file hashes, physical fold identities,
ordered evaluation cores, annotation masks and counts, training populations,
baseline/normalization bindings, and each saved map's hash, shape, dtype and
finiteness. It independently recalculates every core and pooled AP, AUC, BCE,
prevalence, mean response and fixed-half response rate. AP uses grouped score
ties; AUC uses positive-negative rank counts. Both ranking metrics remain
undefined on single-class cores, which still contribute loss and response data.

The final summary is independently rebuilt from those recalculated values:
per-winding mean AP of the three individual seeds, each seed's improvement,
training-fit adequacy, and the supervised-pixel-weighted response on cores with
no positive supervised annotation. Unsupervised pixels never become negatives.
Such cores are not authenticated blank papyrus. Every gate and the final routing
decision must agree with the frozen producer, with no relaxed thresholds.

## Qualification and current live check

**46 synthetic checks passed.** The end-to-end fixture contains nine synthetic
completion records, fifteen maps and sixty core/model rows. The frozen producer
creates the reference metrics and decision from those synthetic saved maps;
neural construction and inference functions are replaced by errors. The new
verifier agrees using different metric arithmetic.

The fixtures also reject premature seals, wrong GO/seed/normalization bindings,
corrupted checkpoint or map bytes, wrong core ordering, duplicate IDs or result
receipts, fabricated metrics, missing/partial/extra maps, altered source labels,
wrong shapes/dtypes, nonfinite unmasked logits and path escapes. Other cases
check grouped ties, single-class handling, ignored out-of-mask ink, unequal-size
negative-core weighting, the distinction between mean seed AP and ensemble AP,
and the practical decision rules. `QUALIFICATION.json` binds the exact scripts
and frozen producer by hash. These are synthetic checks, not research outcomes.

The actual live preflight checked the completed first fit and confirmed its
endpoint and budget-limited stopping classification. It returned
`WAITING_FOR_TRAINING`, one of nine endpoints complete, with zero held-out map
bodies opened. Its selected endpoint is step 15,872; termination at step 16,384
does not establish a confirmed plateau. The status and backup observation in
`WORKER_STATUS.json` is timestamped, not a live heartbeat.

## Run from any restored copy

Use Python 3.12 and the NumPy 2.5.3 / Zarr 3.4.0 environment already qualified for
the study. No new packages were installed for this review. Restore the run and
source using the original [runbook](../../in-progress/pherc0139-native-learning-20261011/RUNBOOK.md).
No new LFS object or Actions job is needed for the review code itself.

Preflight is safe during training:

```bash
python research/planning/pherc0139-native-evaluation-review-20261011/verify_evaluation.py \
  --run /path/to/native-learning-run \
  --preflight-only \
  --output /path/to/review/preflight.json
```

After the native worker has completed its sealed evaluation:

```bash
python research/planning/pherc0139-native-evaluation-review-20261011/verify_evaluation.py \
  --run /path/to/native-learning-run \
  --source /path/to/native-reference \
  --output /path/to/review/verified-results.json
```

A final `PASS` writes a separate report and `.cores.csv`. Waiting statuses are
not a verification PASS. Use a new output filename on each invocation; the
checker refuses to overwrite an existing receipt or write inside the frozen
run. The review is explicitly invoked; no additional background watcher or
automatic post-completion review was launched.

The synthetic qualification is replayable with the existing study environment:

```bash
python research/planning/pherc0139-native-evaluation-review-20261011/qualify_review.py \
  --producer research/in-progress/pherc0139-native-learning-20261011/native_learning.py \
  --output /path/to/review/synthetic-qualification.json
```

The qualifier uses the machine's standard temporary directory and removes the
fixture afterward. It reads only the producer source/plan and synthetic fixture data.
It does not access the running study or its source packet.

## Scientific limits and next action

This verifies recorded artifacts, endpoint selection, metric arithmetic and
decisions. It does not independently rerun the CNN, regenerate the profile maps,
refit the convex baseline, recalculate normalization from CT, or reconstruct
training metrics from checkpoint tensors. Those operations are outside this
review's claim. Hash integrity is not proof of correct physical annotations.

When all nine fits and the producer evaluation finish, run the verifier, preserve
its report, inspect all folds/seeds/core results, and then review the next learned
comparison. Keep any ceiling-limited negative result classified as unresolved
optimization. A favorable annotation result supports designing broader paired
learning; the six-site sensitivity audit still requires a matched smoothing
baseline and prevents treating uncertain transfers as precise truth. GPU
readiness and the existing £10 total pilot conditions remain separate decisions.

No additional research run, new model inference, paid GPU, external VM assignment,
public publication, PR or GitHub Actions runner was started for this work.
