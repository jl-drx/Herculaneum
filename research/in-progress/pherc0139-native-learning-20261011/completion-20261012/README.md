# Native PHerc0139 learning: completed and verified

Completed 12 October 2026 at **14:07:51 UTC**. All nine fits reached 16,384
updates, the joint endpoints were sealed, and the producer finished its locked
evaluation. **The small spatial learner fits its training annotations very well
but does not meet the cross-winding transfer criterion.** The experiment is
complete; no continuation or replacement endpoint was selected.

## Frozen result

AP is average precision against the supplied native annotations under their
supervision masks. CNN means are means of the three individual seed APs, not an
ensemble score. The simple baseline is the fitted 29-parameter depth-profile
logistic model.

| Held-out winding | Constant AP | Profile AP | CNN mean AP | CNN seed range | Mean gain vs profile |
|---|---:|---:|---:|---:|---:|
| w035 | 0.313021 | 0.347319 | 0.328566 | 0.315964–0.335861 | **−0.018753** |
| w039 | 0.319068 | 0.327693 | 0.330473 | 0.316303–0.349532 | **+0.002780** |
| w044 | 0.379561 | 0.421842 | 0.398396 | 0.377380–0.416387 | **−0.023446** |

The descriptive equal-winding mean gain is **−0.013140 AP**. No winding passes
the required mean +0.02 margin with every seed ahead of the profile baseline.
Training-fit adequacy passes for all nine models: selected training AP ranges
from 0.982938 to 0.999056, with training BCE 81.4–95.6% below the training
constant's entropy. The annotation-negative mean-response guard also passes in
all folds. Those cores are not authenticated blank papyrus.

That guard should not obscure the probability failure. Held-out CNN BCE ranges
from **1.881682 to 3.309420**, against profile BCE **0.623549–0.667514**. CNN AUC
is only 0.500293–0.540627. The networks are making confident mistakes on excluded
tissue. A monotonic probability calibration could change BCE but cannot repair
the weak AP/AUC ranking. No calibration was fitted in this review.

Eight fits terminated `ceiling_still_improving`; w035/seed17 terminated
`ceiling_without_confirmed_plateau`. None reached the protocol's confirmed
minimum-learning-rate plateau. The exact frozen routing decision remains:

`optimization_not_demonstrated_saturated_no_native_signal_null`

In plain language: **advancement failed, but optimization saturation was not
established, so the result cannot support a null claim about recoverable native
CT signal.** Strong training fit does not establish physical ink accuracy or
prove that another architecture, training corpus or supervision target cannot
generalize. The folds reuse one physical scroll; three seeds are not independent
specimens or a population confidence interval.

## Verification and preservation

The previously qualified read-only verifier ran only after the producer sealed
all results. It verified the unchanged GO, all nine histories and selected
endpoints, checkpoint/map bindings, **1,093 source files**, **15 saved maps** and
**1,325 core/model metric rows**. Independent metric arithmetic reproduced the
reported values and decision with no neural inference or changed thresholds.

- [VERIFIED_RESULTS.json](VERIFIED_RESULTS.json): verified gates and all fold summaries.
- [POOLED_METRICS.csv](POOLED_METRICS.csv): constant, profile and every CNN seed, including AUC/BCE.
- [VERIFIED_RESULTS.cores.csv](VERIFIED_RESULTS.cores.csv): all scored core/model rows.
- [TRAINING_HISTORIES.json](TRAINING_HISTORIES.json): all 585 training checks, including step zero.
- [FINAL_STATUS.json](FINAL_STATUS.json): producer completed and final private backup pushed.

The GO is `2c20b4491a2c94fdc4c479a84a6ab85f309673a113cfdd8088b30fb7df6bf61c`;
the completed result SHA-256 is
`ce4dd4af21403e3e608e6f79072705ab9ea97c8e7ca56379566595d90955e170`.
The final checkpoint branch commit is
`26c6d8399ff5fc0a4bca9b22afdad4262a2dc322`, with 172 generations and 516 files.
The final 78.2 MiB generation was freshly downloaded into separate LFS storage.
[Restoration](FINAL_RESTORE_VERIFICATION.json) verified all 172 archives and 516
files, using local retained bodies for earlier generations. The verifier then
[reproduced the same report](RESTORED_RESULTS_VERIFIED.json) from the restored
run. This was a same-host restoration, not an independent-device replay. CSV
line endings are normalized to LF for Git; parsed values are unchanged.
See the existing [runbook](../RUNBOOK.md) for selective input and checkpoint
restoration. The old transient backup error is retained historically; the final
successful push is later than that error. No GPU or Actions runner was used.

## Research decision

Do not extend these fits or retrospectively choose an earlier checkpoint using
these held-out results. Earlier checkpoints might transfer differently; this
study did not test that hypothesis. Future training needs a separately reserved
development criterion for generalization-aware endpoint selection, then fresh
confirmation. The current endpoint was intentionally training-only.

The immediate priority remains the [strong native-model comparison](../../../planning/native-common-support-20261012/README.md)
and then a matched broader-coverage training pair. This tests a materially
different hypothesis from prolonging a small model on two windings. Keep original
source exposure, annotation tier and compute controlled. Carry the reference
uncertainty explicitly; do not return to failed local depth/fusion adapters or
declare coarse CT unreadable. Adequately designed GPU experiments remain eligible
for a concrete budget/approval proposal despite this CPU negative result.

Geometry work proceeds independently. The [CT reference follow-up](../../geometry-reference-20261012/README.md)
has provisional PHerc0800 continuation candidates, rejected first PHerc0814
paths, and a qualified source/input footprint for future m7/F0 execution. Those
are reference-development results, not new sheet-recovery or ink-accuracy gains.
