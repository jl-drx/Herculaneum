# Native PHerc0139 learning: launched 11 October 2026

**Completed and verified 12 October:** [final results](completion-20261012/README.md).
All nine fits and fifteen evaluation maps are complete. Strong training fit did
not transfer consistently; the frozen advancement gate failed. The text below
is the preserved launch record, not the current worker status.

This is the substantive CPU learning comparison recommended after the native
cohort intake and soft-target controls. It is running, not a completed result.
The frozen scientific implementation is commit `b9d72e6`; GO was sealed at
09:32:27 UTC and the worker started at09:33:02 UTC. Read live process/receipt
status before assuming the original worker still exists.

Compare one training-prevalence constant, one convex 28-depth logistic profile,
and a fresh122,513-parameter spatial CNN. The latter has three initializations
in each of three physical winding folds. Full28-depth input and256-square
context, central128 output,2.83M supervised pixels in265 admitted cores. Exact
full-context overlap exclusions and held-out regions are unchanged.

The [protocol](PROTOCOL.md) fixes objective, sampling, normalization, optimization,
stopping and decision criteria. Each CNN receives at least4,096 and at most16,384
updates at batch4. Training BCE controls the LR schedule, stopping and endpoint;
held-out maps are locked until all9 training endpoints are committed. A ceiling
while improving is unresolved optimization, not evidence of absent CT signal.

The primary comparison is annotation AP against the profile baseline in each
winding, with individual seeds, training fit and supervised-negative response
guardrails. The practical0.02 AP margin is not an inferred significance threshold.
These are portions of one scroll with released annotations of limited historical
provenance. Neither training fit nor this evaluation authenticates physical ink.

Qualification:21 synthetic checks passed, including exact resumed Adam/model/RNG
behavior, mask semantics, final-plane use, fold separation, normalization, metric
ties, stopping and evaluation lock. The real loader verified1,093 input files,
all265 cores, their mask/class counts and every retained fold. See
[QUALIFICATION.json](QUALIFICATION.json) and
[DATA_QUALIFICATION.json](DATA_QUALIFICATION.json).

At09:40:44 UTC the first CNN committed256 updates,1024 patch presentations and
13,720,912 supervised pixel exposures;138 of139 training cores had appeared.
All three convex baselines had converged. Training BCE was falling; no held-out
predictions had been made. This is a launch observation, not a transfer result.

The sustained first-checkpoint rate was1.12 seconds/update including full
training checks, slower than the initial0.57-second synthetic benchmark. The
nominal4096-update floor alone implies about12 hours across nine fits, but the
LR reductions and plateau patience cannot finish before6144 updates even on
an entirely flat curve. The useful schedule-based forecast is therefore roughly
17–46 hours, likely about one to two days, depending on convergence and host
throughput. CPU is authorized; GPU spend remains£0.

Private code/protocol are on `codex/plinius-pherc343-reproduction`. Immutable
checkpoint generations go to `codex/plinius-native0139-learning-checkpoints`,
approximately every10 minutes. The [runbook](RUNBOOK.md) documents selective
source downloads, checkpoint restore and resume. Source bundles are separately
pinned in [INPUT_ARCHIVES.json](INPUT_ARCHIVES.json). A local checkpoint is saved
every256 updates; backups preserve the latest and best committed checkpoints.
They do not keep an environment alive or automatically restart a process.

The256-update checkpoint was freshly downloaded from private GitHub into empty
LFS storage and restored into a separate directory. All1,093 source pins verified
against the independently relocated input packet; model tensors, Adam state,
sampler state and training history matched the original exactly. See
[TRAINED_RESTORE_VERIFICATION.json](TRAINED_RESTORE_VERIFICATION.json). This
read-only real-checkpoint verification complements the synthetic resumed-update
test; it did not run additional real training or any held-out model prediction.

```bash
/workspace/.venvs/herculaneum/bin/python \
  research/in-progress/pherc0139-native-learning-20261011/status.py \
  --run /workspace/herculaneum-lab/reports/pherc0139-native-learning-20261011 \
  --backup-state /workspace/scratch/native0139-learning-backup
```

No external VM assignment, GitHub Actions runner, paid GPU, new PR or public
publication was needed. All held-out results remain pending at launch.
