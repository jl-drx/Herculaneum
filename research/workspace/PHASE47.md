# Four-origin PHerc343 replay and target preparation

Experiment47 is complete. The exact saved-baseline gate passed before the three
shifted origins ran. Response-location rankings changed materially in this one
exposed region; there is no accuracy comparison or newly authenticated letter.
[Results](../lab/reports/fortyseventh-experiment/RESULTS.md) and the
[next target admission task](../lab/reports/fortyseventh-experiment/NEXT_STUDY_DRAFT.md)
explain the decision. No target prediction or paid job was launched.

## Download and replay on another device

From the private research branch, with Git LFS and Python 3.12 available:

```bash
git lfs pull --include="research/workspace/bundles/pherc343-grid-20261010-001.tar.gz" --exclude=""
python3.12 -m venv .venv-grid-replay
.venv-grid-replay/bin/python -m pip install --require-hashes -r research/requirements-phase47-replay.lock
.venv-grid-replay/bin/python research/tools/replay_phase47.py \
  --bundle research/workspace/bundles/pherc343-grid-20261010-001.tar.gz \
  --destination /tmp/pherc343-grid-replay
```

Choose a new destination; the tool refuses to overwrite an existing directory.
Only NumPy and SciPy are required. No Torch, scan download, model weights, GPU or
cloud credentials are involved. The archive is **41,126,865 bytes** (39.2 MiB),
expanding to **84,254,532 bytes** across 208 files. Its SHA256 is
`ea5452475b5454f0a321e989c0dd96dceeb66eeb22babfacb29aee4468cda1c7`.
[PHASE47_BUNDLE.json](PHASE47_BUNDLE.json) lists every member, length and checksum.

The replay verifies all members, independently accumulates all four float32
weighted blends, recovers the original seed42 crop exactly, and recomputes all six
pairwise comparisons and all 18 pair/rank distance comparisons. It also checks
the retained grid-phase bins. All operations use saved outputs; raw CT and model
execution are not reverified by this small package. A fresh GitHub download into an empty LFS object store matched the archive hash
and length. A local relocated replay
passed, including a fresh environment containing only the two hash-locked
dependencies; a separate physical device has not yet been tested.

The archive retains the failed initial recorder attempt separately from the
corrected run, all probability tiles/sums, schedules, plans, logs, reference ROI,
full valid mask, figures, compact PHerc0800 geometry-screen outputs, and cumulative
seals. Raw target coordinate arrays remain in the historical workspace bundles.
The old workspace manifest and completed four-model results bundle are unchanged.
The target screen is a lead for full-support admission, not proof of fresh tissue.

## Cumulative retained-state anchors

The current lab has passed full readback of 25,861 records / 14,408,518,378 bytes.
[PHASE47_VALIDATION.json](PHASE47_VALIDATION.json) records scope and results.
The small replay bundle does not contain all these historical bytes; full-history
restoration requires the separately indexed workspace bundles and supplements.

- Phase46 seal: `c91572b8763e74331638468208a7d15a5d9f01e2c4e38bbc0b4891352cf0e271`.
- Phase47 seal: `0aef8d525d1140a04796268abb349740860f0416d7fa2ca185ce012786986532`.
- Current register: [v43](../lab/reports/research-register-v43.json), history through47.

For the retained original lab, the read-only check is:

```bash
/workspace/.venvs/herculaneum/bin/python -B \
  /workspace/herculaneum-lab/reports/fortyseventh-experiment/verify_retained_state.py \
  --expected-manifest-sha256 0aef8d525d1140a04796268abb349740860f0416d7fa2ca185ce012786986532
```

Do not run historical producers in sealed evidence directories. The next study
must use a new directory and freeze its support rules before target predictions.
