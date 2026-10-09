# Plinius research: continue on another machine

This directory preserves the research history through experiment 41. Start with the [experiment index](EXPERIMENTS.md), [latest result](lab/reports/fortyfirst-experiment/RESULTS.md), [paired-acquisition assessment](lab/reports/fortyfirst-experiment/PAIRED_ACQUISITION_REVIEW.md), and [next study proposal](lab/reports/fortyfirst-experiment/NEXT_STUDY_DRAFT.md).

Experiment 41 removed only the new crop's supervised BCE gradient from the fixed 512-update phase40 recipe, preserving its inputs, original-source weighting and joint teacher KL. Paris AP fell from 0.317591 to 0.316034; worst AP retention improved while worst AUROC retention worsened. The prespecified mechanism result is a tradeoff, and advancement still fails. Seed42 remains the baseline and GPU work remains deferred. The main proposed direction is a native/synthetic-coarse/real-coarse acquisition comparison. Public Paris4 pair metadata, transforms and paired meshes are available; local registration and independent reference labels remain unverified. No paired learning has run. All current evaluation regions are reused development; no reading or translation is claimed.

## Get the research branch

From an existing clone with access to this private repository:

```bash
git fetch origin
git switch --track origin/codex/plinius-supervision-control
```

For an existing local branch, switch to it and use `git pull --ff-only`. This branch includes the preceding research history, experiments 40–41, and the tested experiment-39 replay fixture. Check out this branch to obtain the complete state regardless of pull-request merge status.

## What is portable

`lab/` contains byte-identical selected scripts, protocols, reports, configuration records, source notices and the research register from the cloud lab. [GIT_EXPORT_MANIFEST.json](GIT_EXPORT_MANIFEST.json) identifies every exported original file. All 41 experiment reports are included. This is an archival source/evidence export, not an installation of every old experiment: historical scripts and pins retain original absolute paths and may require missing inputs, packages and checkpoints. Do not run their write-once workers against sealed evidence directories.

Experiment 39 has a 12 MB replay capsule, described by [CAPSULE.json](CAPSULE.json). It contains the 24 retained score maps, labels/masks, frozen comparison code and reference evidence needed to reproduce experiment 39. It has no scan volumes or model weights. This single bounded fixture is committed under `research/replay/` in the private research branch. Release-asset upload was rejected by this environment's authentication; the empty draft release was removed. R2 remains the intended store for larger outputs and checkpoints.

Experiments 40–41's arrays, training logits and model checkpoints are **not** in the replay capsule or Git. Their versioned plans and receipts identify the retained inputs and outputs; reproducing training requires those separate artifacts.

The full approximately 9.7 GB manifest-covered evidence chain is still in the original cloud workspace. It has **not** all been transferred to GitHub or R2. No bulk-backup or fresh-machine restoration claim follows from this export. Future training may need a separate transfer of retained inputs/checkpoints or verified reacquisition. R2 worker credential delivery remains unverified; do not paste credentials into chat or commit them.

## Reproduce experiment 39

Use Python 3.12 and a new output directory. Only NumPy is needed; no Torch, CUDA, R2 credentials or model download is involved. Linux/macOS example, from the research-branch checkout:

```bash
python3.12 -m venv .venv-replay
.venv-replay/bin/python -m pip install --require-hashes -r research/requirements-replay.lock
.venv-replay/bin/python research/tools/replay_phase39.py --capsule research/replay/plinius-phase39-replay.tar.gz --destination /tmp/plinius-replay
```

On Windows, use the environment's `Scripts/python.exe` and suitable local paths. That platform has not been tested. The replay tool verifies the capsule SHA-256 and every extracted file, reconstructs the frozen comparison in the new directory, independently checks all eight supports and 64 ranking metrics, and compares the entire metric/decision payload to the retained result. Output must end with `status: PASS`; a digest mismatch stops execution. Reusing the same destination is deliberately rejected.

Validation includes a relocated replay using a clean Python environment with only the locked NumPy dependency. This tests relocation and dependency sufficiency on the current machine, not a separate physical machine. See [PORTABILITY_VALIDATION.json](PORTABILITY_VALIDATION.json) for the recorded scope.

## Continue research

Keep historical `lab/` bytes unchanged; create new work in a new experiment directory and bind its inputs and criteria before outputs. The [research charter](lab/RESEARCH_CHARTER.md) and [phase39 protocol](lab/reports/thirtyninth-experiment/PROTOCOL.md) record scientific limitations and decision rules. Inspect source/data terms individually: bundled upstream code and notices retain their original licensing; no blanket project or data license is asserted.

CPU monetary spend is unrestricted. The first GPU pilot remains capped at **£10 total**, including retries and attributable overhead. GPU work requires the established frozen-experiment, CUDA, current-price and independent-shutdown conditions and remains deferred until sufficient CPU evidence.
