# Plinius research: continue on another machine

This directory preserves the research history through experiment 45. Start with the [experiment index](EXPERIMENTS.md), [latest results](lab/reports/fortyfifth-experiment/RESULTS.md), [critical strategy review](lab/reports/fortyfifth-experiment/STRATEGY_REVIEW.md), and [next experiment](lab/reports/fortyfifth-experiment/NEXT_STUDY_DRAFT.md).

The approved [PHerc343 reproduction is now in progress](in-progress/pherc343-reproduction/STATUS.md). The original renderer binary was recovered and verified, and all four models passed strict synthetic CPU checks in a separate pinned runtime. Full CT streaming was interrupted by repeated cloud-tunnel HTTP503 errors; the partial render has not passed the published decoded hashes, and scientific ink inference has not started. The selected working snapshot is separate from the sealed45-study archive and does not add a completed experiment. No paid resources were launched.

The current direction returns to the sealed scroll: reproduce the published PHerc343 native-coarse positive control, then transfer the fixed recipe to PHerc0800. PHerc0139's paired acquisitions provide the supporting learning benchmark. Further profilometry studies and the unsent Paris4 reference request are shelved. Experiment45 closes one native profilometry trajectory at8,192 updates, measures released-checkpoint variability on the reused eight-region panel, and measures sparse PHerc0800 mesh-to-M7 disagreement. A CPU AP gain is no longer a universal prerequisite for GPU work; experiment/data quality, measured runtime, artifact delivery and the £10 total cap govern admission. Older proposals below are historical context, superseded where inconsistent with this decision.

Experiment 42 acquired six pairs of Paris4 CT blocks using 192 MiB of CT payload. Four local correspondence screens failed and two regions lacked the full frozen diagnostic window; no independent ink-positive/non-ink-papyrus reference was established. No learning arm was admitted or run. That study proposed a reference-first native/synthetic positive control, with real coarse learning conditional on verified correspondence. The paired-learning hypothesis remains open. Experiment 41's supervision control remains a tradeoff with no advancement. Seed42 remains the baseline and GPU work remains deferred. All six new regions are reference-review exposed; the existing eight evaluation regions remain reused development. No reading or translation is claimed.

Experiment 43 completed a bounded search for Paris4/PHerc500P2 reference packets; neither CT reference qualified. It found an official profilometry release with paired-file metadata for 14 regions, plus an explicit unresolved sampling discrepancy: 0.68793625 µm/pixel in the release versus 0.34 µm in the paper. Actual image/reference QC was deferred to experiment 44. The proposed supporting CPU control uses relative degradation factors and makes no CT or physical-resolution-cutoff claim. A [specific Paris4 reference request](lab/reports/fortythird-experiment/REFERENCE_REQUEST_DRAFT.md) is prepared but unsent. No media or model execution occurred in phase43.

Experiment 44 acquired three paired profilometry crops with verified raw-height coordinates and source labels, then completed six matched native/4× CPU learners (12,288 updates). The native positive-control training-fit gate failed, while some training fits were still improving. The result is inconclusive about resolution-related information loss; no model or teacher was promoted. It supports a bounded native-only budget continuation before more complex learning. This measures released optical-annotation agreement, not CT transfer or chemically authenticated ink. Source sampling calibration remains unresolved.

## Get the research branch

From an existing clone with access to this private repository:

```bash
git fetch origin
git switch --track origin/codex/plinius-pherc343-reproduction
```

For an existing local branch, switch to it and use `git pull --ff-only`. This branch includes the preceding research history, experiments 40–45, and the tested experiment-39 replay fixture. Check out this branch to obtain the complete selected export regardless of pull-request merge status.

## What is portable

`lab/` contains byte-identical selected scripts, protocols, reports, configuration records, source notices and the research register from the cloud lab. [GIT_EXPORT_MANIFEST.json](GIT_EXPORT_MANIFEST.json) identifies every exported original file. All 45 phase reports are included. This is an archival source/evidence export, not an installation of every old experiment: historical scripts and pins retain original absolute paths and may require missing inputs, packages and checkpoints. Do not run their write-once workers against sealed evidence directories.

Experiment 39 has a 12 MB replay capsule, described by [CAPSULE.json](CAPSULE.json). It contains the 24 retained score maps, labels/masks, frozen comparison code and reference evidence needed to reproduce experiment 39. It has no scan volumes or model weights. This single bounded fixture is committed under `research/replay/` in the private research branch. Release-asset upload was rejected by this environment's authentication; the empty draft release was removed. R2 remains the intended store for larger outputs and checkpoints.

Experiments 40–41's arrays, training logits and model checkpoints are **not** in the replay capsule or Git. Their versioned plans and receipts identify the retained inputs and outputs; reproducing training requires those separate artifacts.

Experiment 42 includes six compact scientific review figures and acquisition/verification code and receipts. Its raw chunks, assembled volumes, score cubes, view arrays and original large figures remain local. The selected export is not a standalone replay of the CT diagnostic.

Experiment 43 includes the source-search receipts, target catalog extracts, selected byte-identical release metadata, findings and an unsent request. No profilometry image data was acquired or bundled. Unexported raw responses remain identified by the request ledger; no complete remote source-audit replay is claimed.

Experiment 44 includes every per-arm metric result, source receipts and compact reference/learning figures. Bulk ranges, crop arrays, score maps and checkpoints remain local. [Input recovery instructions](lab/reports/fortyfourth-experiment/PORTABILITY.md) and [materialize_phase44.py](tools/materialize_phase44.py) reconstruct the nine fixed model-input arrays from six pinned public ranges. A cached-range reconstruction passed byte-for-byte; a fresh-machine/network replay and full training replay were not performed. The original source data and derivative figures are CC BY-NC 4.0, with license and attribution retained. [Checkpoint tensor fingerprints](PHASE44_CHECKPOINT_TENSOR_HASHES.json) and [their verifier](tools/verify_phase44_checkpoint.py) let a regenerated model be compared bit-for-bit without downloading the original weight container. This checks weights only; optimizer state still requires the prescribed deterministic replay.

Experiment45 includes the continuation, checkpoint-panel and geometry results, pinned PHerc343 recipe code, Hecate source/card, source audit and decision record. Its [portability notes](lab/reports/fortyfifth-experiment/PORTABILITY.md) distinguish exact local replay from unperformed full PHerc343 inference and fresh-machine restoration. Selected shell recipes and model TSV pins are now included by the incremental exporter. Bulk maps, chunks and checkpoints remain local.

The full manifest-covered evidence chain is still in the original cloud workspace. It has **not** all been transferred to GitHub or R2. No bulk-backup or fresh-machine restoration claim follows from this export. Future training may need a separate transfer of retained inputs/checkpoints or verified reacquisition. R2 worker credential delivery remains unverified; do not paste credentials into chat or commit them.

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

CPU monetary spend is unrestricted. The first GPU pilot remains capped at **£10 total**, including retries and attributable overhead. GPU work requires a concrete matched experiment, suitable data/reference, measured runtime and memory, verified artifact delivery, a pinned CUDA environment, current pricing and independent shutdown. A statistically convincing CPU AP gain is not a universal prerequisite. No paid job was launched in experiment45.
