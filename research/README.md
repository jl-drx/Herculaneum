# Plinius research: continue on another machine

**For a usable Linux workspace, start with the [Git LFS snapshot and restore instructions](workspace/README.md).** This adds the retained arrays, historical checkpoints, prepared PHerc343 inputs and completed maps omitted from the original text-only export. Downloads are selectable, checked against SHA256 manifests and restorable to a chosen directory.

This directory preserves the selected research history through experiment 49. Start with the [experiment index](EXPERIMENTS.md), [latest results](lab/reports/fortyninth-experiment/RESULTS.md), and [next centre-depth target run](lab/reports/fortyninth-experiment/NEXT_STUDY_DRAFT.md).

The [completed PHerc343 four-model reproduction](lab/reports/fortysixth-experiment/RESULTS.md) closely matches the published display (RGB MAE 0.0181/255; yellow-display IoU 99.83%). All eight maps and all 18 candidate regions passed the recorded checks. Phase46 is now formally sealed; the separate [completed-results download](in-progress/pherc343-reproduction/completion-20261010/README.md) remains unchanged. This is a published-image positive control, not ink accuracy or authenticated letters.

[Experiment47](lab/reports/fortyseventh-experiment/RESULTS.md) completed the four predetermined tiling origins on one exposed seed42 region. The original-grid baseline matched all 262,144 saved pixels exactly; shifted grids had top-10% response Dice of 0.480–0.626 against it. All four weighted blends and all six pair comparisons were independently recomputed from saved outputs. An initial recorder bug was caught by the baseline gate and is preserved alongside the corrected run. The diagnostic is closed: no phase was promoted, and no further offset search is proposed.

[Experiment48](lab/reports/fortyeighth-experiment/RESULTS.md) completed the full-resolution PHerc0800 support check across all five larger meshes and all 40 earlier separated components. No 128×128 core passed the frozen conservative normal/slab/exposure/context gate; the largest square was 69×69. Sixteen archived point checks and independent context calculations passed. This does not establish that every exact patch schedule is infeasible or that ink is absent. A [private spare-VM task](handoffs/pherc0800-support-20261010/START_HERE.md) asks for an independent geometry implementation and an exact single-grid context diagnostic. The main next task prepares explicitly labelled development transfer on the previously inspected larger target field; fresh-coverage claims remain separate. No new target inference or paid GPU work ran.

[Experiment49](lab/reports/fortyninth-experiment/RESULTS.md) passes fixed PHerc0800 original-mesh rendering and four-model input qualification (20 tile forwards, no full target maps). Both rendering comparisons agree within one intensity level; all model batch checks pass. The existing field has79.50% joint geometric context under the strict mask rule. A centre-depth two-phase forecast is2.60 CPU hours per input order, with768MiB conservative source work. The [43.6MiB qualification supplement](workspace/PHASE49.md) includes verified saved-output replay. The next task is the checkpointed centre-depth development transfer; the user’s separate VM geometry check is proceeding.

[Preparation during the active run](planning/pherc343-followup-20261009/README.md) records the result-review checklist, known PHerc0800 exposure exclusions and the compute implications of a full depth sweep. It does not change the executing study or certify a fresh assessment field.

[Sam's independent six-map review and our assessment](reviews/sam-pherc343-20261010/README.md) preserve the supplied report, all 18 candidate panels and a separate saved-map arithmetic audit: 1,594 numeric comparisons passed. Padding sensitivity is broader than the small ownership-boundary effect; neither establishes ink accuracy. At the dated 10 October observation, Hecate plain was complete and Hecate padded was running. At that time the four-model assessment remained pending; the subsequent completed assessment is below.

[Sam's follow-up and the completed four-model assessment](reviews/sam-pherc343-followup-20261010/README.md) supersede that earlier pending status. All25,070 replayed numeric values match, all21 synthetic gates pass, and the final eight maps and ownership merges pass the supplied checker. Whole-crop KD-tree verification and independent ink/CT component checks pass. All18 regions were reviewed with unchanged contrast. That bounded grid-origin comparison is now completed in experiment47; prospective PHerc0800 field admission remains next.

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

For an existing local branch, switch to it and use `git pull --ff-only`. This branch includes the preceding research history, experiments 40–49, and the tested experiment-39 replay fixture. Check out this branch to obtain the complete selected export regardless of pull-request merge status.

## What is portable

`lab/` contains byte-identical selected scripts, protocols, reports, configuration records, source notices and the research register from the cloud lab. [GIT_EXPORT_MANIFEST.json](GIT_EXPORT_MANIFEST.json) identifies every exported original file. All 49 phase reports are included. This is an archival source/evidence export, not an installation of every old experiment: historical scripts and pins retain original absolute paths and may require missing inputs, packages and checkpoints. Do not run their write-once workers against sealed evidence directories.

Experiment 39 has a 12 MB replay capsule, described by [CAPSULE.json](CAPSULE.json). It contains the 24 retained score maps, labels/masks, frozen comparison code and reference evidence needed to reproduce experiment 39. It has no scan volumes or model weights. This single bounded fixture is committed under `research/replay/` in the private research branch. Release-asset upload was rejected by this environment's authentication; the empty draft release was removed. R2 remains the intended store for larger outputs and checkpoints.

Experiments 40–41's arrays, training logits and model checkpoints are not in the phase39 replay capsule. They are now covered by the separately indexed `history` group in the Git LFS workspace snapshot, subject to its explicit exclusion list. Their original versioned plans and receipts remain the scientific record.

Experiment 42 includes six compact scientific review figures and acquisition/verification code and receipts. Its text-only export omitted raw chunks, assembled volumes, score cubes, view arrays and original large figures. The new workspace manifest indexes the separately bundled retained files; the original selected export alone is not a standalone replay of the CT diagnostic.

Experiment 43 includes the source-search receipts, target catalog extracts, selected byte-identical release metadata, findings and an unsent request. No profilometry image data was acquired or bundled. Unexported raw responses remain identified by the request ledger; no complete remote source-audit replay is claimed.

Experiment 44 includes every per-arm metric result, source receipts and compact reference/learning figures. Bulk ranges, crop arrays, score maps and checkpoints remain local. [Input recovery instructions](lab/reports/fortyfourth-experiment/PORTABILITY.md) and [materialize_phase44.py](tools/materialize_phase44.py) reconstruct the nine fixed model-input arrays from six pinned public ranges. A cached-range reconstruction passed byte-for-byte; a fresh-machine/network replay and full training replay were not performed. The original source data and derivative figures are CC BY-NC 4.0, with license and attribution retained. [Checkpoint tensor fingerprints](PHASE44_CHECKPOINT_TENSOR_HASHES.json) and [their verifier](tools/verify_phase44_checkpoint.py) let a regenerated model be compared bit-for-bit without downloading the original weight container. This checks weights only; optimizer state still requires the prescribed deterministic replay.

Experiment45 includes the continuation, checkpoint-panel and geometry results, pinned PHerc343 recipe code, Hecate source/card, source audit and decision record. Its [portability notes](lab/reports/fortyfifth-experiment/PORTABILITY.md) distinguish exact local replay from unperformed full PHerc343 inference and fresh-machine restoration. Selected shell recipes and model TSV pins are now included by the incremental exporter. The new workspace snapshot separately indexes the retained bulk maps, chunks and checkpoints.

The original cloud lab remains the live research location. The new Git LFS snapshot backs up its explicitly indexed retained files, including bulk research artifacts; it does not copy the whole filesystem, current raw CT cache, installed software or live processes. Its manifest and validation record define coverage. Earlier per-study portability notes describe the narrower export available at the time and must be read alongside this new snapshot. R2 worker credential delivery remains unverified; credentials are excluded from the export.

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
