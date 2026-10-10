# Plinius — Herculaneum research project

**Current update, 10 October 2026:** studies46–50 are complete and sealed. Study51
completed its 16-map PHerc0800 development-field comparison and review at fixed
display scale. No defensible lettering; no GPU spend. Start with the
[current research README](README.md) and [study51 completion report](in-progress/pherc0800-development-20261010/completion-20261010/RESULTS.md).
The cumulative seal remains at study50. The 9 October brief below is historical;
its active-study and pending-output statements are superseded by this update.

Project index and continuation brief, 9 October 2026. This document brings together the retained research record, working study and portable workspace. The linked repository is private and requires the owner's GitHub access.

## Purpose and direction

Develop models and techniques for recovering writing from carbonised Herculaneum papyri. Frontier research is the principal aim; prize participation is useful when compatible with that aim. No new reading or translation has yet been established by this project.

The current sequence is to reproduce the published PHerc343 native-coarse positive control, then design a prospectively fixed transfer to PHerc0800. PHerc0139's paired fine/native acquisitions are the supporting learning direction, subject to registration and reference qualification. Further profilometry studies and the unsent Paris4 reference request are shelved.

The last sealed research record is experiment 45 / register v41. Experiment 46 is the active, unsealed PHerc343 reproduction. Earlier plans are historical where superseded by the phase45 strategy review and the phase46 protocol.

## Where to find the work

Repository: [jl-drx/Herculaneum](https://github.com/jl-drx/Herculaneum). Current research branch: `codex/plinius-pherc343-reproduction`. [Draft PR #10](https://github.com/jl-drx/Herculaneum/pull/10) contains the current reproduction setup and portable workspace. Use this branch while the research changes remain unmerged.

| Material | Entry point |
| --- | --- |
| Research overview and continuation instructions | [Research README](https://github.com/jl-drx/Herculaneum/blob/codex/plinius-pherc343-reproduction/research/README.md) |
| All 45 completed study reports | [Experiment index](https://github.com/jl-drx/Herculaneum/blob/codex/plinius-pherc343-reproduction/research/EXPERIMENTS.md) |
| Research principles | [Research charter](https://github.com/jl-drx/Herculaneum/blob/codex/plinius-pherc343-reproduction/research/lab/RESEARCH_CHARTER.md) |
| Last sealed findings | [Experiment 45 results](https://github.com/jl-drx/Herculaneum/blob/codex/plinius-pherc343-reproduction/research/lab/reports/fortyfifth-experiment/RESULTS.md) |
| Current research rationale and prior art | [Strategy review](https://github.com/jl-drx/Herculaneum/blob/codex/plinius-pherc343-reproduction/research/lab/reports/fortyfifth-experiment/STRATEGY_REVIEW.md) |
| Active experiment design | [Phase46 protocol](https://github.com/jl-drx/Herculaneum/blob/codex/plinius-pherc343-reproduction/research/in-progress/pherc343-reproduction/PROTOCOL.md) |
| Rendering and inference qualification | [Phase46 status and recovery instructions](https://github.com/jl-drx/Herculaneum/blob/codex/plinius-pherc343-reproduction/research/in-progress/pherc343-reproduction/STATUS.md) |
| Preparation for result review and target transfer | [Follow-up planning](https://github.com/jl-drx/Herculaneum/blob/codex/plinius-pherc343-reproduction/research/planning/pherc343-followup-20261009/README.md) |
| Restore and run on another Linux device | [Workspace setup guide](https://github.com/jl-drx/Herculaneum/blob/codex/plinius-pherc343-reproduction/research/workspace/README.md) |
| Exact stored files, hashes and exclusions | [Workspace manifest](https://github.com/jl-drx/Herculaneum/blob/codex/plinius-pherc343-reproduction/research/workspace/MANIFEST.json) |
| Restore, download and runtime verification | [Portability validation](https://github.com/jl-drx/Herculaneum/blob/codex/plinius-pherc343-reproduction/research/workspace/VALIDATION.json) |

## Active study and dated observation

Phase46 uses the original rendering recipe, fixed d=+1 reading-side inference, three Villa models and Hecate, with plain and padded tile phases. Published ensemble statistics are fixed. Both primary decoded rendering checks passed; the whole-directory checksum differs and is documented. All four models passed real-input qualification. CPU fp32 execution and the rebuilt public dense_native checkpoint remain explicit differences from the published run.

At **23:33 BST on 9 October 2026**, the supervisor and inference worker were alive with a fresh heartbeat. Six Villa maps were complete. Hecate was at **3,704 of 13,348 positions** on its first map; its second map and final composition remained pending. This is a dated observation. Recheck actual processes, heartbeat and logs before treating the run as active. Computation completion still requires output verification and scientific image review before sealing the study.

The live lab is `/workspace/herculaneum-lab/reports/fortysixth-experiment`; the current runtime is `/workspace/.venvs/pherc343/bin/python`. Inspect `PIPELINE_STATUS.json` and its referenced log. Preserve the frozen executing scripts and inputs, and avoid duplicate workers. Completed map receipts support recovery; there is no within-map Hecate checkpoint.

## Data and another device

The Git LFS snapshot published with validation commit `f85c6ab579da054d3ca94c3b7e4b58699f7c6a3a` contains **26,417 files in 29 bundles**: 11.29 GB compressed and 15.03 GB expanded. It includes retained arrays, historical checkpoints, four current models, prepared PHerc343 inputs and six completed maps. The current-experiment subset expands to about 4 GB. The manifest defines the exact coverage.

Credentials, installed environments, the current raw CT cache and live process state are excluded. Hecate's pending outputs and final composition are absent from that snapshot. Historical scripts retain their original absolute paths; the new portable entry points accept a chosen lab directory.

The second device is Linux with a Xeon Platinum 8573C, nine CPU cores, 9 GB RAM, 30 GB storage and no GPU. Start with the current-experiment groups and one model job at a time. With Git LFS installed and GitHub access configured, clone without automatically downloading every bundle:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch codex/plinius-pherc343-reproduction https://github.com/jl-drx/Herculaneum.git
cd Herculaneum
git lfs install --local
python3 research/tools/workspace.py list
```

Then follow the workspace setup guide. The restore command checks archive and file hashes, rejects conflicting destination files and supports deleting each verified local download cache after extraction. A clean runtime, relocated seed42/Hecate real-tile checks, full current-subset readback and a fresh GitHub reference-bundle download passed on the cloud host. Execution on the second physical device remains untested.

## Research and spending constraints

- Previously inspected Paris regions are development/regression evidence. Released-checkpoint differences do not establish a calibrated seed-noise floor.
- Exact rendering reproduction does not verify optical alignment. Texture-depth proxies and M7 agreement do not independently establish the true ink surface; blanket recentering is unsupported by the sparse geometry audit.
- Phase46 is a bounded reproduction. New depth/model selection, PHerc0800 transfer and training require their own concrete designs.
- CPU monetary spend is unrestricted. GPU spend to date is £0; the first GPU pilot cap is **£10 total**, including failures, retries and attributable overhead. A CPU AP gain is not a universal admission requirement. A suitable experiment, measured runtime, pinned CUDA environment, current pricing, verified artifact delivery and independent shutdown remain required.
- R2 worker credential delivery remains unverified. Credential values belong in supported secret storage and are excluded from this index and GitHub. Historical credential readiness or expiry statements must be rechecked before use.
- Preserve source-specific licenses, attribution and data terms. Check applicable prize rules before submitting or making eligibility claims.

## Continue from this index

Read the research overview, phase45 strategy review and phase46 protocol first. Check the current Git branch and actual worker state. Review completed phase46 outputs against its frozen criteria before proposing target transfer. Keep new experiments in separate directories and preserve the sealed evidence. Use the workspace manifest and validation record to distinguish retained files from inputs that still need acquisition.
