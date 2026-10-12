# Completed PHerc343 output supplement —10 October2026

The original selectable snapshot below remains unchanged. For the finished eight
maps, four-model composition, reference and valid mask, use the separate
[86.1MiB completed-results bundle](../in-progress/pherc343-reproduction/completion-20261010/README.md).
It has its own checksum manifest and tested standard-library restore command.

# Use the research workspace on another Linux machine

The latest [partial population checkpoint](../in-progress/native-coverage-admission-20261012/PORTABILITY.md) preserves 2,887 files, including completed CT downloads, native annotations, exact pending requests and offline verification, in a 490.9 MiB private archive. A separate 19.0 MiB addendum preserves 52 more complete CT chunks and the failed continuation records. The full intake is blocked by outbound S3 proxy errors; these checkpoints are resumable preparation, not an admitted training dataset. They add no new VM assignment or Actions run.

The [completed original-stream continuation](../in-progress/native-coverage-completion-20261012/PORTABILITY.md) prepares a cumulative update against that first archive, including the earlier 19.0 MiB addendum. It qualifies all 876 original candidates and retains the latest partial added-source cache. **The new update is local only: GitHub upload failed with HTTP 503.** Once pushed and verified, use the first archive plus the newest cumulative update; do not stack both addenda on one restore. No VM assignment or Actions run is required.

The current [native-coverage preparation supplement](../in-progress/native-coverage-preparation-20261012/PORTABILITY.md) contains bounded source data, mesh-based candidate support, real-model mechanics receipts and the CT corridor review in a 24.1 MiB private bundle. Its 308 files and numerical support calculation can be checked without models or inference. The completed [five-model comparison](../in-progress/native-common-comparison-20261012/PORTABILITY.md) has a separate saved-output review bundle. Neither changes the historical snapshot below.

The earlier [PHerc0139 native learning preparation](../planning/pherc0139-learning-readiness-20261011/PORTABILITY.md) is a separate selective supplement: soft-target controls, native annotations,265 fixed cores, conservative training/evaluation exclusions, and536 native CT chunks. Its analysis and CT archives are separate; no unrestricted LFS download or model weights are needed to replay it. This supplements the dated original workspace snapshot below.

This is a selectively restorable snapshot of the research lab, stored in this private GitHub repository using **Git LFS**. It adds the arrays, historical checkpoints, prepared PHerc343 inputs and saved maps that the earlier text-only export omitted. The original files retain their bytes and relative paths; [MANIFEST.json](MANIFEST.json) records every included file, its SHA256, its bundle and the exclusions.

The target machine is Linux x86-64 with nine CPU cores, 9 GB RAM and 30 GB storage. Start with the current PHerc343 groups, run one model job at a time and leave room for new outputs. The full snapshot expands to about 15 GB; the restore tool can delete each downloaded LFS cache object after verified extraction so all compressed archives need not remain on disk. Installed Python environments and operating-system packages are recreated separately.

## Clone without downloading every large object

Prerequisites: Git, Git LFS, Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/getting-started/installation/). Authenticate GitHub on this device using your own account with access to `jl-drx/Herculaneum`; the cloud environment's injected authentication is not transferred. GitHub CLI browser login (`gh auth login`, then `gh auth setup-git`) is one option. Do not put tokens into commands, chat or the repository.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch codex/plinius-pherc343-reproduction https://github.com/jl-drx/Herculaneum.git
cd Herculaneum
git lfs install --local
python3 research/tools/workspace.py list
```

For an existing clone, fetch and switch to that branch with `GIT_LFS_SKIP_SMUDGE=1` set while checking it out. `main` may not contain this research history until the research PRs are merged. Do not run an unrestricted `git lfs pull` on a space-constrained machine; it downloads every bundle.

## Restore the current experiment inputs

The following location is arbitrary and works with the new portable commands. `.local/` and `.venvs/` are ignored by Git.

```bash
PLINIUS_LAB="$PWD/.local/lab"
python3 research/tools/workspace.py restore \
  --destination "$PLINIUS_LAB" \
  --groups core pherc343-models pherc343-inputs pherc343-reference pherc343-results \
  --drop-cache
bash research/tools/setup_portable_cpu.sh pherc343
.venvs/pherc343/bin/python -B research/tools/run_pherc343.py smoke \
  --lab "$PLINIUS_LAB" --model seed42 --output "$PWD/.local/runs/seed42-smoke"
```

The smoke command strictly loads the pinned checkpoint, verifies the exact debugging CT tile and compares a fresh probability map to the cloud qualification output. It must print `status: PASS`. It does not rerun the complete study or authenticate ink. A clean Python 3.14.7 / Torch 2.13.0+cpu environment is installed without changing another environment. The install uses package hashes and the official CPU wheel index; no CUDA or paid service is needed.

Run one complete fixed-depth map in a **new** output directory:

```bash
.venvs/pherc343/bin/python -B research/tools/run_pherc343.py map \
  --lab "$PLINIUS_LAB" --model seed42 --phase plain \
  --output "$PWD/.local/runs/seed42-plain"
```

Models are `seed42`, `seed43`, `dnative` and `hecate`; phases are `plain` and `padded`. Defaults preserve four CPU threads, CPU fp32, the original full tile grid, fixed d=+1 and reverse direction. Hecate resampling is already prepared; do not substitute a voxel-size declaration for resampling. Hecate is an hours-scale CPU job. Its padded map on this second device would still be a reproduction of the same field, not a new independent research sample. Coordinate distributed work before duplicating the active cloud run.

## Restore historical research as needed

```bash
python3 research/tools/workspace.py restore --destination "$PLINIUS_LAB" --groups history --drop-cache
python3 research/tools/workspace.py verify --destination "$PLINIUS_LAB" --groups all
bash research/tools/setup_portable_cpu.sh history
```

`restore --groups all` can also restore the whole snapshot directly. Existing matching files are reused; a differing file stops restoration instead of being overwritten. The destination must be fresh or created by this restore tool. Interrupted downloads/extraction can be retried; a pre-existing `.plinius-partial` is preserved for inspection rather than silently discarded. Restoring another snapshot into the same destination is deliberately rejected.

The historical runtime is Python 3.12.14 / Torch 2.12.0+cpu. Historical scripts remain archival originals: many embed `/workspace/herculaneum-lab`, `/workspace/Herculaneum` and `/workspace/.venvs/herculaneum`. Their bytes and pins were not rewritten. For exact archival execution, reconstruct those paths on a separate machine or make an explicitly documented working copy with adapted paths. Do not execute write-once training workers inside the retained evidence directories. Having the inputs and dependencies does not mean every historical experiment has been replayed on the new hardware. The existing [phase 39 replay](../README.md#reproduce-experiment-39) remains a smaller, independently checked entry point.

## What the snapshot contains and omits

| Group | Contents |
| --- | --- |
| `core` | Research code, notebooks, reports, configurations, full retained manifests, pinned upstream runtime source, terms and attribution |
| `history` | Retained historical arrays, CT chunk subsets, figures, trained checkpoints and other binary inputs through the earlier studies |
| `pherc343-models` | The four exact checkpoint containers used by the active reproduction |
| `pherc343-inputs` | Completed 66-layer render, padded render, both prepared Hecate volumes and valid mask |
| `pherc343-reference` | Original/replayed geometry and published reference image |
| `pherc343-results` | Six completed Villa maps, their receipts, and saved synthetic/real qualification arrays as of packaging |

The snapshot excludes credential/infrastructure setup, installed environments, the extracted renderer and its download archive, the current native CT cache, temporary/process state and files flagged by credential-pattern scanning. Exclusions are explicit in the manifest. Some flagged source fixtures may contain synthetic credential examples; they are conservatively omitted from these bundles rather than silently sanitized. Previously exported originals may still be available in `research/lab/` under their existing reviewed provenance.

This is a dated data snapshot, not a synchronized workspace. The two Hecate maps and final composition were still running when packaging began and are **not** included as completed results. No live PID, heartbeat or automatic job restart is restored. Historical status prose refers to its original date; inspect local processes and new receipts before claiming anything is running.

Large files appear as small pointers in Git itself. They are usable only after the corresponding LFS objects download successfully; the loader verifies both the object and every extracted file. `--drop-cache` removes the verified local LFS download after extraction, not the remote object or restored data. The selected research licenses/notices remain in their original locations; no blanket license is applied to upstream code, scan derivatives or models.

See [VALIDATION.json](VALIDATION.json) for the exact checks performed, including whether each used a local retained bundle or a real GitHub download. A same-host isolated restore is not a test on your physical device.

## Completed follow-up supplement

[Phase47 replay and current research anchors](PHASE47.md) add the sealed four-origin
comparison and PHerc0800 geometry preparation. Its separate 39.2 MiB LFS bundle
replays saved outputs with only NumPy/SciPy. The historical workspace manifest
and completed four-model bundle remain unchanged.

## PHerc0800 support follow-up

[Phase48](PHASE48.md) adds the full-resolution support masks and current seal.
The compact [spare-VM geometry assignment](../handoffs/pherc0800-support-20261010/START_HERE.md)
is separately packaged through the existing manual-only Actions workflow. It
contains no CT intensity or model weights and needs no LFS download.

## PHerc0800 transfer qualification

[Phase49](PHASE49.md) adds the fixed-patch renderer/model checks, geometry-support
planning,43.6MiB evidence bundle and no-inference replay. It is separate from the
VM’s phase48 assignment; there is no new duplicate VM job or Actions dispatch.

## PHerc0800 independent-audit follow-up

[Phase50](PHASE50.md) preserves the returned VM audit, complete replay, smaller
full-recipe candidate, observed rendering and all16 fixed-centre maps. Its
separate LFS supplement supports read-only, no-inference saved-result checking.
Short checks remain in this chat; delegate only when local long-running work
makes independent parallel work useful. No new Actions task was needed.
