# PHerc343 reproduction: rendering qualified, full CPU inference running

9 October 2026. This is the approved phase46 experiment, still open. It is not a new completed scientific result, a negative ink result, or permission to infer over incomplete CT. The last sealed experiment remains45 and registerv41 is unchanged.

**10 October recovery:** the cloud runtime restarted after seven maps completed. The final padded Hecate map restarted at10:10 BST after saved-map and runtime checks; the original seven frozen files were preserved. A separate recovery runner fixes cleanup when completed Villa maps are skipped. See the [recovery record and current startup guidance](recovery-20261010/README.md); it supersedes the original driver's recovery instruction below. The combined result is still pending.

A [read-only checkpoint observer](recovery-20261010/CHECKPOINTING.md) now saves completed rows without restarting the worker. Seven bounded interruption/recovery cases, including a real Hecate fixture, matched uninterrupted output bit-for-bit. The local saves and dated GitHub backup have distinct coverage; the platform restart's trigger remains unverified.

## Completed setup

- Recovered the **original VC3D41701aa AppImage** from GitHub Actions artifact10821272485. The archive SHA256 matches GitHub's artifact digest; the AppImage SHA256 and size exactly match Nieuwlaar's record. Its renderer binary also matches the author SHA256. The extracted command runs in this cloud environment.
- Installed a separate Python3.14.7 / Torch2.13.0+cpu / torchvision0.28.0 environment, keeping the historical Python3.12 environment unchanged. NumPy2.4.4, OpenCV5.0.0, Zarr3.2.1 and tifffile2026.7.14 match the source recipe's recorded versions. LZW/BigTIFF encode/decode passed with imagecodecs2026.3.6.
- Retrieved481 pinned Villa Python/JSON source files at0fb38c45b488e192566ea83ae36369cb6db92652 and verified each against its Git blob identity. No latest-source substitution.
- Verified Hecate9.6 weights against the source SHA256. Rebuilt dense_native from the already-verified public safetensors/config using the author's script; this is not the unavailable original training checkpoint container.
- All four models passed strict state loading and synthetic fp32 CPU inference. Twenty tile forwards checked single-versus-batch4 behavior; maximum probability discrepancy was below1e-4. These are runtime checks, not papyrus predictions.
- The original padding/Hecate-preparation scripts passed a synthetic affine-depth fixture. Padding40/200 native pixels becomes36/180 Hecate pixels; cropping it away gives byte-identical prepared data. Analytic depth values agree within one downward uint8 truncation unit. The initial exact-integer expectation failed because float32 OpenCV area weights put some integral ideal values just below the integer; the author algorithm was preserved. Villa reversal selects layers42..26 exactly.
- A channels-last Hecate benchmark made eight additional synthetic tile forwards. It reduced the measured batch4 time from6.54s to5.13–5.49s with maximum probability difference5.29e-6. This is a candidate runtime optimization only; the scientific runner retains the original layout pending a real-tile comparison. No optimization benefit is claimed for ink recovery.
- Fixed ensemble statistics were transcribed from the public sidecar without refitting. The Villa display transformation `clip((v-64)/128,0,1)` is preserved even though the inference TIFF encodes `trunc(255*p)`. These display values are not calibrated probabilities.

## Recovery and qualified inputs — 9 October 2026

The environment restart restored sustained public S3 access. All five fresh metadata/chunk probes returned200. Direct renderer attempt7 resumed the original cached render and completed. The complete published-area layer34 SHA256 and the all66-layers-positive mask SHA256 both match the author exactly. `RENDER_VERIFICATION.json` isPASS. The directory has1009 files and945,626,675 bytes,26 fewer than the published tree; its whole-directory hash differs. No byte-identical directory or independent equality of every decoded layer is claimed. The two preregistered decoded fidelity gates passed.

The original preparation scripts produced the plain/padded inputs. An independent full-extent readback verifies that every native padded plane equals the original after removing40 rows/200 columns, with zero padding. The Hecate prepared interiors are also byte-identical after removing36 rows/180 columns. Both input phases retain the full published field and source tiling origins.

All four models passed strict real-input qualification at the predefined debugging coordinates. Twenty real tile forwards checked single-versus-batch4; the worst saved-output difference was9.39e-6, below1e-4. See `INFERENCE_QUALIFICATION.json` and `REAL_INPUT_VERIFICATION.json`. These checks establish runtime/batching behavior, not ink accuracy.

## Active computation

The supervised CPU pipeline started at19:34:31 UTC on9 October2026. It runs `run_fixed_depth.py`, then `compose_and_compare.py` only if inference succeeds. Source scripts, protocol, display statistics and qualification receipts are frozen in `work/pipeline-20261009T193431Z/INPUTS.json`; a lock prevents duplicate supervisors. The wrapper writes a heartbeat and exits on any failed stage. The current-instance live state is `PIPELINE_STATUS.json`; Git contains only a dated observation and must not be used as evidence that a process remains alive.

Three Villa models each schedule2797 plain and2870 padded tiles. Hecate has13,348 plain and14,162 padded tile origins before blank skipping. The one-batch real-input timing gives a conservative scheduling estimate of16.83 hours for all eight maps, almost entirely Hecate; skipped blank tiles reduce work and sustained throughput may differ. This is not a promised completion time. CPUfp32/eager execution and the public rebuilt dense_native container remain explicit differences from the published run. No scientific reproduction outcome, reading, or translation is claimed while maps are incomplete. GPU spend is£0.

## Continue or recover

Use `/workspace/.venvs/pherc343/bin/python` with `-B` and the existing checkout; do not create a worktree. Inspect `PIPELINE_STATUS.json`, its heartbeat, exact process command lines and the current stage log before taking action. A saved PID or RUNNING flag alone is insufficient. Do not launch another worker while the current one is alive, change executing scripts/inputs, or rerun rendering/preparation unnecessarily.

- The renderer and real-input qualification stages have passed. Their earlier interrupted logs remain preserved; the transport error was not a negative ink result.
- If the supervisor has finished, inspect its stage exit codes and output receipts. `COMPUTE_COMPLETE_AWAITING_REVIEW` means image computation finished; independent output checks and visual scientific review are still required before sealing phase46.
- If it failed, inspect the retained error and partial outputs first. `run_fixed_depth.py` resumes complete maps by receipt, but does not checkpoint within a map. An interrupted Hecate map may need recomputation. Never silently reuse unreceipted maps.
- If the environment is restored, live processes may be gone. Validate completed map receipts against actual files and frozen inputs before resuming. The original composition worker expects a fresh `work/maps` directory; inspect partial composition output before any rerun.
- `execute_pipeline.py` is a process supervisor, not a paid-resource launcher, a scientific validator, or a durable scheduler across environment shutdown. It leaves scientific review pending and does not promote a model or start PHerc0800 inference.

## Persistence

This directory is unsealed work in progress. Selected scripts and receipts are copied to `research/in-progress/pherc343-reproduction/` for review, separately from the sealed historical export. No completed experiment is added to the research register until this reproduction has an actual outcome. Bulk CT, partial render, checkpoints, extracted renderer and installed runtime remain cloud-local. A Git commit is not their backup or a fresh-machine replay.
