# PHerc343 reproduction: in progress, blocked at CT streaming

9 October 2026. This is the approved phase46 experiment, still open. It is not a new completed scientific result, a negative ink result, or permission to infer over incomplete CT. The last sealed experiment remains45 and registerv41 is unchanged.

## Completed setup

- Recovered the **original VC3D41701aa AppImage** from GitHub Actions artifact10821272485. The archive SHA256 matches GitHub's artifact digest; the AppImage SHA256 and size exactly match Nieuwlaar's record. Its renderer binary also matches the author SHA256. The extracted command runs in this cloud environment.
- Installed a separate Python3.14.7 / Torch2.13.0+cpu / torchvision0.28.0 environment, keeping the historical Python3.12 environment unchanged. NumPy2.4.4, OpenCV5.0.0, Zarr3.2.1 and tifffile2026.7.14 match the source recipe's recorded versions. LZW/BigTIFF encode/decode passed with imagecodecs2026.3.6.
- Retrieved481 pinned Villa Python/JSON source files at0fb38c45b488e192566ea83ae36369cb6db92652 and verified each against its Git blob identity. No latest-source substitution.
- Verified Hecate9.6 weights against the source SHA256. Rebuilt dense_native from the already-verified public safetensors/config using the author's script; this is not the unavailable original training checkpoint container.
- All four models passed strict state loading and synthetic fp32 CPU inference. Twenty tile forwards checked single-versus-batch4 behavior; maximum probability discrepancy was below1e-4. These are runtime checks, not papyrus predictions.
- The original padding/Hecate-preparation scripts passed a synthetic affine-depth fixture. Padding40/200 native pixels becomes36/180 Hecate pixels; cropping it away gives byte-identical prepared data. Analytic depth values agree within one downward uint8 truncation unit. The initial exact-integer expectation failed because float32 OpenCV area weights put some integral ideal values just below the integer; the author algorithm was preserved. Villa reversal selects layers42..26 exactly.
- A channels-last Hecate benchmark made eight additional synthetic tile forwards. It reduced the measured batch4 time from6.54s to5.13–5.49s with maximum probability difference5.29e-6. This is a candidate runtime optimization only; the scientific runner retains the original layout pending a real-tile comparison. No optimization benefit is claimed for ink recovery.
- Fixed ensemble statistics were transcribed from the public sidecar without refitting. The Villa display transformation `clip((v-64)/128,0,1)` is preserved even though the inference TIFF encodes `trunc(255*p)`. These display values are not calibrated probabilities.

## Current blocker and retained state

The renderer streamed more than2GB of observed source chunks and wrote a partial66-layer surface before repeated HTTP503 errors interrupted the cloud environment's outbound tunnel. The same `envoy://cloudflare_https_tunnel/` transport error affected S3, GitHub and PyPI; intermittent200 responses allowed dependency installation to finish, but sustained streaming did not recover.

Initial retries preserved logs and reused completed chunks. A loopback read-through cache then served retained bytes and retried only transient upstream errors, retaining the existing HTTPS proxy and TLS verification. Optional absent Zarr metadata remains observed404; missing required chunks never become invented zero CT. One five-minute attempt recorded143 HTTP503s and11 successful upstream200 responses. The renderer itself retried exhausted requests, so that attempt was stopped explicitly to avoid an unbounded transport loop. A later successful Git push and two successful CT-chunk probes justified another bounded attempt. It still failed on required chunk `0/77/29/41`, after102 further503 responses and9 upstream200 responses; it exited after142 seconds, before the15-minute timeout. The helper also needed support for optional `meta.json`/`metadata.json` probes, which is now included but has not passed an end-to-end rerun. All workers are stopped. See `TRANSPORT_STOP.json`, `CACHED_RENDER_EXECUTION_V5.json` and retained transport ledgers. Intermittent success does not establish sustained streaming readiness.

**The complete decoded layer34 and valid-mask hashes have not been checked.** There are no scientific ink forwards, completed maps, image comparison or reproduced reading. The partial render must not pass into inference.

CPU batch4 times on the synthetic probe were about0.55s dense_native,0.38s seed42,0.41s seed43 and6.54s Hecate. Full-field Hecate inference is an hours-scale CPU job. This is a measured runtime consideration, not a reinstated CPU-AP gate. No paid job was created; GPU spend remains£0 and the first-pilot total cap remains£10.

## Resume on this machine

Use `/workspace/.venvs/pherc343/bin/python`, with `PYTHONDONTWRITEBYTECODE=1` /`-B`, four Torch threads, and this existing checkout. Do not create a worktree. First recheck representative S3 metadata/chunk and GitHub requests through the existing proxy. No new user credentials are needed for these public assets. Do not repeatedly launch against a confirmed tunnel outage.

From `/workspace/herculaneum-lab/reports/fortysixth-experiment`:

1. Prefer `python -B render_surface.py` once sustained access recovers; it uses the original remote URL and cache directly. `python -B render_cached_transport.py` is the retrying loopback alternative, with observed optional-metadata404 caching. Both resume the original renderer, then invoke `verify_render.py`, and retain separate logs with a15-minute renderer timeout. The latest recovery changes remain untested end to end. Both published decoded hashes must pass. Render metadata/tree encoding may differ; report that separately.
2. Only after `RENDER_VERIFICATION.json` saysPASS: `python -B prepare_and_qualify.py`. This builds the author tile phases and Hecate inputs, then checks fixed real tiles and measures full schedules. It is **unexecuted** because CT is incomplete. Its qualification reader was corrected to convert the upstream H/W/Z array to Z/Y/X before normalization.
3. Only after real-input qualification passes: `python -B run_fixed_depth.py`. The runner uses unchanged upstream tiling/blending, strict weights, fp32 CPU, eager execution, batch4 and no data-loader workers. Those device/compilation differences from the published GPU recipe remain explicit. It preserves individual reading-side maps on both tile phases. No new depth/checkpoint selection.
4. After all eight maps complete: `python -B compose_and_compare.py` merges original ownership, runs the author's display recipe and records descriptive image differences. This stage is **unexecuted**, as is full inference. Visual assessment and independent output verification remain required before claiming reproduction.

Do not treat syntax checks or the synthetic smoke test as end-to-end validation of steps2–4. Any correction should be additive and recorded before scientific outputs are interpreted. The exact CT and reference hashes remain the gate.

## Persistence

This directory is unsealed work in progress. Selected scripts and receipts are copied to `research/in-progress/pherc343-reproduction/` for review, separately from the sealed historical export. No completed experiment is added to the research register until this reproduction has an actual outcome. Bulk CT, partial render, checkpoints, extracted renderer and installed runtime remain cloud-local. A Git commit is not their backup or a fresh-machine replay.
