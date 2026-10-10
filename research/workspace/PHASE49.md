# PHerc0800 fixed-patch transfer qualification

[Study49](../lab/reports/fortyninth-experiment/RESULTS.md) qualifies the pinned
original-mesh rendering and all four models on a fixed previously exposed target
patch. The31-layer render is99.65% byte-exact to the published surface; the66-layer
independent sampler agrees at99.71%, with every remaining discrepancy at most1
intensity count. All20 tile forwards pass. No full target map or reading claim.

The larger-field forecast is2.60 CPU model hours per direction for all four models
and both padding phases, with a conservative768MiB raw-source bound.79.50% of the
scoring field has full joint geometric context under the declared rule. The
[next study](../lab/reports/fortyninth-experiment/NEXT_STUDY_DRAFT.md) prepares a
checkpointed centre-depth comparison of both input orders. The separate VM's
phase48 geometry review is running according to the user; its result is pending.

## Download and replay saved outputs

From this private branch, fetch only the new43.6MiB supplement:

```bash
git lfs pull --include="research/workspace/bundles/pherc0800-transfer-qualification-20261010-001.tar.gz" --exclude=""
python3 research/tools/replay_phase49.py --destination /tmp/plinius-phase49-replay
```

Use Python with NumPy installed and a fresh destination. The command checks the
archive SHA256, all127 members, the cumulative seal, and replays saved comparison
metrics. It runs no renderer, model, network request or paid job. A relocated
replay passed in the minimal NumPy/SciPy environment on the cloud host; physical
second-device inference has not been claimed.

[PHASE49_BUNDLE.json](PHASE49_BUNDLE.json) lists45,739,738 compressed bytes,
71,284,173 expanded bytes, all member checksums and archive SHA256:
`111b120807057d7f42143ed575cb18f76f2b5744774497c6a4366aba288ee0bb`.
The bundle includes the small raw-CT acquisition, renderer outputs, fixed model
inputs/probabilities, scripts, planned support masks and receipts. Older checkpoint
weights and upstream software remain in their earlier groups; they are not needed
for saved-result replay. Study49 virtual canvases contain explicit unknown
placeholders outside the fixed observed patch and must not be used as full inputs.

Current whole-lab seal:
`0e864a6e95214235ee1d13c28cd6b415d8f0a51f497503fb1bab6972e0eef618`,
26,012 records /14,553,010,459 bytes; register v45 contains49 studies. Full retained
readback passed. [PHASE49_VALIDATION.json](PHASE49_VALIDATION.json) separates local
restore/replay and actual remote readback evidence. No historical bundle changed.

The [next-render metadata preflight](../planning/pherc0800-transfer-20261010/RENDER_METADATA_NOTE.md) records VC3D’s fallback output voxel-size attribute. Explicit8.64um preparation was used for Hecate; future render metadata must also state that spacing and pass byte-preservation checks.
