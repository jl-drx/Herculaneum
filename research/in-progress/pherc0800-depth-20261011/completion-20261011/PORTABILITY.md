# Self-contained private map review

Use jl-drx/Herculaneum branch `codex/plinius-pherc343-reproduction`. GitHub access
is required because the repository and its LFS objects remain private.

```bash
git lfs pull --include='research/workspace/bundles/pherc0800-depth-completion-20261011.tar.gz' --exclude=''
python3 research/in-progress/pherc0800-depth-20261011/completion-20261011/restore_packet.py --bundle research/workspace/bundles/pherc0800-depth-completion-20261011.tar.gz --destination /workspace/review-depth
python3 /workspace/review-depth/verify_saved_maps.py --packet /workspace/review-depth > /workspace/rechecked-depth.json
```

The restorer needs Python3.11+ standard library and checks archive plus every
file hash before permitting successful restoration. The read-only checker needs
NumPy and Pillow; the tested versions are in PORTABLE_DEPENDENCIES.json. It verifies
all48 included aligned float32 maps, support,672 comparison scores and original
panel pixels. Historical absolute paths resolve only through the included
SAVED_INPUTS_MANIFEST.json; no fallback to the old workspace is allowed. It
prints JSON and performs no model execution, network request or output-file write.
A fresh relocated packet replay is retained in verification/.

The archive includes all48 saved aligned maps, so the full878MB checkpoint chain
is NOT needed for this review. To inspect committed raw logits, every production
receipt or frozen worker inputs, use the separately pinned21-generation chain
at commit02812de445f6ab0069174805bb14b6014bed159c on private branch
`codex/plinius-pherc0800-phase52-checkpoints`. PORTABLE_DEPENDENCIES.json gives
all archive hashes, sizes and the existing restore tool. That restoration is
separate from running the worker; do not launch a completed experiment.

Some earlier study51 prerequisites are absent from this runtime; no new inference
reproduction is claimed. Original reporting scripts and receipts remain archival,
with their historical paths preserved. The portable checker is the supported
route for a new directory/machine. All figures use fixed0–255 display; metrics
are response agreement, not ink accuracy. No new Action or spare-VM task is needed.
