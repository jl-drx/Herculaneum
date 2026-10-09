# Preparation while PHerc343 inference runs

9 October 2026. This is a planning note, not a new experiment or a change to the executing [phase46 protocol](../../in-progress/pherc343-reproduction/PROTOCOL.md). It was written after inference began and some maps completed, before inspecting their scientific images. Runtime logs, checksums and qualification outputs have been reviewed. The existing endpoint and stopping rules remain in force; this note adds no numerical success threshold.

## Review the fixed reproduction

When all eight maps and composition finish, check their receipts and retain an equal-scale comparison of the published image, reproduction and difference image. Keep the fixed area, direction, depth, ensemble statistics and display stretch. Review the entire panel and all 18 source letter-candidate boxes, including failures and the surrounding papyrus. Record spatial correspondence, missing structure, extra responses and seams. The boxes locate author candidates; they do not authenticate characters or provide pixel-level labels.

Preserve the four individual model contributions so that a similar-looking ensemble cannot conceal a materially different contribution. The author's individual raw maps are not supplied, so do not claim their numerical reproduction or reconstruct them from the composited PNG. Report RGB differences as display differences, not AP or calibrated ink probability. Do not tune display thresholds, weights, depths or checkpoints after seeing discrepancies.

| Outcome under the existing protocol | Next decision |
| --- | --- |
| Source-linked published structure recurs, with numerical and visual differences documented | Qualify this positive control for planning a prospectively fixed PHerc0800 transfer. This does not independently authenticate a reading. |
| Rendering/preprocessing/output checks fail | Correct the identified implementation problem; do not interpret the image as a scientific null. |
| Checks pass but published structure fails to recur | Report failure to reproduce; localize model, precision or composition differences before target inference. |
| Correspondence is too ambiguous to judge | Report it as inconclusive. Specify the missing reference or bounded comparison needed; do not declare success from a favourable-looking subset. |

The fixed d=+1 run alone cannot determine depth stability or adjacent-wrap contamination. Those questions belong to the separately frozen target study.

## Establish an actual prospective PHerc0800 field

[SOURCE_INVENTORY.json](SOURCE_INVENTORY.json) records retained source identities and the minimum known exclusions. Coordinates below are native surface `(x, y, width, height)`, not CT-volume coordinates.

| Segment | Previously inspected extent |
| --- | --- |
| 20251029010146 | `(0, 0, 980, 980)` |
| 20251028222030 | `(512, 512, 1536, 1536)` |
| 20251028220955 | `(768, 256, 768, 768)`, including the earlier `(1024, 384, 384, 384)` crop |

These are development exclusions, not three certified independent physical regions. Different mesh identifiers may map to the same papyrus. A candidate reserve still needs a CT-coordinate support comparison against prior acquisitions, including the full depth envelope and model context, and a wider history check before it can be called uninspected. An overlapping bounding box is not proof of identical tissue, and comparing mesh centres alone is insufficient.

The five retained larger meshes provide about 1.54–2.36 cm² of nominal valid mapped tissue each. Select a coherent field using geometry, support and coverage before viewing new predictions; do not join unrelated tissue merely to reach a nominal area. The prize's 4 cm² window is a maximum region containing the required letters, not a minimum amount of valid papyrus. No fresh assessment field is certified by this metadata review.

Original geometry remains the baseline. Sparse mesh-to-M7 disagreement does not justify snapping every surface; a locally justified geometry comparison must be a separately specified factor. See the existing [target-transfer design](../../lab/reports/fortyfifth-experiment/NEXT_STUDY_DRAFT.md).

## Budget the next computation before launching it

The [read-only workload count](HECATE_WORK_ESTIMATE.json) finds 17,820 nonblank Hecate tiles across the two current phases. At the measured 8.486 seconds per batch of four, Hecate alone is approximately 10.5–10.8 hours before sustained-throughput and I/O differences. The six faster model maps add roughly an hour. This is an estimate, not a completion guarantee.

The proposed −15…+15 centre sweep in both directions is **62 times** the current one-depth, one-direction workload on an otherwise equivalent field: 496 maps versus 8 when retaining all four models and both tile phases. Straight CPU extrapolation is roughly a month for a PHerc343-sized field, before additional geometry variants. That is a sizing warning, not a PHerc0800 runtime measurement; its actual field, occupancy and necessary tile phases must be frozen first.

Prepare a small GPU throughput/memory benchmark and a costed workload after the current positive-control assessment. GPU readiness depends on a concrete workload, pinned CUDA environment, real scoped artifact delivery, current pricing and independent shutdown within the £10 total pilot cap, including failures and overhead. It does not require an arbitrary CPU AP improvement. The historical author cost is not a current quote, and this note neither launches nor authorizes a paid job. Do not promise both a full inference sweep and a 20k-step training pair inside the same cap without measuring them.

For now, document review, region provenance and workload sizing are useful. Another training arm, depth sweep or bulk CT download would compete with the active four-core worker and should wait for a specific decision it can resolve.
