# Study51: fixed centre-depth PHerc0800 development-field comparison

Frozen before new CT acquisition or model outputs. Approved continuation of
study50 NEXT_STUDY_DRAFT. Segment20251028222030; global canvas2580x2580;
scoring x512:2048/y512:2048. This is previously inspected development material,
not a pristine holdout, labelled accuracy benchmark or authenticated reading.

Render the pinned original mesh, flip normals,66 planes at offsets k-32.5,
8.64um isotropic, using the original qualified VC3D renderer. Partition the
study49 required-valid context into maximal vertically joined horizontal runs;
render each fully valid rectangle in global coordinates. Unknown geometry stays
unknown. Freeze their source union before access, at most384 level0 chunks
(768MiB), reusing only verified same-source bytes. Missing chunks are fatal.

Check physical attributes and output shape for every rectangle. Assemble without
overlap into explicit observed-mask canvases. Independently sample all66 layers
at up to8x8 evenly spaced positions in EACH rectangle and9 geometry-selected
32-square sites nearest a fixed3x3 scoring lattice; freeze these sites before
access. Required agreement: >=95% exact, mean absolute difference<=0.1, maximum
<=1 uint8 count, per rectangle/site as well as pooled. No output-based site choice.

Use dnative, seed42, seed43 and Hecate checkpoint identities from studies49–50.
Native Villa indices25:42, reverse41:24:-1. Hecate original global0.9 area resize,
16 planes centred32.5 at9.6um; preserve physical-centre half-voxel difference.
Both plain and padded(top40,left200) phases and both array depth orders:16 maps.
No trained parameter, depth sweep, M7 snap, fitted contrast or ensemble selection.

Retain each family's complete original grid. Compute only fully observed patches
contributing to the fixed scoring field. Exclude any scoring pixel affected by an
unobserved original contributor, including Hecate output interpolation. Recompute
and require the frozen49 common geometry mask of1,875,648 pixels. Individual
maps retain explicit family support masks; interpret comparisons only on the
common support. Unknown pixels are not observed zero or negative evidence.

Batch contract: fp32 CPU,4 threads, deterministic operations, maximum batch4.
Villa selected patches stay in original row-major grid order. Hecate uses the
original global four-origin cohorts, with skipped noncontributing/unknown tiles;
retain original denominators. Validate actual normalization/support on four
distinct predetermined observed tiles per family/model/order against singleton
inference: maximum probability discrepancy<=1e-5. Preserve qualification arrays;
stop on failure, do not silently alter batching. This does not assert exact
full-canvas batch1/GPU equivalence. Compare the cache wrapper to direct original
framework output on these cohorts and require bit-exact saved-logit replay.

Commit raw logits plus input/cohort hashes after every forward batch using fsync,
atomic rename and a receipt last. Recovery replays verified committed batches
through fresh original accumulation, never double-adds a prior prefix. Inject a
stop after two committed forwards and require replay equality before the long
worker. Validate missing/corrupt/mismatched receipts fail closed in qualification.
Bind protocol, all source/input/script/checkpoint identities before long launch.
Write map completion last after exact read-only logit replay; whole study complete
only after all16 map receipts and numerical checks. A PID alone is not completion.

The previous5.21h model estimate came from repeated fixed tiles; re-estimate using
actual cohorts and report uncertainty. No time limit changes scientific scope.
GPU spend0; no paid resources or Actions dispatch. Short checks stay here; spare
VM work only when a local hour-plus run makes an independent task useful.
