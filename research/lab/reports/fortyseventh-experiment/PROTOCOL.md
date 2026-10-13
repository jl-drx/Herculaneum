# Phase47: bounded seed42 grid-origin comparison

Approved10 October2026 after completion/review of phase46. This is one exposed
PHerc343 diagnostic, not training, fresh validation or a new reading claim.

Keep the qualified seed42 step075000 checkpoint, CPU fp32/eager, four Torch
threads/one interop thread, deterministic algorithms, batch4, full raw CT source,
17 reversed planes42..26, original patchwise robust preprocessing,128px patches,
stride64, original floored Hann weights and float32 accumulation/truncation.
The ROI is native rows768:1280, columns2816:3328, centred on the original debugging
coordinate. Read complete patches and preserve their full global batch cohorts.

The fixed row/column origins are(0,0),(0,32),(32,0),(32,32). Each full-axis schedule
uses its offset modulo64 plus explicit0 and length−128 starts at the field ends;
the ROI is far from these boundary adjustments. Apply the original full-volume
occupancy rule to each schedule. Group row-major scheduled tiles into batches of4
before selecting complete batches that contain at least one ROI-overlapping tile.
The same upstream reader, dataset, forward loop and chunk accumulator are used.
Only the list of global scheduled blocks and the output storage backend change;
NumPy stores receive the unchanged original128px chunk buffers. All contributing
patches use actual CT context; no cropped-volume padding is introduced.

Before new model outputs, freeze full schedules, retained batch indices, source,
CT chunks, model, baseline image, valid mask, scripts and the phase46 seal hash.
Offset0 must exactly reproduce all512×512 saved seed42 plain pixels, including
invalid support. A failure stops shifted inference; it is a harness qualification
failure, not a scientific negative. Preserve the output and discrepancy receipt.
Do not relax the gate or select another ROI/precision/batch size after a failure.

After a baseline pass, run the other three fixed origins once. Stop after four
origins or for a changed identity, missing CT, nonfinite output or resource error.
Retain raw per-batch probabilities, coverage weights, encoded response maps and
all input/output receipts. Independently reconstruct ROI weighted sums from saved
tiles before interpreting differences. There is no best-origin/model selection.

On the fixed valid support, report all six pairwise uint8 MAEs/correlations,
top5/10/20% tie-inclusive Dice and bidirectional nearest-response fractions within
0/2/4/8 native pixels. Also report offset0-relative disagreement in native source
coordinates and64px grid-phase bins for both participating grids. Use all offsets
and unchanged display transforms; show invalid support explicitly. A shifted
bin pattern alone is not a causal demonstration: at half-stride translations,
relabelled phases are algebraically related. Do not claim grid locking from that
identity or treat dependent pixels/overlapping patches as independent samples.

Decision: exact-baseline-qualified changes in high-response support demonstrate
operational grid sensitivity at fixed CT. Stable source-located structure supports
an uncertainty envelope, not accuracy. This limited design cannot by itself
separate patchwise normalization, model context and blending mechanisms, nor
authenticate features. Regardless of a weak ROI result, finish this bounded test
and return to prospective PHerc0800 preparation; do not search more offsets.

No target ink inference, new CT download, GPU launch, R2 upload or messages to
others. CPU time is unrestricted; the first GPU cap remains£10 total. Target
geometry/exposure planning may proceed using retained meshes and metadata.
