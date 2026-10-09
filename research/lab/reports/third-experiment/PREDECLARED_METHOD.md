# Larger PHerc0800 context: fixed method

The next experiment enlarges spatial context using a region chosen from mesh geometry before inspecting new detector predictions. Segment identifiers alone do not establish independent physical surfaces: the larger meshes occupy the same CT neighborhood as the initial small segment. Coordinate overlap will be reported.

## Acquisition and inference

Prefer a 1536×1536 native crop with a chunk-aligned origin on one of the five larger released surfaces. Record all ranked geometry candidates, the deterministic selection rule, mesh/source metadata and final selection before retrieving surface pixels. Retain 31 genuine depth layers at 8.640 µm, with an exact payload cap computed from required chunks. Do not download a whole CT volume.

Run the two unchanged pinned final-step models on forward layers 7:24. Preserve upstream patchwise robust-MAD normalization, 128×128 windows, stride 64, Hann blending and CPU FP32 at four threads. Keep the original adapters unchanged; a separate bounded larger adapter may widen only size/tile-budget guards and corresponding documentation.

Independently of the forward result, prepare a fixed central 1024×1024 crop. For a 1536 crop its relative origin is (256,256), preserving tile phase. On that crop run seed42 with forward windows 5:22 and 9:26 and reversed window 7:24. The original full-field forward result is the comparator on common central pixels. Verify crop/full-field forward equality on the shared interior before interpreting controls. Controls cover only this fixed subregion, not every candidate in the larger field.

## Analysis domains

Retain the raw float32 scores and all masks. The primary domain requires conservative mesh validity, both evaluated masks, any nonzero source intensity in the selected depth layers, a 64-pixel crop-edge margin, and at least 64 pixels from invalid geometry in the full source geometry mask. Use the same fixed coordinates and common domain for depth-control differences.

A 64-pixel geometry margin does not guarantee that every contributing 128×128 tile avoids invalid input. Therefore report a second, stricter domain: exclude the entire support of each contributing scored tile containing any invalid geometry or any selected-depth pixel whose complete profile is zero. This is a conservative support diagnostic, not a papyrus or ink label. Zero intensity or excluded geometry does not establish absence of ink.

Report score distributions, seed differences and the count/fraction where both raw scores exceed 0.5 on each domain. This threshold is a behavior diagnostic under smoothed 0.25/0.75 training targets. Report connected-component locations and areas only as image-analysis leads; do not equate pixels or components with ink or letters. No target accuracy, confidence interval or text is inferred from these measurements.

Use fixed 0.25–0.75 score-map display limits and label all source-image contrast changes. Compare observed depth layers and maps before suggesting any glyph. If a lead lies outside the fixed control crop, further controls belong in a separately documented follow-up.

## Geometry overlap and interpretation

Compare same-volume coordinate footprints with the earlier tiny segment using predeclared nearest-point distances of 1 and 3 native voxels (8.64 and 25.92 µm). Bounding-box overlap is insufficient to identify a layer. Even nearby points and aligned normals may reflect adjacent wraps, so report overlap descriptively and refrain from independent-data claims.

The preceding PHerc1667 annotation control supports execution and alignment of that released representation. It overlaps model training and uses annotations that may include pseudo-labels; it does not authenticate PHerc0800 ink. Both seeds share training data. Native target spacing differs from the pooled reference by 9.96%.

The experiment succeeds operationally if it produces reproducible, source-linked image evidence and honest uncertainty. New text requires image-supported strokes and specialist review; transcription and translation remain empty until then.
