# PHerc343: completed saved-map detectability calibration

**The raw fixed-scale review can recover several of the reported positive structures.** The concern that study52's individual-map display might hide *all* structures recognized in the published composite is not supported. Strong broad structures survive both padding phases and the Villa/Hecate comparison; weaker candidates remain diffuse, disconnected or ambiguous. This supports using the existing gate to select strong leads, while leaving its ink sensitivity and false-positive rate unknown.

The [plan](CALIBRATION_PLAN.md) was written before opening the maps in this review. All eight saved encoded and decoded hashes, dimensions and model/phase identities passed against their original receipts and the completed study46 checker. The inherited published-crop valid mask passed its original **packbits** SHA-256. The first script attempt mistakenly compared unpacked bool bytes with that packed digest; it stopped before making panels, and the failed source plus [attempt note](CALIBRATION_ATTEMPT1.json) are retained. This was a reviewer-script encoding mistake, not a failed retained mask or a changed tolerance.

All eighteen original boxes and all eighteen deterministic 1536×512 contexts were inspected. No model forwards, new CT acquisition, fitted contrast, teacher labels, threshold optimization or partial study52-map review occurred. The context width/height matches study52's nominal physical extent at the shared 8.64µm spacing. Contexts overlap, and are not independent samples. The full box12 is taller than512 pixels; its complete-box panel was reviewed separately and not silently truncated.

## Recorded judgments

“Yes” below means recognizable broad structure at the same source coordinates in both phases and at least one Villa checkpoint plus Hecate. It does **not** mean a character was read, the structure is ink, every model agrees, or the complete study52 advancement rule is passed. “Uncertain” means location-level response recurrence is present but the coherent-structure interpretation is too weak or dependent on the reference cue. These are one model-assisted reviewer's qualitative assessments. The published composite was shown as a labelled comparison, so this is a **cued**, exposed-reference assessment, not blind reader validation.

| Box | Family/phase corroboration of broad structure | Observation, including disagreement |
|---|---|---|
| [1](calibration/box-01.png) | Yes | Left vertical/base arrangement persists in dense_native and Hecate; the separate upper-right response is much weaker in Hecate. Seed43 fragments the base. |
| [2](calibration/box-02.png) | Uncertain | The elongated upper response recurs across families/phases, but it remains essentially one mass; a stroke group is not established. |
| [3](calibration/box-03.png) | Uncertain | Several positions recur, especially upper-left/lower-right. Middle responses and connectivity differ substantially; no coherent common group is secure. |
| [4](calibration/box-04.png) | Yes | The broad bent horizontal arrangement and descending left component recur clearly in dense_native and Hecate. Seeds change the connecting bridge. |
| [5](calibration/box-05.png) | Yes | Lower horizontal response and right upright recur strongly in dense_native and Hecate. Seeds retain components but weaken or disconnect the centre. |
| [6](calibration/box-06.png) | Uncertain | Upper/right and lower responses recur. Hecate suggests a connected bent structure, whereas Villa mainly supplies separated masses with changing strength. |
| [7](calibration/box-07.png) | Yes | Vertical two-part arrangement persists across families/phases; Hecate splits a continuous Villa response and adds a lower horizontal component. |
| [8](calibration/box-08.png) | Yes | Left upright and separated right bent/horizontal arrangement recur. The gap and right-side linkage vary across models. |
| [9](calibration/box-09.png) | Yes, at broad-group level | The roughly surrounding arrangement persists in dense_native and Hecate, but Hecate breaks it into separate components; seed42 plain is substantially weaker. No shared closed contour is claimed. |
| [10](calibration/box-10.png) | Yes | Central crosswise response and lower/right descending mass recur across phases/families. Upper tips differ; seed43 is weak under padded sampling. |
| [11](calibration/box-11.png) | Uncertain | Left/lower and upper/right responses recur, but the putative enclosing structure is diffuse in Villa and disconnected in Hecate. |
| [12](calibration/box-12.png) | Uncertain | Multiple features recur over the full tall box. Their connectivity and grouping change, and the small context clips the full box. No single coherent full-box form is asserted. |
| [13](calibration/box-13.png) | Uncertain | Villa's upper-right feature has only weak/fragmented Hecate correspondence; a lower Hecate spot changes strength. A coherent corroborated stroke is not established. |
| [14](calibration/box-14.png) | Yes | Curved left/upright arrangement is recognizable in dense_native and Hecate in both phases. Seed43 changes orientation/extent of its brightest mass. |
| [15](calibration/box-15.png) | Yes, at broad-group level | Upper left-to-right elongated arrangement recurs in seed43 and Hecate. Dense_native is diffuse and seed42 weak; lower components are not corroborated. |
| [16](calibration/box-16.png) | Yes | Horizontal base and right upright recur clearly across families/phases. Thickness and the top connection vary. |
| [17](calibration/box-17.png) | Yes | Upper/right bent structure and lower return are supported by seed42 and Hecate across phases. Dense_native and seed43 are much less informative. |
| [18](calibration/box-18.png) | Uncertain | Upper and lower response groups recur, but linkages differ enough that a common coherent form is not secure. |

The table records eleven “yes” and seven “uncertain” judgments. **11/18 is not recall, sensitivity, AP, an independent-sample estimate or a new advancement threshold.** Some boxes overlap; the original candidates are not all independently authenticated letters; model/teacher ancestry and the reference display are shared. A different reviewer may reasonably classify borderline groups differently. All contrary views remain available.

## What changes, and what does not

This closes the *qualitative detectability* gap identified in the initial [scientific review](REVIEW.md): the raw per-map display and conjunction are capable of recognizing strong published-reference structures under matching nominal voxel spacing/energy and field size. It gives no reason to alter the running study52 or loosen its fixed display. It also shows why requiring complete topology or agreement from every Villa checkpoint would be too strong: those are absent even for several clearly recurring positive-reference structures and are not the frozen rule.

It does not validate the entire study52 gate. The retained PHerc343 maps only cover the published d+1 reading direction, so they cannot test **improvement over d0**, unknown direction selection or sensitivity to centre spacing of fifteen voxels. The annotation boxes and published PNG remain locations/display evidence, not ink truth. The inherited valid mask is a full-depth input mask; this analysis did not recompute study52-style model-context support for PHerc343. Almost all pixels inside the eighteen full boxes are valid (minimum99.9101%), but contexts can include more excluded material (minimum80.9179%). Brown pixels were retained in view and never treated as negatives.

The practical decision remains: finish all study52 comparisons unchanged. Advance a qualifying nonzero-depth candidate only to independent image/geometry review. Otherwise close the screen as inconclusive and proceed with PHerc0139 correspondence qualification. A negative target result would be less plausibly explained by *universal inability of the raw display to show strong shapes*, but weak signal, material/acquisition transfer, geometry and detection-depth sensitivity remain unresolved. No additional PHerc343 inference is justified by this calibration alone.

## Reuse

[`calibrate_saved_maps.py`](calibrate_saved_maps.py) reproduces the36 panels and verification evidence from the retained inputs in the existing PHerc343 CPU environment. It uses no neural imports. All output goes into a new `calibration` directory and an existing output directory is rejected. [`calibration/VERIFICATION.json`](calibration/VERIFICATION.json) records input identities, hashes, bounds and support counts; [`calibration/SOURCE_MANIFEST.json`](calibration/SOURCE_MANIFEST.json) binds code, plan and sources. The packet manifest binds these reports and all figures.

Each `context-NN.png` corresponds to `box-NN.png`. Grayscale intensity mapping is unchanged; nearest-neighbor resizing is display-only. Hecate uses full-canvas float32 division by255, the author linear resize to native coordinates, then native padded-crop alignment. The published reference panel has its original composite display and is expressly not an equal intensity scale or an accuracy target.
