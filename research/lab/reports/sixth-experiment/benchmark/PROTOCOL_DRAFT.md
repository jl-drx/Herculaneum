# Sixth experiment: fixed CPU checkpoint comparison — draft

This protocol is not frozen and does not authorize inference. The root agent will review the completed model/data contracts and execution plan before freezing it. Existing reference inference code remains unchanged.

## Question and scope

Does Russo's fixed ex016 75k dense-label checkpoint behave differently from its matched manual-label 75k control on the exact same supported corrected w029 validation crop? The primary comparison is paired ex016 minus control ROC AUC and average precision on supplied binary labels. Retained final seed42/seed43 checkpoints provide two additional fixed reference arms. No checkpoint search, fitted threshold, retraining, depth search or learned ensemble is part of this study.

The author's reported full-w029 AUC is lower for ex016 than control (0.8853 versus 0.8945), with seed43 at 0.9298. Those numbers are context, not expected results for this small selected crop. We will retain an unfavorable or null comparison without selecting a replacement checkpoint.

## Data and domains

Use the corrected curated w029 source contract: a bounded uint8 `[21,H,W]` crop, `128 <= H,W <= 256`, formed by float32 mean pooling source depth `[13:97]` in groups of four followed by `np.rint`. The fixed model window is pooled `[2:19]`, forward, with corrected label plane 10. Source coordinates, pooling, annotation version, file hashes and binary encoding must be supplied and verified by the data contract; none is inferred from a filename.

Validation-mask membership defines eligibility. Supervision-mask membership is recorded separately and is **not automatically treated as actual training exposure**: the pinned official loader `data/dataset.py` lines 292–303 subtracts validation voxels for non-validation patches; the flat loss takes supervision depth-amax. The current conditional labeled training mask is S &~V. This does not certify the historical Russo trainer revision or actual historical teacher/physical exposure, which remain unresolved. No direct training-overlap exclusion has been established for this diagnostic, so the frozen contract excludes none.

The pre-inference primary domain is the declared validation mask, actually acquired source support, any established training-overlap exclusions, and a fixed 64-pixel crop margin. It must contain **at least 256 supplied positive and 256 supplied negative pixels before predictions**. If it fails, stop without changing crop, margin or thresholds. The final primary domain additionally intersects all four evaluated masks and finite score masks; the same minimum is checked again. The secondary domain is the same supported validation domain without the crop margin.

Observed zero depth profiles remain eligible when their annotations and acquired source are valid and model support was evaluated. Nonzero occupancy is retained and stratified separately. It is neither an acquisition mask nor evidence of geometric validity. Whole entirely zero model tiles are skipped by the unchanged inference function and remain unevaluated. No authenticated mesh-validity mask is available, so no output is named or described as geometry-valid.

Existing training-overlapped w013 evidence is sufficient for integration context; no new w013 run is included. The four w029 arms are the entire fixed comparison.

## Inference and runtime

Four fixed checkpoints, each verified by byte length and SHA-256. The shared model-contract helper must use CPU `weights_only=True` loading, strict factory tensor-key/shape/dtype matching and the exact audited architecture/config. The driver imports the unchanged `run_ink_reference.py` inference function through that helper and calls `infer_surface(..., layer_start=2, direction='forward', valid_mask=None)`. Its all-ones canvas mask does not establish geometric validity.

Use CPU FP32, batch one, four Torch threads, one interop thread, deterministic algorithms, complete 128-square tiles at stride 64 with the original Hann blend and robust-MAD patch normalization. No mask enters normalization. Launch each model in a fresh subprocess, sequentially, with a **120-second hard timeout**, no automatic retries and a modest fixed output budget. Save raw float32 scores, evaluated masks, model/source/code provenance and measured runtime before analysis.

## Analysis and interpretation

Compute tie-aware ROC AUC and noninterpolated average precision using complete equal-score threshold groups. Save ROC/precision–recall curves. Report prevalence and class counts for every domain. At the fixed midpoint 0.5 using **score >= 0.5**, report confusion counts, sensitivity, specificity and balanced accuracy as diagnostics; 0.5 is not fitted or an authenticity threshold. The operator matches the retained Russo scorer; earlier target diagnostics using > 0.5 are a different convention. Report primary and secondary domains separately, plus zero/nonzero-profile strata where both supplied classes exist. No pixel bootstrap or confidence interval is claimed from this single correlated field.

Retain ex016/control paired differences and all baseline results on identical common support. Raw model scores remain uncalibrated. Supplied-label agreement does not establish authentic ink, correct geometry, physical/teacher independence, letters or reading accuracy. An integration reference cannot establish generalization. The study is a finite-sample descriptive checkpoint/supervision diagnostic, with no promised superiority of the newer model.

## Pending review inputs

- Final model-contract helper, artifact records, authored-config comparisons and hashes.
- Final data-contract records, label/mask encoding, fixed crop selection and pre-inference class counts.
- Final source-selection timing and label-support counts; a label-supported first-adequate crop is developmental rather than untouched.
- Final execution/analysis code hashes and root freeze. No new w013 run is planned.

Changes after root freeze require a separate protocol version. During execution, reviewers write additive reports rather than editing the executing code or source inputs.
