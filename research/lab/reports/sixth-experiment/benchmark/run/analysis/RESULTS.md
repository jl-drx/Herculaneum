# Fixed w029 supplied-label comparison

Primary common support contains 16384 pixels: 7006 supplied positives and 9378 supplied negatives.

| Model | ROC AUC | Average precision | BA at >= 0.5 |
|---|---:|---:|---:|
| russo_ex01675k | 0.912934 | 0.874074 | 0.538334 |
| russo_control75k | 0.947220 | 0.920451 | 0.750845 |
| seed42 | 0.924449 | 0.895401 | 0.840397 |
| seed43 | 0.947973 | 0.912323 | 0.864291 |

Ex016 minus its matched control: AUC -0.034287; average precision -0.046377.

These are descriptive supplied-label results on one fixed crop. The higher numerical score is not evidence of authentic text or generalization. No checkpoint, crop, threshold or depth window was fitted after seeing scores.

Raw supervision/validation overlap: 0 pixels; it is retained because the pinned current loader subtracts validation during training. This does not certify historical trainer exposure or teacher/physical independence.

Acquisition support and selected-depth nonzero occupancy remain separate. Zero profiles are retained when actually observed and evaluated. No authenticated geometry-validity mask is available. Full-mask secondary results, occupancy strata, grouped ROC/PR curves and raw float maps are preserved in the linked JSON/NumPy artifacts.

This stage does not choose a new model or authorize target inference, training, a reading, or a GPU job.
