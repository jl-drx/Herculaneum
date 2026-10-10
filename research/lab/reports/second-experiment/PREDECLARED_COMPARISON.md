# Known-ink control and additional PHerc0800 contexts

This comparison tests whether the prepared detector can respond to a traceable scan-derived ink label, and whether its earlier PHerc0800 response is unusual among contexts chosen by geometry and source coverage. It does not establish target ink, readable characters or translation.

## Analysis fixed before the new runs

- Retain the two pinned final-step checkpoints, upstream robust-MAD normalization, CPU FP32, 128×128 patches, stride 64 and Hann blending. Do not choose a checkpoint or depth window by its agreement with labels.
- Use 256×256 source contexts and report the central 128×128 region at local X/Y 64:192. Compare the centered 17 genuine source layers in their documented forward order; record any difference in layer origin and physical spacing. Never pad invented depth layers.
- For each context, include only geometry/source-valid pixels that both models actually evaluated and that have some observed nonzero data in the selected 17 depth layers. On a label reference, additionally apply its documented annotation-validity mask. Retain the analysis mask and excluded counts.
- Report each seed's raw-score mean, median, 1st/99th percentiles, maximum, and the fraction where both scores are greater than 0.5. The last quantity is a diagnostic midpoint count under 0.25/0.75 training targets; it is not an authenticated-ink fraction.
- If the label source defines both ink and non-ink annotation, report score distributions by class and descriptive precision/recall and ranking on this fixed crop. Exclude unknown or ignore classes. If it supplies positive-only markings, report positive coverage and refrain from false-positive/precision claims.
- Display score maps on the same fixed 0.25–0.75 scale; label any source-image contrast adjustment. Preserve raw arrays and do not interpret a display adjustment as new evidence.

## Interpretation limits

The known-ink crop may come from the detector's stated training corpus. It is a positive control of data alignment and detector behavior, not an independent accuracy benchmark. Both seeds share training data. Pixel-level statistics are descriptive, with no independent-sample confidence interval.

The full small PHerc0800 segment was already scored in the first experiment. Additional contexts selected without querying predictions remain within that same physical region; they are not unseen test data. Preserve the deterministic selection record before opening their new outputs.

The selected PHerc1667/w013 reference has 9.596 µm sampling after its documented four-slice pooling. Native PHerc0800 sampling is 8.640 µm, a 9.96% difference relative to the reference spacing. This first comparison preserves these documented inputs and records that mismatch. It does not fit a resampling or orientation transform to make the target look like text.

No new transcription or translation follows from score agreement. The next gate is an image-supported stroke pattern, followed by specialist assessment.
