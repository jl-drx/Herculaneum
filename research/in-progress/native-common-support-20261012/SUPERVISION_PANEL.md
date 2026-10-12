# Separate supervision-supported panel, before any model inference

The original union-selected panel is retained unchanged in `pilot/`. Its 18
PHerc0841 cores contain 169,462 native-supported scoring pixels and zero positive
reference pixels. All three 0841 segment AP/AUC comparisons would be undefined.
The cause is visible in the decoded masks: 0841 supervision is a strict subset
of a much larger validation area, whereas 0009B validation is contained within
supervision. Union order itself changes zero pixels. The original panel is a
failed admission attempt, not a successful two-parent ink-model benchmark.
Do not describe these reference-negative regions as independently verified blanks.

Freeze a second development panel in `supervision-pilot/`, without overwriting
the first. Preserve the same four segments, 128 grid, six cores per segment,
4096-pixel eligibility threshold, context calculation and deterministic spread
algorithm. Change selection support and primary score participation to the
separately decoded/resized **supervision mask** only. Use all chosen cores;
require both reference classes at segment level to proceed with ranking metrics.
If that fails, stop instead of selecting another core. Record per-core single
classes as undefined, not AP/AUC zero. The 0009B choice is expected to be unchanged
because its published union equals supervision; verify rather than assume it.

This is an explicitly adaptive development-design correction made after viewing
reference class counts, before any model outputs. The new selector receives only
participation masks, metadata and the original class-count failure receipt; ink
values, CT intensities and model predictions do not select coordinates. Labels
have already been decoded in the original intake: no claim of label blindness
at the project level or untouched evaluation is made. Reusing cached source
objects happens only after the new coordinates and chunk plan are frozen.

All other model, preprocessing, depth, global-grid, precision and reporting rules
in PROTOCOL.md apply. Primary results are agreement within the explicitly supplied
supervision region, not a reproduction of the full published union metric.
Keep the broader union mask for optional descriptive stratification; do not use
it to rank models after seeing their scores. Missing mapping validity, uncertain
registration, model-reviewed references and training ancestry remain limitations.
No models or paid compute are launched by this selector.
