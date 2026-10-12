# Native common-support released-model comparison

Frozen before this panel's first trained-model forward, 12 October 2026.
Continue the admitted supervision-only panel from `native-common-support-20261012`.
The original union-mask panel and the label-informed reason for replacing its
selection remain in that packet. This is an exposed development comparison.

Run Reader v2, dense-native, seed42, seed43, then Hecate on all four segments in
the frozen order. Each model retains its original normalization, physical depth
window, global grid, support and blending semantics. CPU fp32, batch 1, four
intra-op threads, one inter-op thread, deterministic execution, no compilation,
TTA, padding, geometry changes or calibration fitting. The prepared input packet
is immutable. The source protocol controls sampling. There are 1,698 unique
forwards, 20 model/segment jobs, 120 core maps and 232,250 common score pixels.

Qualify sparse accumulation against full-grid original pipelines using nonconstant
synthetic outputs before running trained models. Check all 216 real hybrid
normalized tensors against the original reader. Commit each tile's logits and
receipt, including raw and normalized input hashes and GO/model identity. Test
cache mismatch rejection and a controlled interruption. The first real Reader
job also stops after two committed forwards, then resumes; this is an injected
exception, not a process-crash or lost-success-response claim. Replay every job
using saved logits and require bitwise equality without extra forwards. Stop on
input/runtime/hash mismatch, nonfinite values, missing support, cache corruption
or replay disagreement. Implementation failures are recorded before correction.

Do not inspect model metrics until all 20 jobs have completed. Preserve all 24
cores, including the 14 one-class cores. Primary scores are the released hybrid
uint8 truncation divided by 255, or Hecate uint8 rounding on its own grid followed
by float32 linear resize to the native grid. Retain hybrid unquantized maps and
blend components. Hecate's saved sparse global map is meaningful only on the
declared cores. Never interpret its omitted area as negative evidence.

Report for every core and segment:

- AP with scores grouped at exact ties, and trapezoidal ROC AUC. Both are null if
  either class is absent; never silently omit those cores from other measures.
- Prevalence, Brier score, BCE with clipping to [1e-6,1-1e-6] only for logarithms,
  mean supplied-positive and supplied-negative response, and the fraction of
  supplied-negative pixels scoring >=0.5. These are reference-agreement measures.
- Ten fixed equal-width probability bins and their weighted absolute calibration
  gap (ECE); no fitted threshold or calibration.

Pool pixels only within each declared segment for segment metrics. The primary
summary is equal-parent macro: average the three PHerc0841 segment metrics, then
average that result with PHerc0009B. Show both parents separately. Also report
pooled-parent and pooled-panel metrics as explicitly descriptive quantities.
No core bootstrap or pixel-level confidence interval is treated as independent
physical replication. A constant 0.5 predictor supplies the ranking/prevalence
control. A fixed arithmetic mean of all five aligned maps is a prespecified
ensemble comparator, with five-model inference cost; no weight or subset search.

The two released seeds measure checkpoint variability, not a controlled training
seed noise floor. An opposite ordering across parents, or an AP gain smaller than
their observed gap, blocks a universal-ranking interpretation. Even an apparently
consistent result on these two exposed parents cannot establish a general winner.
This diagnostic may motivate broader development coverage and adequately powered
GPU training; a small or negative CPU result is not a universal GPU veto.

Reference mapping and provenance remain imperfect. Annotations are transferred
and model-reviewed; upstream training and selection exposure are not excluded.
Reader's upstream selection on PHerc0841 is particularly relevant. Agreement is
neither authenticated letters nor physical ink accuracy. PHerc0175 A/B and
PHerc1203, including aliases and scans, remain unopened confirmation families.

No paid compute or GitHub Actions. Back up immutable committed progress to a
dedicated private Git/LFS branch during execution. This can restore progress,
not guarantee the lifetime of the cloud process. No new PR is requested.
