# Plinius: calibration-only transfer control

Frozen before new calibration predictions or threshold selection. This is the fourteenth local CPU study. It follows the thirteenth stable 1e-6 continuation controls; no new model training is authorized by this design. Root owns the scientific decision. The purpose is to test whether the original development-only operating-point rule transfers after adaptation, separating cutoff transfer from the already measured ranking.

## Fixed roles and inputs

Use both final thirteenth states, A_preserve and B_preserve, without choosing a winner. A checkpoint SHA-256 is f91cef62694c51eab240fe6501cc6f49d1a41db6fa24be3307a5e705900f5bdb; B is 3b28cc5c7d511142afc557599e74a65f46e7d1992c94db4dc9f3a3355a35cdcb. Keep seed42 as the baseline. Reuse its existing calibration maps and threshold lock; no baseline model forwards. Reuse all four thirteenth diagnostic maps for each state and baseline, with the same canonical masks and labels. Do not evaluate old 1e-4 models again.

The original eighth-experiment calibration cases are exactly:

1. PHerc1667-w029, source XYWH [6912,3200,256,256].
2. PHerc0814-46527-rank01, source XYWH [256,1664,384,384].
3. PHerc0814-46527-rank07, source XYWH [640,1536,384,384].

Use the immutable eighth plan's literal case records and arrays. Calibration participation is public validation > 0, intersected with acquisition-observed support and the original 64-pixel interior margin. Labels must retain binary encoding. Negative counts must remain 9378, 40400 and 14644 respectively. Positive scores do not enter cutoff selection. No surface occupancy or prediction-dependent shrinking of participation is permitted.

These PHerc0814 contexts have no 2D context overlap with diagnostic fresh-rank01 [1152,1152,384,384] or fresh-rank14 [128,1152,384,384]. They are the same specimen and historical development corpus: disjoint 2D boxes do not establish physical or teacher-history independence. Frag4/PHercParis1Fr39 and its aliases remain entirely excluded from calibration. PHerc0139 adaptation labels are not calibration data.

## Unchanged inference and threshold algorithm

Reuse the reviewed sixth model helper, eighth array/mask contract, original reference adapter and normalization, and verified capture wrapper. Load raw final states with weights_only=True, strict schema and storage/alias validation. Check semantic state before and after inference and require no gradients or state changes. Set eval mode, deterministic algorithms, four CPU threads. Use original uint8 pooled 21-plane surfaces, forward depth direction, layers [2:19], no adapter mask, unchanged per-tile robust-MAD normalization and sigmoid/Hann blending. Expected single-tile forwards per state are 9 + 25 + 25 = 59; total 118. Save normalized input hashes, raw tile logits, scores, coverage and runtime receipts. No weights, optimizer or checkpoint outputs are needed.

Threshold selection must call the unchanged seventh-experiment operating_point.py (SHA-256 38a0ebbab0835845ee2213c2be9505cf9b41a129ab13dd6668ff797bfacacb73). For N finite float32 negative scores in [0,1], let m=floor(N/20). Sort descending, take the boundary at zero-based index m, and choose the next float32 value toward positive infinity. Classification is float64(score) >= threshold. This excludes the entire crossing tie and permits a finite cutoff above 1. The primary model-specific cutoff is the maximum of the three regional cutoffs. Secondary is the PHerc1667-w029 cutoff alone, reported as a fixed sensitivity analysis, never selected instead of primary based on diagnostics.

The baseline's literal primary and secondary cutoff is 0.7558947205543518. Independently reconstruct it from its retained development negatives as a provenance check, without replacing its frozen artifact. Both new threshold locks must be written and hashed before the analysis program loads diagnostic score or label arrays. Existing diagnostic results are already known: ordering the new computations does not make this prospective validation. Do not fit any function or threshold to diagnostic maps, labels, ROC/PR curves or outputs.

## Predeclared comparisons

Apply each model's primary cutoff to its own four retained diagnostic maps. Also report secondary and common baseline-cutoff operating results. For every context retain exact TP/FP/TN/FN, sensitivity, specificity, precision and differences from (a) the same model at the common cutoff and (b) the frozen baseline at its original cutoff. Report equal-context macros separately for PHerc0814 and Frag4; no pixel-weighted or cross-specimen pooled headline.

The primary descriptive recovery screen, separately for each arm and specimen, requires all of:

- Every context meets specificity >= 95%, checked using exact integer arithmetic (20 * FP <= Nnegative).
- Equal-context mean sensitivity is at least the frozen baseline's mean, using exact rational arithmetic from counts.
- No individual context loses more than 0.02 absolute sensitivity relative to baseline, using exact rational arithmetic.

All four contexts and both specimen-level screens must be reported, including failures. Also report whether calibration increases sensitivity relative to the unchanged common cutoff, without using that secondary comparison to redefine a failed primary screen. These tolerances reuse the existing specificity requirement and per-context sensitivity allowance; they are descriptive, not statistical guarantees. Meeting them against a weak baseline is not evidence of a useful ink detector or grounds for promotion.

AUC/AP and their masks must be unchanged because score maps are reused literally. Recompute them as an integrity check against thirteenth values (absolute tolerance 1e-12), not as new ranking evidence. No score rescaling, model/region/threshold selection, calibration based on positives, post-result criterion changes, letter interpretation or significance claim. No automatic model promotion even if both screens pass.

## Execution, verification and limits

Prepare a fully pinned machine-readable plan, source freeze and narrow independent technical review before root executes. Preparation and source fixtures must perform no model calls. Root alone runs the actual scientific inference. Run A then B once each, with no scientific retry or substitution on failure. Worker limits: 90 seconds wall, 240 CPU seconds, four threads, 3 GiB peak-RSS acceptance, 12 GiB address space. Combined inference wall limit 240 seconds; all run artifacts <= 32 MiB. NumPy-only analysis: 120 seconds wall, 120 CPU seconds, 1 GiB RSS acceptance, 4 GiB address space, <= 16 MiB output. Disable network in execution. Zero GPU, downloads, teacher/baseline/diagnostic forwards, backward passes, updates, new checkpoints, uploads or publication.

An independent NumPy verifier must reconstruct canonical calibration masks, negative counts, float32 cutoff bits/ties and locks, diagnostic confusion counts, exact screens, macros and unchanged AUC/AP. It must not import the author selector/scorer. Separately reconstruct all six new blended maps from 118 raw logits using the already validated NumPy float64 sigmoid/Hann method with FP32 accumulation, predeclared maximum absolute score error 2e-6 and exact evaluated-mask equality. This is a numerical-tolerance check, not bitwise identity across libraries. No reviewer model calls or checkpoint reads.

Freeze all previous studies. Keep code, inputs, output receipts and failed attempts (if any); build an additive retained-state manifest and register after interpretation. CPU work is authorized. GPU spend remains GBP 0 and first paid pilot cap GBP 10 TOTAL; no GPU launch follows from this control. Credentials and paid resources are unnecessary. Plinius is a working research-series name for credited adaptations of scrollprize/ink_9um, not an assertion of novel architecture or authentic reading.
