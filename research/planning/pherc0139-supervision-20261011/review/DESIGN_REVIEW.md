# Design review: w040 site 1 supervision sensitivity

Prepared on 11 October 2026 before inspecting any target pixels. This review read only the retained CT correspondence summaries, source/helper code and previous research interpretation. It did not load targets, change registration, run inference, or edit Git-tracked files.

## Recommendation

Run one finite **target-transport sensitivity audit**, conditional on independently verified target coordinates. Its primary output should be descriptive worst-case changes over the retained CT candidate sets. Do not replace the failed CT registration gate with a newly invented claim of verified geometry. A separately frozen operational tolerance can decide whether a later experiment should spend effort on correspondence uncertainty; it cannot certify ink, calibration, or a correct ink-bearing surface.

The key distinction is between a 2D teacher's lateral target sensitivity and full 3D correspondence uncertainty. One site 1 support has candidates at dz=1 as well as dz=0. A depth-aggregated 2D map cannot independently tell us which depth/face supplied its response. Consequently, near-identical 2D transfers do not by themselves clear that ambiguity.

## 1. Freeze the finite problem before target access

Use the four existing 32 by 32 check supports only: support IDs 4–7 at core-relative centres (128,48), (48,128), (208,128), (128,208). The native site core is x2304:2560, y1792:2048. Retain the exact pre-existing half-open support bounds from the CT plan rather than reconstructing a new centring convention.

For support j, freeze every integer displacement whose retained NCC is at least NCC at the common CT-fit displacement minus 0.05. Counts must be exactly 11, 6, 22 and 4. Include the common fit (dz,dy,dx)=(0,-1,0). Save exact displacement lists, original NCC scores, score-array hashes and their source contract; the component-count summary alone is insufficient. These are descriptive score sets, not probabilities, confidence regions or independently calibrated error bounds. Use their envelope; do not weight by exp(NCC), select a convenient quantile, or tune the 0.05 band after target access.

Candidate sets differ by support. Do not stitch them into a whole-core displacement field, infer smoothness between them, or report 4,096 pixels as independent experimental units. Preserve all four per-support results; a pooled favourable mean cannot clear a failing support. Stop after these supports. A blank or masked result is useful and does not authorize searching for a prettier crop.

Provenance entrance requirements: exact historical teacher asset and content hash; its shape, pixel type, origin, padding/crop, source volume and mesh/renderer version; score decoding; missing-data/support semantics; whether any manual edits already enter the teacher; exact native manual-mask coordinates and precedence. The existence of equal shapes is insufficient. Unresolved origin/version is a stop for claimed physical transfer. An explicitly labelled assumed-coordinate calculation can still be reproducible, but must not answer the physical question.

## 2. Separate three contrasts

Keep these contrasts separate in every table:

1. **Published transport versus source-frame transport:** reproduce the pinned complete-image `cv2.INTER_AREA` resize with the original integer decoding/rounding. Compare against the independently derived mesh/source-frame correction at zero residual CT displacement. This measures the proposed source-coordinate intervention, not the local CT uncertainty.
2. **Source-frame transport versus the common CT fit:** fixed zero-residual mapping versus (0,-1,0). This is the contribution of that CT-only fit.
3. **CT fit versus its retained alternatives:** assess all candidate shifts against the same fit, on the same value/support contract. This is the main residual-sensitivity endpoint.

The known 7.43-native-pixel source correction must not be counted inside an uncertainty interval around the local fit. Also keep float area integration versus released integer rounding separate from coordinate differences. In particular, the pinned helper first right-shifts uint16 by eight bits to uint8 and then resizes uint8. A new float calculation is not byte-for-byte replay of that released operation.

## 3. Area and point transport must not be conflated

For the primary 2D sensitivity analysis, use the exact source-frame lateral mapping supported by provenance, composing residual lateral displacement with the retained sign: native[y,x] was compared to mapped_fine[y+dy,x+dx]. Use area integration over each native pixel footprint, not one bilinear sample of the teacher at its centre. The released area-resized teacher and a centre-sampled replacement otherwise differ in both geometry and smoothing.

For an axis-aligned affine lateral map G, map the native pixel cell [x-0.5,x+0.5] by [y-0.5,y+0.5] through G after adding (dx,dy). Integrate the piecewise-constant source teacher using exact source-cell overlap lengths. Source pixel cells use the same index-centre convention. Carry the source validity mask through the very same weights; require complete valid footprint coverage for the primary value comparison. Do not renormalize a partly missing source footprint and call it fully supported. Numerical tolerance should be fixed, e.g. missing weight no larger than 1e-10 of the full footprint area, with raw values retained.

If the proven mapping is nonlinear, simple axis-aligned box weights are no longer exact. Either derive and verify an appropriate footprint integration or explicitly label a fixed quadrature approximation, test its convergence on the exact saved footprints, and bound its error before target outcomes influence choices. Do not claim polygon-corner integration is exact for a curved mapping.

A point-sampled lateral replay may be included as a separate, predeclared resampling sensitivity result. Do not choose whichever mode makes target transfer look more stable. A public-resize reproduction, unrounded area transport and centre-sampled diagnostic should have distinct names and numerical contracts.

## 4. Handle dz honestly

The primary 2D lateral audit can project each candidate to (dy,dx), deduplicate identical lateral maps for computation, and retain the full list mapping each 3D candidate to its projected map. This provides a precise **conditional lateral-sensitivity** result. It does not evaluate the omitted normal displacement.

Support 6 contains dz=1; supports 4, 5 and 7 have dz=0 only in this particular retained set. Report that distinction. Do not report support 6 as fully stable across 3D alternatives merely because dz was dropped. The remaining sets also do not prove actual ink-face identity.

A supplementary inverse-geometry query at native depth 13.5+dz followed by reading the 2D fine teacher is possible only as a stated normal-extrusion assumption: the 2D teacher value is being treated as constant along the fine surface normal. This is a sensitivity proxy, not evidence that its response belongs at that physical depth. It must not silently replace the primary mapping, average native planes 13 and 14 as an alleged exact 13.5 mapping, or convert a depth-aggregated teacher into a volumetric ink label. Without a verified surface/depth meaning or 3D teacher, leave the normal/depth contribution unresolved.

## 5. Metrics that resist a background-only pass

Let q_t be an area-transferred source score divided by 255, called a normalized teacher score unless calibration is independently documented. Let V_t be its valid supervision support after applying the declared source/native masks, before manual overrides. For each support, define C as the intersection of V_t over fit and all alternatives, V as their union, and q_min/q_max over alternatives including fit on C. Do not replace missing target values with zero.

Report all of the following per support, with exact integer counts and denominators:

- Native CT-eligible area, each |V_t|, common |C|, union |V|, support-change fraction |V minus C|/|V|, and lost fit-support fraction |V_fit minus C|/|V_fit|. Empty denominators are null/inconclusive, not zero.
- On C: mean and 95th percentile of range r=q_max-q_min; maximum over candidates of mean |q_t-q_fit|; all individual candidate changes. These summarize finite-set sensitivity, not confidence bounds.
- Fix the threshold to the released label-builder's threshold and comparator after checking its code, with no recalibration on these supports (the retained documentation says 47/255). Define active union U={q_max meets threshold} and active intersection I={q_min meets threshold}. Report |U|, |I|, variable-positive fraction |U minus I|/|U|, and each candidate's positive count, gained/lost positives relative to fit. The union denominator prevents sparse positives being hidden by the background.
- Repeat continuous range/fit-change statistics restricted to U. Also report sum_C r / sum_C q_max, with null when the denominator is zero. This normalized range is a descriptive mass-change ratio, not a probability of error.
- Preserve the actual binary label and supervision-support construction from the released recipe in a second table. The continuous-score analysis is not a substitute for replaying a thresholded training target. If teacher-zero is treated as excluded rather than a supervised negative, preserve that distinction.

Use common support to isolate changed values, and report support changes separately rather than treating the common-support result as the whole answer. Do not compute AP/AUC or call teacher-versus-teacher agreement accuracy. Four nearby supports in one training winding do not support population p-values or independent-pixel confidence intervals.

Manual native labels remain stationary in native coordinates and retain their actual precedence. Report pseudo-only pixels, manual-override pixels and final-target pixels separately. A large stationary manual region can otherwise make the combined supervision look robust while the transported pseudo-labels move substantially. Supplied manual status does not resolve unknown optical alignment/provenance.

## 6. A bounded operational interpretation, not a new registration gate

My preferred report gives the complete descriptive metrics first and does not claim scientific materiality from a universal cutoff. If the root needs a predeclared routing rule, the following is defensible as an expressly chosen engineering tolerance, frozen before pixels:

- Mark a support **operationally sensitive** if support-change fraction exceeds 1%, variable-positive fraction exceeds 5%, OR worst candidate mean absolute normalized-score change on U exceeds 0.05. The 1% threshold addresses changing loss coverage; 5% asks whether one in twenty union-positive target pixels can switch label; 0.05 measures five percentage points of normalized score. None is empirically calibrated to downstream AP.
- Mark it **insufficient foreground/support evidence** if fewer than 100 active-union pixels remain on common pseudo-only support, if any required denominator is empty, or if the coordinate/depth assumption required for the intended claim is unresolved. The 100-pixel minimum is a reporting floor (five pixels at the 5% tolerance), not 100 independent observations. Show the numerical result even when inconclusive.
- Mark **within the declared lateral tolerance** only when all three operational thresholds are met and sufficient foreground/support is present. Across all four supports, report the worst status; do not use majority voting. Report normal/depth uncertainty separately, including the known dz=1 issue.

For soft BCE, the per-pixel derivative with respect to the student logit is p-q; changing q therefore changes this derivative by exactly the target difference. This gives continuous score change an objective interpretation if those scores are actually used as soft targets, but does not bound learned-model performance. For thresholded BCE, a switched binary label changes that derivative by one. This is why both continuous and thresholded outcomes are useful and must not be conflated.

Outcome consequences:

- **Sensitive:** the target moves enough under the fixed CT alternatives to justify designing an uncertainty treatment with support-matched and smoothing/permuted-uncertainty controls. It does not yet show that such a method will help.
- **Within lateral tolerance:** defer elaborate lateral-uncertainty modelling for this checked support and prioritize a matched acquisition comparison, once depth meaning, source provenance, physical split and adequately powered training are independently settled. It does not admit the whole winding or itself authorize training/GPU.
- **Inconclusive:** preserve the finite result and state the exact absent prerequisite. A nearly blank target or dominant manual override is not evidence that registration is unimportant. Do not replace supports after seeing this outcome.

All original CT gate failures remain failures under every outcome. No averaged-target training, new inference, new crop search, or GPU job is part of this audit.

## 7. Evidence to preserve

Freeze source hashes, code hashes, exact displacement lists, coordinate equations, pixel-centre conventions, footprint weights/support masks, score/label semantics and numerical tolerances before decoding target regions. Verify sign with an asymmetric synthetic pattern, exact area weights with constant and off-centre step patterns, and missing-support behaviour with a hole that cannot become a negative label. An independent direct overlap sum on selected footprints is a useful numerical check.

Retain all four supports and all alternatives, per-candidate metrics and fixed-transform panels with no fitted contrast. A panel is explanatory; the frozen metrics govern the stated operational labels. No visual preference may remove a candidate. Preserve any source-provenance failure as the final outcome rather than back-filling it with an assumption advertised as verified.

## Addendum: root's final feasible operator proposal, before target access

The provenance agent now reports a declared full-fine teacher canvas supported by pinned v42 generation source and matching TIFF header, with no crop/pad/orientation change. Historical content identity is not established, and current manual-label counts differ from the historical builder report. Consequently, the immediate calculation is a **current released-teacher-only diagnostic**, not a historical merged-target reproduction. Omit stationary manual labels entirely from its primary endpoint and preserve this scope distinction.

The root proposes to reuse the exact retained geometry inverse queries, pool the source teacher in fixed 4 by 4 blocks without rounding, then bilinearly sample that pooled field at fine-group2 UV coordinates. This is defensible as an explicitly fixed approximate target-resampling operator; it is not exact mapped native-pixel area integration. Freeze block origin, decoding order, float precision, bilinear support convention, and source-mask pooling before any target metrics. The source-frame control and corrected mapping must use the same operator; a historical cv2 calculation, if included, is an additional contrast and cannot substitute for that matched-operator control.

Use native j14 as the primary conditional layout if its annotation convention is source-documented, not because target pixels favour it. Separately retain all eight already-scored native reference depths j10:18. For each j, transfer the fit and every frozen candidate using that j (candidate queries include j+dz), then compute its own support intersection, envelope and metrics. Report the worst metric over the eight depths in a separate column, with the responsible j and candidate. Do not take a joint max over depths and candidates and call it purely lateral correspondence sensitivity. All this assumes the 2D teacher extends along the fine normal; normal displacement remains physically unresolved. Stable scores across those queries cannot certify a correct ink face or the full CT correspondence.

For normalized scores under this unrounded operator, explicitly define whether the descriptive binary threshold is q >= 47/255. This is not numerically identical to released uint8 area resize followed by integer thresholding, due to rounding. The report must not call it the exact historical training target.

The recommended metrics and optional routing labels in sections 5–6 still apply per support and per j. Primary scientific output remains the finite descriptive sensitivity table, not a calibrated materiality claim. An area-based follow-up can subsequently test dependence on the chosen smoothing operator if the conditional result makes such a test worthwhile; the current result must be labelled conditional without presenting operator-independent stability.
