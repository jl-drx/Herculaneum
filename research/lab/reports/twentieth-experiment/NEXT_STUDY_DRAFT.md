# Next study after phase 20: establish preservation support and source geometry

**Recommendation: retain seed42, complete the small Frag6 structural check, and audit the training objective's preservation support before another adaptation experiment.** Phase 20 answered its matched feature question: explicit depth beat raw features but failed baseline retention. A GPU would accelerate the same unresolved research problem.

This is a post-result recommendation, not a preregistered or execution-ready scientific protocol. No additional training, development selection or scientific acquisition was performed under it. Root retains scientific review.

## First: bounded Frag6 structural work

Implement only Stage A of [the retained plan](frag6-plan/GEOMETRY_ACQUISITION_PLAN.json): the first 1,024 bytes of result.ppm, three classic TIFF headers and IFDs, and at most one bounded ImageDescription per TIFF. Maximum 13 sequential requests and 202,301 reserved response-body bytes; no retries, redirects or full-object fallback. Require unchanged object validators and exact ranged responses, retaining failures. Test oversized/truncated/ignored-range responses, changed validators, malformed TIFF offsets/types/counts, absent descriptions and bounded parsing using synthetic fixtures before real requests. Freeze source and the derived execution plan and perform root review before running.

The result should be a literal structural contract: canvas, byte layout, declared descriptions and explicit missing fields. Pixel bytes incidentally included in structural buffers stay uninterpreted. Do not infer a zero plane, step size, native scan identity or normal orientation from file count or plausible coordinates. Stages B/C are separate gates; C still lacks the declared source UUID/frame, raster map, offsets, normal handling, interpolation and quantization. Do not automatically escalate acquisition because Stage A succeeds.

Frag6/PHerc51/PHerc0051 and all scan aliases remain reserved from training and teacher fitting. A structurally readable package is not yet a valid inference input, authenticated annotation lineage or an independent confirmation set.

## In parallel: audit preservation coverage using training data only

The current KL protects only manual-labeled outputs. It does not constrain every prediction on the training patch, and changing a shared representation can affect outputs beyond the supervised support. This makes preservation coverage a testable hypothesis, not a demonstrated explanation for phase 20's regional losses.

Audit the retained PHerc0139 schedule and existing full teacher logits without new development predictions: quantify manual coverage, repeated exposures and trustworthy CT support; distinguish unlabeled pixels from padding, unmapped surface and invalid source support. Do not reinterpret an absent ink label as background. Freeze any valid-output support from source/data contracts alone. If the retained data cannot establish it, record that blocker before proposing a broader teacher objective.

Use synthetic fixtures to show that a proposed broader KL gives the expected output gradients on supported unlabeled pixels, zero gradients outside support, and leaves supervised BCE restricted to manual labels. Demonstrate reductions and per-batch weights explicitly so a support change cannot silently change loss scale. No real scientific model call is needed for this preparatory work.

## Conditional follow-on experiment

If support can be authenticated, a useful next controlled comparison is **manual-only versus valid-training-output teacher preservation**, using the same explicit-depth architecture, fresh matched initialization and unchanged schedule/optimizer. Both arms would keep the same manual supervised term. Specify in a separate frozen protocol whether KL is an equal mean over valid support or a stratified mean, its weight, numerical tolerances, all stopping rules and meaningful effect margins before either run. This draft deliberately does not leave those choices to the observed Paris results.

That comparison changes preservation support, not depth encoding. It must therefore have an objective-specific decision rule and cannot inherit phase 20's feature-advantage interpretation. Include unchanged seed42 as reference and require preservation against it in every declared development region. Report losses on each arm's objective and on shared manual/full-support diagnostics; objectives with different supports are not directly comparable as a single training-risk win. Broader teacher agreement can preserve teacher errors, so a lower KL alone is not success.

Use no Paris images, aliases, labels or predictions in gradients, teacher fitting, support construction or cutoff fitting. Keep the existing six regions explicitly marked reused DEVELOPMENT if subsequently evaluated. Freeze final-iteration evaluation and report both final arms in all regions even if training is unfavorable; no intermediate-state search or favorable-region selection. A new study is still exploratory because this hypothesis was motivated by prior development outcomes. Do not present it as confirmation or infer independent-pixel significance.

Do not sweep learning rates, KL weights, feature scales, depth windows or seeds against these six regions. Do not warm-start from either rejected phase 20 adapter. If preservation improves only by eliminating meaningful adaptation, report that tradeoff. A genuine efficacy claim will eventually need an appropriately qualified, prospectively locked parent-level evaluation with source geometry and upstream ancestry limitations stated.

## Resources and research scope

The immediate work is bounded CPU implementation, structural metadata and training-only audit. No paid resource is justified yet. Before any first GPU pilot: a useful frozen workload, pinned CUDA environment, current pricing with currency/tax/overhead margin, independent shutdown and verified bucket-scoped R2 worker delivery must all be established. The cap remains £10 TOTAL across failures/retries. Temporary credentials conservatively expire 2026-10-15 09:58:59 UTC; worker import remains unverified.

Preserve source attribution and retained use restrictions. Research progress takes priority over prize optimization, but source/license and any later submission requirements still apply. No uploading, publishing, outreach, new terms acceptance or prize submission is included in this draft.
