# Independent assessment of the fixed three-site CT experiment

**The frozen strict screen fails at all three sites, but the evidence now supports local fine/native structural correspondence substantially more strongly than an undifferentiated “alignment failed” conclusion would suggest.** Keep the failures and distinguish their causes. Site1 is suitable for a bounded follow-up about correspondence uncertainty and its consequences, not immediate admission as perfectly paired training data.

## Numerical review

`replay_real_scores.py` independently recalculates all52,728 correlation values from `native_full.npy` and `fine_remapped.npy`, using directly centered spatial windows and dot products. It imports none of the authored scoring code and uses neither FFT correlation nor integral-image variances. All saved output hashes passed. Maximum difference across all scores is5.04e−13. Every fit, competitor index, physical residual, numerical criterion and final failure decision reproduces under the final1e−6 arithmetic tolerance. Results and three independent landscape arrays are retained in this directory.

This verifies calculations on the saved remapped arrays; independent raw-chunk assembly and all-pixel resampling are outside this particular check. The separate81-query scalar geometry check validates a finite subset of the coordinate pullback, not physical registration.

| Site | Baseline fit-support NCC | Optimized fit-support NCC | Main checking limitation |
|---|---:|---:|---|
|1|0.93279|0.95103|Two broad tangential alternatives have margins0.03480 and0.03130, below0.05.|
|2|0.52878|0.94456|Two checking optima differ tangentially by2.019 and3.023native voxels; the common fit does not carry across the full core.|
|3|0.66942|0.84167|Several checking supports prefer about one normal voxel and one tangential voxel of additional displacement; the broad single-translation screen fails.|

All12 checking NCC values at the fitted translation exceed0.676. This establishes useful repeatable structural signal; it does not establish accurate ink labels. The integer search itself limits fine discrimination. No shift, crop, filter or threshold was altered after CT viewing.

## Why site1 should not be described as a wrong-sheet failure

Site1's four checking NCC values are0.9490,0.8681,0.9378,0.9262. Three checking optima exactly equal the fitting translation(0,−1,0). The fourth differs by0.986native voxel tangentially and0.014normally. All frozen residual/boundary criteria pass. Only the minimum distinct-peak margin fails.

Both failing competitors have unchanged depth and are two steps away in a tangential coordinate. Strongest competing positions more than two physical normal voxels away are much less correlated: their margins are0.245–0.600 across the four checks. This distinction matters. A broad lateral correlation maximum is imprecise localization; it is not evidence for a competing wrap.

As an explicitly **post-screen descriptive calculation**, `check_near_components.py` examines the already retained values at least NCC-at-fit−0.05. Each check has exactly one connected component containing the fit (11,6,22,4 integer candidates), with no search-boundary contact. The two offending competitors are in that same component. The third checking support's near-score component includes an additional depth step, so it is not a claim of zero normal uncertainty. These sets are objective-level descriptions, not calibrated confidence regions or replacement admission criteria.

## Fixed image review

The reviewer inspected all six fixed0–255 panels: comparison and complete-source-depth context for each of the three sites. The native and fine views visibly share substantial folds, void shapes and lamellar boundaries. At site1, the prominent vertical openings and adjoining bends recur in the layer10/13 views; the crossing central profiles show corresponding nonuniform folds through depth. This is compelling evidence that the intended local tissue structures have been paired. Site2's two-step normal correction visibly brings the large corrugated boundary into better correspondence. Site3 retains more heterogeneous residual mismatch.

There are qualifications. Fine full-depth views reveal thin laminae/voids that the native scan blurs together. Whole-depth context fixes fine UV from the nativej13 mapping and is correctly labelled as contextual rather than an exact paired-ray plot. Current panels show baseline and fitted correspondences, not a dedicated side-by-side identification of every competing normal candidate. They therefore support anatomical correspondence without certifying a particular writing face or assigning all intermediate pixels to one sheet. A brighter fine feature is not thereby ink; no label or prediction map was inspected in this review.

The fitting/checking templates contain essentially no literal zero-valued native samples, so the high scores are not simply agreement on identical zero-filled masks. Nevertheless, void/material boundaries and bulk substrate can dominate correlation. They remain structural evidence, not an ink detector's accuracy measurement.

## One finite next step

Keep the current experiment closed with three failed strict screens. Do not rescue it by lowering the margin or fitting per-check corrections. Use site1, which was already the designated convention-development site, for **one explicitly new uncertainty-consequence study**:

1. Freeze the local correspondence alternatives from the present CT evidence, their physical support and a bounded sheet-review packet before opening targets. Include the strongest normal alternatives; maintain unknown support where the writing-side/sheet cannot be distinguished.
2. Verify the released w040 target coordinate/provenance contract, then compare how transferred teacher/manual training support changes across those fixed alternatives. Preserve pseudo-label versus manual versus unknown status. Do not choose geometry by whichever transform matches targets best.
3. Report stable target coverage and disagreement under the fixed alternatives, with all failed/unknown support retained. This asks whether the observed geometric uncertainty is consequential at the intended supervision scale; it does not yet train or claim an acquisition advantage.

The result changing the next decision is straightforward: if a useful same-sheet support has stable supervision across plausible transforms, proceed to design the simple matched-tissue native-versus-degraded training pilot on that support. If targets vary materially, uncertainty or exclusion is a concrete intervention to compare against matched smoothing and support controls. If distinct sheets remain indistinguishable, stop transfer there. None requires more blind crop search or another tiny training arm.

Sites2–3 remain exposed checking/development observations and document that one common translation does not generalize across each entire256-square core. They cannot become untouched tests or be silently omitted from this study. Their spatial variation would motivate a separately designed local geometry model only after a useful supervision consequence has been demonstrated.

This recommendation preserves the frozen decision while using the information the experiment actually recovered. No model, GPU, teacher/label pixel, active study52 output or remote service was used for this independent review.
