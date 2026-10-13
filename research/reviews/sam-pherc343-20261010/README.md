# Assessment of Sam's PHerc343 review

10 October 2026. **The calculations reproduce, and the interpretation supports completing the unchanged four-model recipe.** The strongest finding is sensitivity to padding and the resulting sampling grid/context. The ownership switch creates measurable local discontinuities, but its spatial extent does not explain the candidate structures in these saved maps. Neither consistency nor disagreement establishes ink accuracy.

This assessment was requested while phase46 was running. It uses Sam's six-map snapshot at `f85c6ab`, the corresponding retained lab files and the original author code. It is a review of existing predictions, not another inference experiment or a change to the frozen endpoint. All 18 previously published candidate boxes and their supplied panels were inspected. They are exposed reproduction references, not an untouched evaluation set.

## Independent checks

[VERIFICATION.json](VERIFICATION.json) records **PASS: 1,594 numeric comparisons**, including all 18 regions and 54 CSV rows. The largest absolute difference was `7.771561172376096e-15`. The independent [checker](verify_review.py) uses integer joint histograms and exact integer moments, rather than executing Sam's floating-vector analysis. It also verifies all six encoded/decoded map hashes, their receipts, mask and reference conventions. No model forwards were performed.

[SNAPSHOT_CHECK.json](SNAPSHOT_CHECK.json) verifies the retained results/reference archives and source metadata against the workspace snapshot. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) identifies the uploaded ZIP and all ten original files, preserved byte-for-byte under [original/](original/). This confirms computational agreement on the same data; it is not a second biological reference or independent verification of when the supplied analysis plan was written.

| Model | Plain/padded correlation | Mean absolute difference, uint8 levels | Mean merged ownership-edge gradient |
| --- | ---: | ---: | ---: |
| dnative | 0.946475 | 6.28247 | 6.38948 |
| seed42 | 0.776461 | 10.37393 | 7.44474 |
| seed43 | 0.785210 | 9.75315 | 8.97683 |

The first two columns use 8,887,441 valid published-crop pixels. The last uses the same 2,244 full-field valid edges in each map; source-phase means at those edges are 0.878–1.000. The crop contains 2,159 of those edges. Keeping those populations distinct matters.

## What the findings change

**Padding robustness deserves a controlled follow-up.** The author merge code confirms Villa alignment `padded[40:2600, 200:6920]` and the ownership rule selecting padded predictions for source node IDs 1 and 4. Our retained `REAL_INPUT_VERIFICATION.json`, whose hash matches the snapshot, records matching aligned native CT interiors across all 66 layers and zero differing aligned Hecate voxels. We rechecked that receipt's identity here; we did not repeat the full CT-volume comparison. This supports attributing the saved prediction differences to the padding/sampling/context pipeline rather than a different aligned CT interior. It does not isolate one internal mechanism, establish optical correspondence or verify the biological ink surface.

The sensitivity is broad. Near versus away from ownership boundaries, crop MAEs are 6.32/6.28 for dnative, 7.25/10.40 for seed42 and 9.07/9.76 for seed43. These spatially different populations are descriptive, not a causal control, but they do not support concentrating the phase sensitivity at ownership boundaries.

**The ownership discontinuity is real and local.** Boundary endpoints occupy 4,268 valid crop pixels (0.0480%); their 20-pixel neighborhood occupies 84,519 (0.9510%). Only box5 intersects this neighborhood: 333 boundary endpoints and 8,973 neighborhood pixels, or 3.76% of valid box5 pixels. The supplied overlay places the boundary at its lower-right margin, away from the main highlighted structures. Calling these ownership-switch discontinuities is more precise than treating them as evidence that generic tile seams produced the candidates. The evidence does not support that explanation.

**Greater stability does not select a better model.** Merged-map correlations are 0.582645 for dnative/seed42, 0.552099 for dnative/seed43 and 0.529680 for seed42/seed43. All-valid correlations can be dominated by broad background structure; the regional inspection matters. Components recur across the panels, but connectivity and prominence differ, with particularly weak seed agreement in boxes3, 10 and 15. Dnative often looks broader and more diffuse. Stability alone could reward smoothing or a constant output. These findings do not justify increasing its weight or dropping a seed. Shared training and differing calibration further limit ensemble-independence claims.

The reference is an annotated, displayed four-model composite; the inspected individual maps are different objects. Their visual resemblance or difference cannot provide AP, calibrated ink probability, authenticated letters or numerical reproduction of absent author prediction maps.

## Relationship to the current run

The dated [run observation](CURRENT_RUN_OBSERVATION.json) is separate from Sam's snapshot. At that observation all six Villa maps and Hecate plain were complete; Hecate padded was still progressing. Hecate plain passed the earlier morning file/decoded-pixel integrity check, but a completed plain map alone does not settle Hecate's phase sensitivity or the final composition. The final inference and image-comparison records were not yet available. No complete reproduction claim is made.

All seven files pinned by the executing pipeline were unchanged at this review's verification and observation. There was no restart, interruption, new inference, weight/depth/display adjustment or GPU expenditure for this assessment. Partial-map inspection is now recorded explicitly; the earlier preparation note was written before this inspection.

## Next decision

1. Complete the original eight maps and fixed author composition, then assess the entire published area and all 18 boxes with their surroundings. Keep the existing endpoint and stopping rules.
2. Extend this saved-output audit to the completed Hecate pair and four-model composition. Hecate uses its own 9.6 µm grid: offsets are **36 rows and 180 columns**, with the author's nearest-neighbor ownership-mask resize. Reusing the Villa offsets would be incorrect. Report these as additional diagnostics, not a replacement success criterion.
3. If the full assessment warrants a robustness experiment, freeze its offsets, supports, baseline, display and structure-retention checks before additional inference. Do not optimize weights, contrast or phase choices on these already inspected boxes. Averaging or consistency learning must demonstrate useful preserved structure as well as stability; a smoother map alone is insufficient evidence of progress.

No new robustness run is launched by this review. The unchanged four-model result remains the next scientific decision point.

## Replay this arithmetic audit elsewhere

Restore the PHerc343 results and reference bundles and the core source metadata using the [workspace instructions](../../workspace/README.md). Place the verified `valid.npy` at `reports/fortysixth-experiment/work/valid.npy` under the restored lab root. The existing private Actions review artifact supplies that mask without downloading the CT volumes; model checkpoints are not required. Use a Python environment with NumPy, SciPy and tifffile (the tested PHerc343 runtime has these).

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /path/to/python -B verify_review.py \
  --lab /path/to/restored/lab \
  --output /path/to/new-verification.json
```

Run from this directory or provide the full script path; `original/` is located relative to the script. Choose a fresh output path outside the active study. `--pipeline-pins /path/to/INPUTS.json` optionally checks the live pipeline's frozen files; that runtime manifest is not needed for replaying the archived maps. Recorded PASS is on the current machine, not a separate physical machine.

The original `analyze.py` retains Sam's absolute paths unchanged. It generates the overview and three regional contact sheets, but the archive does not include the generator for `region5_ownership_boundary.png`. The supplied overlay is preserved and its boundary counts were independently recomputed; pixel-identical regeneration of every review figure is not claimed.
