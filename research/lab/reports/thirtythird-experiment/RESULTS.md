# Frag6 cached reference-sensitivity diagnostic

Small raster shifts and exclusion of uncertain label edges do not resolve the two weak regions within this fixed diagnostic family. The supplied optical image agrees closely with the supplied labels, while the CT model still ranks many clear pixels above ink pixels in those regions. Keep the unchanged seed42 baseline; phase32’s negative averaging result and primary scores are preserved.

This is post hoc analysis of four previously scored development contexts from PHerc51 (Frag6), one parent. Optical imagery and annotation may share provenance. Their agreement is not independent ink truth, benchmark qualification or recovered text. No model was trained or called in this phase.

## Original support

AP is average precision; its level depends on prevalence. All predictors in each row use the same known support. The optical column is a descriptive consistency check, not a competing CT model.

| Context | Ink / clear | Ink prevalence | Baseline AP | Averaging AP | Optical AP | Baseline AUROC | Optical AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| r544-c4128 | 2920 / 1088 | 0.7285 | 0.97414 | 0.98664 | 0.99850 | 0.93895 | 0.99593 |
| r2080-c3616 | 3485 / 549 | 0.8639 | 0.99293 | 0.98912 | 0.99923 | 0.95710 | 0.99510 |
| r4640-c1056 | 129 / 3942 | 0.0317 | 0.11756 | 0.09323 | 0.95856 | 0.87969 | 0.99852 |
| r6688-c1568 | 388 / 3652 | 0.0960 | 0.15700 | 0.15765 | 0.98238 | 0.72394 | 0.99778 |

## All nine label shifts

Every dy, dx ∈ {−1, 0, +1} was evaluated. One coarse pixel is three original raster pixels, not a measured physical registration error. The original 128×128 label canvas supplies the shifted 64×64 core without padding or interpolation.

The common known-support intersection removes 683 of the original 16,153 known pixels. Only 20 of the remaining 15,470 pixels can change class across the nine shifts. Two contexts have no changeable labels at all. Therefore their exactly constant scores are uninformative about alignment; the common-support test removes most of the boundary sensitivity it was intended to inspect.

| Context | Common known | Removed | Potentially changeable | Common-support baseline AP range | Own-support baseline AP range |
|---|---:|---:|---:|---:|---:|
| r544-c4128 | 3748 | 260 | 2 | 0.98489–0.98511 | 0.96105–0.98169 |
| r2080-c3616 | 3833 | 201 | 0 | 0.99625–0.99625 | 0.98976–0.99427 |
| r4640-c1056 | 4008 | 63 | 0 | 0.08879–0.08879 | 0.08392–0.14475 |
| r6688-c1568 | 3881 | 159 | 18 | 0.13443–0.14230 | 0.13533–0.17917 |

The secondary test uses each shift’s own known support. It changes coverage and class prevalence as well as labels. Baseline AP remains 0.08392–0.14475 and 0.13533–0.17917 in the two weak regions. These ranges do not identify a true warp, and no preferred shift or corrected reference is selected.

[All common-support shifts](figures-v2/common-shift-ap.png) · [All own-support shifts](figures-v2/moving_support-shift-ap.png)

## Fixed boundary-distance bands

Distance is measured to the nearest different class or unknown pixel in the conservative supplied mask. Bands are ≤2, >2 to ≤4, and >4 coarse pixels. These are mask distances, not independently authenticated ink edges. Every band retains both classes here.

| Context | Band | Ink / clear | Ink prevalence | Baseline AP | Averaging AP | Optical AP | Baseline AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| r544-c4128 | near_le2 | 190 / 178 | 0.51630 | 0.72337 | 0.77920 | 0.88558 | 0.70710 |
| r544-c4128 | middle_gt2_le4 | 250 / 165 | 0.60241 | 0.92025 | 0.94966 | 0.99852 | 0.86310 |
| r544-c4128 | interior_gt4 | 2480 / 745 | 0.76899 | 0.99597 | 0.99905 | 0.99986 | 0.98765 |
| r2080-c3616 | near_le2 | 182 / 118 | 0.60667 | 0.89339 | 0.89146 | 0.87091 | 0.82283 |
| r2080-c3616 | middle_gt2_le4 | 222 / 124 | 0.64162 | 0.96783 | 0.96039 | 0.99306 | 0.93708 |
| r2080-c3616 | interior_gt4 | 3081 / 307 | 0.90939 | 0.99900 | 0.99776 | 1.00000 | 0.98995 |
| r4640-c1056 | near_le2 | 47 / 53 | 0.47000 | 0.76641 | 0.67829 | 0.83724 | 0.76114 |
| r4640-c1056 | middle_gt2_le4 | 45 / 67 | 0.40179 | 0.78376 | 0.63263 | 0.99414 | 0.83085 |
| r4640-c1056 | interior_gt4 | 37 / 3822 | 0.00959 | 0.02851 | 0.02448 | 1.00000 | 0.84685 |
| r6688-c1568 | near_le2 | 126 / 138 | 0.47727 | 0.53772 | 0.53945 | 0.89028 | 0.59472 |
| r6688-c1568 | middle_gt2_le4 | 95 / 161 | 0.37109 | 0.44451 | 0.48164 | 1.00000 | 0.63138 |
| r6688-c1568 | interior_gt4 | 167 / 3353 | 0.04744 | 0.08882 | 0.09265 | 1.00000 | 0.76832 |

On the >4-pixel interior, baseline AP is 0.02851 and 0.08882 in the weak regions, versus optical AP 1.0 on the identical masks. Only 37 and 167 ink pixels remain, at prevalences 0.00959 and 0.04744. AUROC is 0.84685 and 0.76832: there is ranking information, despite poor AP. The lower AP compared with the original support cannot be attributed to erosion itself because prevalence and sampled pixels changed. Perfect optical ordering in these small interiors does not establish perfect annotation or independent accuracy.

## Visual and numerical review

[All four optical / label / CT / band panels](figures-v2/all-context-panels.png) show the same fixed probability scale. The third region’s prediction extends across clear areas along its right side; the fourth has broadly low probabilities including its annotated ink. These descriptive observations do not identify the cause. All four contexts and every shift are retained, with no region selected for the headline.

All 276 metric records are defined; none were omitted. Separate scalar indexing verified 147,456 translated labels, scalar sums verified 16,384 optical core values, and direct lattice comparisons verified all 16,384 boundary-band assignments without a distance transform. A separate ranking formulation reconciled all 552 AP/AUROC values with maximum difference 4.44×10⁻¹⁶. All eight original CT predictor/context records and the sealed phase32 primary result are unchanged. Six mechanics tests passed with no failed, errored or skipped tests. This is root numerical and visual review, not a separate agent review.

Execution took 1.432 seconds with peak RSS 274,337,792 bytes; verification took 1.458 seconds with peak RSS 251,703,296 bytes. Both fit their individual 120-second / 1-GiB limits. No network requests, model forwards, optimizer updates, R2 writes or GPU usage were added. Plot layout was revised in figures-v2; the first render and script are retained. Repository tracked files and the pre-existing untracked .codex directory were preserved.

Full precision, all predictors, counts, prevalence and both metrics: [all-metrics.csv](all-metrics.csv). Per-shift support changes and arrays: [run-v1/COMPLETE.json](run-v1/COMPLETE.json). [Verification receipt](VERIFICATION.json).

## Decision and next step

This narrow test leaves small rigid shifts and uncertain boundaries insufficient as an explanation under the supplied reference. It does not exclude larger or nonrigid misregistration, incomplete labels, historical renderer effects, depth mismatch or limitations of the model’s representation. It does not establish that CT lacks ink signal.

Run one fixed, cached CT observability audit next: examine depth-local intensity and local spatial contrast through every retained layer and its corresponding depth-triplet mean. Report all profiles without choosing a best depth or sign. Use the outcome to define one representation hypothesis and a prospectively selected, disjoint development replication, rather than another checkpoint sweep on these four regions. The [next study draft](NEXT_STUDY_DRAFT.md) is not yet executed or frozen.

GPU spend remains £0; the £10 total pilot cap remains in force. A paid workload still requires its frozen experiment, pinned CUDA environment, current all-in pricing, independent shutdown and verified worker delivery of scoped R2 credentials. No maintainer request, publication, submission, transcription or translation was made. Independent reference/exposure qualification remains unresolved.
