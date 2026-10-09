# Fixed cached CT contrast audit

The fixed local-contrast feature does not expose a consistent shared depth signal in these four supplied-reference regions. On original support, neither polarity exceeds AUROC 0.6 at any of the 63 single layers or 21 depth triplets in any context. Median AUROC stays near 0.5. Intensity profiles vary more with depth and region, with changing polarity and no selected winning layer.

This is an inconclusive diagnostic for choosing a learned representation. The scalar contrast feature also performs weakly in the two regions where the retained CT backbone ranks labels well; therefore its failure cannot establish absent CT ink or dismiss learned spatial/depth structure. Follow the protocol’s inconclusive branch: prepare a concrete geometry/reference review before another model or depth sweep. Retain unchanged seed42 and the original negative averaging decision.

## Fixed experiment

All four previously scored PHerc51/Frag6 contexts were included, one parent. Each retained 63×384×384 uint16 crop produced 63 spatial 3×3 means and 21 nonoverlapping spatial/depth 3×3×3 means. The two features were intensity and intensity minus a fixed 9×9 coarse-pixel mean. Exact integer numerators were used to preserve ties, with real margins around the central 64×64 scoring area. Both signs and the original/interior supports were fixed before execution. No new labels, fitted thresholds, selected depths, model forwards, training or acquisition.

Historical layer numbers and retained TIFF intensities are not authenticated physical ink depths or calibrated attenuation. Phase30’s local reproduction of the historical interpolation behavior does not establish a globally correct optical warp or anatomical normal. Unknown teacher/model ancestry and annotation provenance remain unresolved.

## Original support: every context and both feature scales

Table values are medians across all depths, never the best depth. “+ / −” means the feature and its fixed reversed ordering. AP is prevalence-sensitive; AUROC 0.5 is the no-ranking reference. These dependent depth profiles do not supply confidence intervals or independent samples.

| Context | Feature | Ink prevalence | Single median AUROC + / − | Triplet median AUROC + / − | Single median AP + / − | Triplet median AP + / − |
|---|---|---:|---:|---:|---:|---:|
| r544-c4128 | intensity | 0.72854 | 0.48080 / 0.51920 | 0.48144 / 0.51856 | 0.72219 / 0.74757 | 0.72117 / 0.74698 |
| r544-c4128 | contrast | 0.72854 | 0.49606 / 0.50394 | 0.49791 / 0.50209 | 0.73135 / 0.73604 | 0.73078 / 0.73530 |
| r2080-c3616 | intensity | 0.86391 | 0.49587 / 0.50413 | 0.50230 / 0.49770 | 0.86174 / 0.86299 | 0.85896 / 0.85786 |
| r2080-c3616 | contrast | 0.86391 | 0.49769 / 0.50231 | 0.49939 / 0.50061 | 0.86266 / 0.86134 | 0.86064 / 0.86096 |
| r4640-c1056 | intensity | 0.03169 | 0.44080 / 0.55920 | 0.44651 / 0.55349 | 0.03094 / 0.03906 | 0.03178 / 0.04039 |
| r4640-c1056 | contrast | 0.03169 | 0.48780 / 0.51220 | 0.49049 / 0.50951 | 0.03364 / 0.03415 | 0.03259 / 0.03421 |
| r6688-c1568 | intensity | 0.09604 | 0.51800 / 0.48200 | 0.50466 / 0.49534 | 0.10495 / 0.09467 | 0.10203 / 0.09218 |
| r6688-c1568 | contrast | 0.09604 | 0.49852 / 0.50148 | 0.50816 / 0.49184 | 0.09741 / 0.10088 | 0.09791 / 0.10169 |

For local contrast, all 672 original-support feature/sign/depth/context combinations have AUROC ≤0.6. This descriptive cutoff was frozen; it is not a statistical test or model acceptance threshold. Triplet curves broadly follow the single-layer curves in the complete plots; this visual observation does not prove that averaging preserves every useful learned feature.

[Original AUROC profiles](figures/original-auc-profiles.png) · [Original AP profiles](figures/original-ap-profiles.png)

## Fixed interior support

The >4-pixel interior masks are exactly those from phase33. Some isolated local-contrast depths exceed AUROC 0.6 in the two sparse regions, but the first two contexts do not; there is no common consistent contrast profile across all four. All isolated responses are retained, without choosing a depth or sign. Only 37 and 167 ink pixels remain in the two sparse interiors; adjacent pixels are dependent.

| Context | Scale | Ink prevalence | Contrast median AUROC + / − | Depths AUROC >0.6, + / − | Contrast median AP + / − |
|---|---|---:|---:|---:|---:|
| r544-c4128 | single | 0.76899 | 0.49713 / 0.50287 | 0/63 / 0/63 | 0.76981 / 0.77775 |
| r544-c4128 | triplet | 0.76899 | 0.49683 / 0.50317 | 0/21 / 0/21 | 0.76944 / 0.77865 |
| r2080-c3616 | single | 0.90939 | 0.49492 / 0.50508 | 0/63 / 0/63 | 0.90881 / 0.91114 |
| r2080-c3616 | triplet | 0.90939 | 0.49865 / 0.50135 | 0/21 / 0/21 | 0.91001 / 0.91087 |
| r4640-c1056 | single | 0.00959 | 0.51218 / 0.48782 | 3/63 / 6/63 | 0.01225 / 0.01092 |
| r4640-c1056 | triplet | 0.00959 | 0.51068 / 0.48932 | 1/21 / 2/21 | 0.01241 / 0.01169 |
| r6688-c1568 | single | 0.04744 | 0.50969 / 0.49031 | 3/63 / 0/63 | 0.04742 / 0.04845 |
| r6688-c1568 | triplet | 0.04744 | 0.50429 / 0.49571 | 1/21 / 0/21 | 0.04687 / 0.04773 |

AP comparisons between original and interior support also change prevalence and sampled pixels. They cannot isolate an effect of boundary removal. The full summary export includes intensity and contrast on both supports, each sign and scale, interquartile ranges and both descriptive AUROC counts.

[Interior AUROC profiles](figures/interior-auc-profiles.png) · [Interior AP profiles](figures/interior-ap-profiles.png)

## Class distributions and visual review

The supplied ink and clear classes have substantially overlapping local-contrast distributions. In the later retained layers the plotted intensity distributions converge and contrast spreads narrow in all four regions. This is descriptive evidence of changing spatial structure in these arrays; it does not identify an ink-bearing surface, justify choosing a cutoff or authenticate which side of the papyrus is sampled.

[Original class distributions](figures/original-class-distributions.png) · [Interior class distributions](figures/interior-class-distributions.png). Curves show medians and shaded interquartile ranges, not confidence intervals. Feature axes are shared across contexts within each figure. All six figures were inspected by root; no separate agent review is claimed.

## Verification and resources

All 2,688 metric records are defined; none were dropped. Independent strided-slice sums and direct 81-offset box sums reproduced all 2,752,512 feature numerators exactly. There were 1,680 scalar spatial/depth pooling checks. All 1,376,256 prior pooled FP32 values were reproduced under the original scale contract. Signed-integer ranking, independent of the worker’s float encoding, reconciled all 5,376 AP/AUROC values with maximum difference 2.22×10⁻¹⁶. All 1,344 distribution records (10,752 summary values) and all 64 profile summaries were verified; maximum distribution difference was 1.46×10⁻¹¹ retained TIFF units.

Seven synthetic tests passed, none failed, errored or skipped. The initial shell launch was rejected because its new working directory did not yet exist; no command or scientific run started. The directory was created, then setup/tests succeeded. This is preserved in SETUP_RECEIPT.json. The scientific worker completed its sole attempt in 2.329 seconds, peak RSS 92,049,408 bytes; independent verification took 1.751 seconds, peak RSS 97,697,792 bytes. Both meet the 120-second/1-GiB limits. No model inference, optimizer updates, source requests, R2 uploads or paid resources occurred.

Raw metric records: [all-metrics.csv](all-metrics.csv). Every profile summary: [all-profile-summaries.csv](all-profile-summaries.csv). Class distributions: [all-class-distributions.csv](all-class-distributions.csv). [Verification](VERIFICATION.json). The frozen [protocol](PROTOCOL.md) and [GO](GO.json) identify exact sources and execution code.

## Decision

Keep the baseline and stop feature/depth sweeps on these four contexts. These scalar features do not establish a representation change worth scaling. Equally, they do not rule out useful learned texture or joint depth/spatial evidence. No model promotion, recovered text, translation or independent accuracy claim follows.

Next prepare a cached structural review packet that separates source-value reproduction, physical surface/depth interpretation and optical registration. Include all four regions, exact coordinates and the complete layer sequence or a prospectively fixed uniform montage, with a structured landmark form. Only genuinely shared non-ink landmarks or attributable curator evidence can strengthen the registration claim; if neither exists, keep it unresolved. A proposed correction would need held-out landmark checks. See the [next study draft](NEXT_STUDY_DRAFT.md). It has not been executed.

GPU spend remains £0 and the £10 total first-pilot cap is unchanged. No maintainer request has been sent; no new reply was supplied. The other parent reserves remain untouched. R2 worker delivery remains unverified; a paid launch still needs a useful frozen workload, pinned CUDA, current all-in pricing and independent shutdown.
