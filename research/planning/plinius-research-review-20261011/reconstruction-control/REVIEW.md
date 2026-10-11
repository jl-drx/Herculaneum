# Independent review of the same-scan reconstruction control

11 October 2026. Reviewed by a separate research agent from the implementation author. **The frozen execution and reported arithmetic pass review. The data support a useful local reconstruction/export comparison, but site2 does not pass the frozen exact-pair screen against the usual reconstruction. No physical one-voxel displacement is established.**

I inspected `PLAN.md`, both scripts, request plan/ledger, source freeze, self-check record and results; independently recomputed all nine comparisons from the saved bytes; and viewed all three fixed orthogonal panels. I did not import the author's analysis code, fetch more data, run inference, alter frozen files, resample a volume or choose a new region. Numerical review ran with one numerical thread. The independent checker and its machine-readable result were subsequently preserved at the parent's request; that replay repeated the same calculations without adding scientific comparisons.

The saved [verify_independently.py](verify_independently.py) requires Python3.12+, NumPy and SciPy; the verified run used Python3.12.14, NumPy2.5.3 and SciPy1.18.1. It forces one numerical thread, reads only saved objects/results and emits JSON to stdout. Keep the sibling `metadata/` directory alongside this directory when replaying. Run `python verify_independently.py > /tmp/reconstruction-verification.json`; compare with [INDEPENDENT_VERIFICATION.json](INDEPENDENT_VERIFICATION.json). The receipt records source hashes, runtime versions, exact checked values and limitations. It does not automate the visual panel review.

## Protocol and source checks

- All five frozen source-file SHA-256 values match. The recorded freeze time precedes the first recorded chunk request; this is a consistent local provenance record, not an externally timestamped attestation.
- The retained catalog excerpt matches the request plan's hash. Each selected landmark is exactly the corresponding first, second or third recorded destination of the fine78-to-native113 transform. XYZ reversal, containing chunk index and whole-chunk bounds reproduce independently. All three complete chunks avoid the catalog's warned masking ranges.
- All nine planned URLs, names and identities match the ledger in order. All nine saved bodies have the stated SHA-256 and exactly2,097,152bytes, totaling18,874,368bytes. There are nine successful recorded chunk reads, no replacement sites and no recorded retries. Separately fetched metadata is outside this stated chunk-byte total, as the plan says.
- The three retained Zarr metadata records support the decoder: uncompressed, unfiltered, C-order uint8,128³ chunks, slash-separated chunk keys. This is not a compressed byte stream mistakenly reshaped as intensities.
- All three volume pairs were reported at every site. The saved panel-generation code uses every site's fixed midplanes, `vmin=0`, `vmax=255`, nearest interpolation and no fitted contrast. The optional local lag cube/FFT band study was not part of the frozen plan and was not needed to fulfill it.

## Independent numerical reconstruction

For intensity Pearson, I accumulated a256×256 joint histogram and used uncentered weighted moments:

`r = (N*Sxy - Sx*Sy) / sqrt((N*Sxx-Sx²)*(N*Syy-Sy²))`.

This differs from the author's explicitly demeaned voxel-vector dot products. Tie-corrected Spearman used marginal histogram midranks in the same weighted-moment formula. MAE used the joint histogram weighted by `abs(i-j)`. Gradient checks used `np.diff` and a511×511 histogram of paired signed differences on the independently constructed common-valid edges, rather than concatenating edge vectors. Cube means, variance, nonzero/code255 fractions and all seven reported quantiles were recomputed from histograms/order statistics.

For phase correlation, I used full complex SciPy FFTs with one worker and normalized the cross spectrum using `exp(i*angle(cross))`, excluding exactly zero magnitudes. This independently reproduces the author's real-FFT/division implementation. The far-peak exclusion used modular grid distances rather than the author's indexed cube. It retains the **same frozen estimator**, not a newly tuned registration method.

| Quantity | Independent result |
|---|---:|
| Nine full/common-support Pearson comparisons | Maximum absolute discrepancy1.64e−12 |
| Nine full/common-support Spearman comparisons | Maximum absolute discrepancy7.27e−13 |
| All full/common-support MAEs | Exact agreement |
| All common-support counts and mask-agreement fractions | Exact agreement |
| Nine valid-edge counts | Exact agreement |
| Nine pooled gradient correlations | Maximum absolute discrepancy6.02e−14 |
| All nine phase-peak integer coordinates | Exact agreement |
| Phase-peak values | Maximum absolute discrepancy5.56e−17 |
| Far-peak values | Maximum absolute discrepancy1.05e−17 |
| All nine cube summaries and quantiles | Agreement within1e−11, with counts/quantiles exact |

No implementation bug affecting the reported results was found. `common_nonzero` correctly means both intensities greater than zero; gradient edges correctly require both endpoints nonzero in both cubes. Gradients are signed, computed in int16 before subtraction, and pooled across all three axes. They are not independently estimated per-axis displacement or a surface-normal gradient statistic.

## Site2: a real estimator disagreement, not established physical movement

The prescribed shift **to apply to the sibling array** is `[+1,0,0]` in ZYX for usual-versus-pag0 and usual-versus-pag50 at site2. All seven other comparisons peak at zero, including pag0-versus-pag50 at site2. The independent complex-FFT calculation confirms these exact peaks.

The following zero-lag values are an explicitly supplementary review of the already defined correlation surface; no inputs, filter, search range or coordinates were changed:

| Site2 comparison | Zero-lag phase score | Chosen phase peak | Zero/peak |
|---|---:|---:|---:|
| usual / pag0 |0.259347278|0.438672908|0.591209|
| usual / pag50 |0.250274576|0.420615488|0.595020|

Thus this is not a rounding tie between adjacent integer peaks. But the printed peak ratio20.97/23.06 is the ratio to the strongest peak **outside a radius-two cube**, which excludes zero lag here. It is not a20-fold preference over zero displacement, nor a confidence interval. The uniqueness check also does not quantify physical registration uncertainty.

Phase retrieval, differently shaped edges, export clipping and the fixed Hann window can affect phase correlation when the images are not translated copies. These reconstructions share projection noise and nominal coordinates. A nonzero peak therefore blocks treating site2 as an exact pair under this protocol, but cannot by itself overturn the publisher's coordinate-system claim. A global-Z voxel is9.362µm; it is not necessarily a displacement along the papyrus normal. Do not apply an inferred +1 correction or recenter an ink window from this result.

## All three panels contain corresponding structure

I viewed every supplied panel, including all three reconstruction rows and all three orthogonal planes at each site:

- **Site1:** corresponding layered bands and a bright irregular inclusion/edge are visible across all variants. The usual reconstruction appears smoother; both siblings show stronger fine edge structure.
- **Site2:** corresponding bent sheets, a large dark region and a granular/inclusion-rich region remain recognizable across all three. The panels support gross correspondence; they do not resolve the interpretation of the one-voxel phase peak.
- **Site3:** the curved sheet configuration and bright irregular central feature recur. Again the usual reconstruction appears smoother, while the siblings are very similar to each other.

This agreement is not explained solely by an all-zero background: common nonzero support occupies approximately76.2–87.6% of these cubes, and sibling gradient correlations remain0.98165–0.98516 on edges excluding zero endpoints. It still does not establish independent truth, fine-to-native registration, correct wrap identity or ink. Bright features in these raw CT panels are not being labeled as ink.

## One important metric-label caveat

The frozen JSON key `saturated255_fraction` computes the literal fraction of samples equal to255. The computation is correct; interpreting it as all saturation or clipping would be incorrect. Every pag50 cube has a maximum observed code of254, so its zero code255 occupancy cannot demonstrate absence of clipping.

As a descriptive review supplement, literal code254 occupancy in pag50 is0.559616%,2.529049% and0.767565% at sites1,2 and3 respectively. These values are computed on the saved arrays and do not identify which voxels were clipped. Keep the frozen key unchanged and explain its literal meaning. Similarly, zero intensity may represent intensity conversion/clipping or masking; the binary nonzero masks are observed-intensity supports, not validated tissue masks.

## Scientific interpretation and next decision

The pag0/pag50 siblings are very similar at these sites: full-cube Pearson0.98568–0.98769 and zero phase peak at all three. The usual-versus-sibling correlations0.72789–0.83160 and visible appearance changes show that reconstruction/export is a substantive input factor, not merely a filename change. The catalog's different conversion windows and possible detector-distortion handling prevent attribution to one Paganin parameter. Neither high correlation nor visibly sharper edges establishes improved ink information.

Use this as evidence that the public same-acquisition variants can support a controlled local study. Sites1/3 are consistent with zero integer displacement under the chosen diagnostic; they are not subpixel-validated labels. Site2 remains a documented exception to exact-pair admission. Stop this control as frozen: no corrective warp, adaptive chunk search, model inference or changed acceptance threshold. Any future surface-render comparison should fix geometry and contextual support independently, preserve the intensity/export confounds, and use an explicit reference rather than selecting a reconstruction because its prediction looks brighter.
