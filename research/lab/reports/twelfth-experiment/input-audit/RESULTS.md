The fixed input audit passed. The observed normalizer behavior does not support a large failure caused by the scalar division by 257 or by a collapsed MAD denominator on these retained contexts. Acquisition, pooling, numeric representation, support and texture differ; this audit does not determine whether those differences explain the Frag4 model results.

Root executed the sealed seven-context diagnostic once: 159 tiles, 209 normalizer calls, 7.364 seconds and 163.762 MiB peak RSS. All bound input identities were checked before and after. This review reads only the resulting JSON. No source-array rerun, label or prediction-map read, checkpoint access, model call or new normalization was performed for the review. The result file SHA256 is `bbca418976914a26b466b0f416797e01403c26e37001f6818ab6ede993a815b5`.

The ordinary normalization trace matched the pinned NumPy implementation exactly on every tile. All 159 used the scaled-MAD branch; none used preclip-standard-deviation, percentile-span or unit-scale fallback. None was an entirely zero raw patch. Median/MAD normalization does not enforce standard deviation 1, so the different normalized standard deviations below are not by themselves a malfunction.

For the 50 Frag4 tiles, the fixed comparison was `normalize_robust(x)` against `normalize_robust(float32(257*x))`. Across 13,926,400 overlapping tile elements, the largest absolute normalized-input difference was **4.2915344×10⁻⁶**. The mean absolute differences were 3.9978×10⁻⁷ and 4.2822×10⁻⁷ for contexts 1 and 2. The comparison is near-invariant, not bit-exact. It tests scalar conversion within this retained float32 representation; it does not recreate source precision, reverse pooling or prove unchanged model scores. The model was not evaluated on the scaled inputs.

The table uses whole selected-17 inputs for raw values, and the full fixed tile grid for normalized values. Zero percentages count voxels, not missing objects. The normalized standard-deviation span is the minimum to maximum across overlapping tiles, not a confidence interval.

| Context | Role | Raw median | Zero voxels | Normalized tile std: min / median / max | Whole-input adjacent-depth correlation |
| --- | --- | ---: | ---: | ---: | ---: |
| Frag4 context 1 | New supplied reference | 94.834 | 10.5813% | 0.517 / 1.606 / 3.132 | 0.840 |
| Frag4 context 2 | New supplied reference | 97.666 | 0% | 1.553 / 1.594 / 1.677 | 0.527 |
| PHerc0139/w016 rank 01 | Adaptation | 46 | 0.000080% | 0.856 / 2.170 / 3.475 | 0.721 |
| PHerc0139/w016 rank 02 | Adaptation | 53 | 0.000479% | 0.730 / 1.060 / 1.725 | 0.678 |
| PHerc1667/w029 | Calibration | 74 | 0.000359% | 0.779 / 0.848 / 2.139 | 0.691 |
| PHerc0814/46527 rank 01 | Calibration | 88 | 2.3634% | 0.688 / 0.757 / 2.267 | 0.712 |
| PHerc0814/46527 rank 07 | Calibration | 87 | 2.8590% | 0.662 / 0.764 / 1.302 | 0.691 |

The actual adaptation schedule used six unique tiles per PHerc0139 context, rather than every grid tile. Their normalized-standard-deviation medians were 2.329 and 1.255, with spans 1.666–3.475 and 0.845–1.701. Each context contributed 32 scheduled exposures. These repeats and the full-grid summaries are not additional independent training support.

Support matters to descriptive statistics. Frag4 context 1 had 10.5813% zero voxels in the whole input, 0.1648% in the fixed 64-pixel interior and 0.9561% within its supplied exposed-surface mask. Its most zero-heavy tile was 47.2229% zero and had the same percentage tied at the clipped lower bound, while still retaining a positive MAD. Whole-input adjacent-depth correlation was 0.840; it was 0.566 in the fixed interior and 0.617 within supplied surface support. Context 2 had no zero voxels and its interior correlation was 0.537. The old interiors had adjacent-depth correlations 0.655–0.728. Consequently, a whole-context correlation comparison alone hides an important support difference. The full patches remain the actual normalization domain; the descriptive interior does not replace any evaluation mask.

There are documented representation differences. Frag4 uses 54 keV, nominal 3.24 µm uint16 surface-volume samples, direct 3×3×3 pooling divided by 27×257 and a float32 result at 9.72 µm. The older sources use 78 keV, nominal 2.399 µm published uint8 surface planes at XY level 2, followed by four-plane averaging and nearest-even uint8 rounding at 9.596 µm. Fractional values comprised 89.406% and 99.984% of the Frag4 inputs, and none of the older inputs. These facts distinguish the actual recipes; they do not provide a common attenuation calibration or a measured source-to-label registration.

The audit narrows the scalar-conversion and denominator-failure hypotheses. It leaves specimen and energy transfer, texture, pooling and quantization, optical/source alignment and annotation lineage unresolved. Marginal similarity cannot rule out a semantic domain shift, and marginal differences do not identify a cause. No offset, depth window, normal direction, input correction or threshold was fitted. Two Frag4 contexts are one specimen; the PHerc0139 and PHerc0814 context pairs likewise share their respective specimens. Adjacent stride-64 tiles overlap by 64 pixels in each axis, so there is no tile-independent significance claim.
