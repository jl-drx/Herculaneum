# PHerc0139 paired-acquisition qualification

Research review, 11 October 2026. This work inspected retained primary code/catalogs and fetched five small pinned source files, totalling **18,621 bytes**. It ran no model, fetched no CT chunks or prediction images, and did not inspect or alter study52. The exact sources and hashes are in `SOURCE_PINS.json` and `sources/NEW_SOURCE_LEDGER.json`; the machine-readable inventory is `ASSET_ADMISSION_MATRIX.json`.

**Recommendation:** admit PHerc0139 to a bounded, CT-only correspondence qualification. Do not yet admit its fine/coarse pixels as registered training pairs. The pairing mechanism and a useful public reference-agreement benchmark exist. A new causal comparison should keep physical tissue, labels and exposure matched across observation qualities; adding ordinary fine-to-coarse pseudo-labels alone would repeat substantial existing work.

## What is available

| Released observation | Volume / scan | Sampling and acquisition | Qualification |
|---|---|---|---|
| Fine 78 keV | `20260102150214` / `20251205005141` | 2.399 µm, 220 mm propagation distance, masked uint8 | Public surface volumes and canonical-model maps; main teaching/reference family |
| Real coarse 113 keV | `20250728140407` / `20250720065842` | 9.362 µm, 1200 mm propagation distance, masked uint8 | Public native surface volumes, shape depth 28 |
| Finer 59 keV | `20260413113053` / `20251202104544` | Original volume 1.129 µm, 220 mm | Public `-L1` surface renders are effectively 2.258 µm, as root's fresh metadata check confirms; availability is partial |

The coarse source itself was exported from a `4.681um...binmean_2` reconstruction. “Native” here means the released 9.362 µm acquisition volume, not unbinned detector sampling. The acquisitions also differ in energy, propagation distance, reconstruction/windowing, masking and time. A native-versus-degraded experiment measures **acquisition mismatch**, not resolution alone. Pooling 4 × 2.399 gives 9.596 µm, not9.362; the pooled fine representation and native representation require explicit physical sampling and coordinate handling.

The fine volume is now catalogued with a newer remasked successor. Reproduction must pin the older published volume/teacher combination; changing to the successor would be a separate data-version intervention. Arrays with identical shapes or identity volume transforms are not automatically byte-equivalent.

The retained catalog provides a shared source surface, transformed meshes, published affine matrices, landmark correspondences, volume-coverage boxes and rendering provenance. These make qualification practical. They do not establish correct local tissue correspondence or middle-sheet selection everywhere.

## A measurable registration limitation, already in the public metadata

We replayed the saved affine matrices on the landmark pairs stored with them, without fitting a new transform. `analyse_landmarks.py` performs both NumPy and scalar calculations; their distance differences are below 1e−10 target voxels. `LANDMARK_INPUT.json` preserves the small catalog extract and original catalog hash; `LANDMARK_RESULTS.json` preserves every residual vector.

| Published transform | Correspondences | Median residual | Maximum residual |
|---|---:|---:|---:|
| Fine78→native113 | 14 | 2.05012 native voxels =19.193 µm | 5.35855 native voxels =50.167 µm |
| Fine59→fine78 | 12 | 0.85790 fine78 voxels =2.058 µm | 1.95469 fine78 voxels =4.689 µm |

These are **residuals on supplied fit correspondences**, not independent target-registration errors. They are Euclidean 3D vectors, not errors along a papyrus normal. No evidence here says a given benchmark crop is displaced by the median, and no recentering follows. They do establish that the published mapping does not reproduce all its own correspondences to subvoxel precision. A pixelwise fine/native learning comparison should therefore measure local residuals from CT structure, then either exclude uncertain support or carry that uncertainty into supervision. The published resizer is insufficient proof.

Reproduce the arithmetic with:

```bash
python analyse_landmarks.py --input LANDMARK_INPUT.json --output LANDMARK_RESULTS.json
```

Only NumPy is required; the script neither fetches data nor fits registration.

## Which regions have defensible roles

Nieuwlaar's pinned benchmark and config explicitly exclude the four below from its native-dense training. The source README also calls title/w024 outside the `ink_9um` label sets. That claim does not establish Hecate exclusion, cross-model teacher independence, absence of physically overlapping alternate surfaces, or a pristine test for our project.

| Public surface | Proposed role | Limitation |
|---|---|---|
| `20260422000000-title_2026042222_zmid_flatboi` | Reproduce frozen reference agreement after coordinate qualification | Published benchmark repeatedly compared/checkpoint-selected; no independent fine59 map in the retained asset summary |
| `20260215000000-w024_2026021572` | Reference agreement plus a second scan/model comparator | Fine59 volume bounding-box overlap proxy is 0.3665, not full coverage or a pixel validity mask |
| `20260206000001-w047_2026020613` | Freeze operating threshold/component recall | Calibration region is not a final test |
| `20260220000000-w053_2026022021` | Author-designated blank response control | Blank designation is not independent chemical or papyrological proof |

Do not train on these or use their model predictions to fit a registration. Preliminary CT geometry checks need not consume labels, but choosing an accepted crop after observing prediction quality would make it a development crop.

The ten documented native-dense **training candidates** are:

| Winding | Exact public segment |
|---|---|
| w030 | `20250108000005-w030_2025010818` |
| w032 | `20260203000000-w032_2026020303` |
| w033 | `20260219000000-w033_2026021913` |
| w034 | `20260303000000-w034_2026030317` |
| w040 | `20250831000000-w040_2025083102` |
| w041 | `20260108000000-w041_2026010816` |
| w043 | `20260112000000-w043_2026011217` |
| w046 | `20260325000000-w046_20260325` |
| w049 | `20260318000000-w049_20260318` |
| w050 | `20260302000001-w050_2026030220` |

All ten have fine78 and native113 array paths in the retained catalog. This review has not fetched their arrays or established local registration. w040/w041 also have manual sparse native labels merged into the teacher supervision. The five official sparse-native training windings are w035,w039,w040,w041,w044. Their overlap with the ten means 15 representations cover 13 distinct native windings.

**Alias trap:** training `pherc0139-w016` maps to public w029; `w017` to public w030; `w028` to public w044; `w029` to public w045. A split keyed only by the training alias would be wrong. The catalog also contains two different long IDs called w030; alternate surfaces of the same winding must stay in the same physical split group until their physical separation is established. Exact public IDs plus physical winding/overlap grouping are required.

For an initial CT-only registration development region, an existing training winding such as w040 is preferable to adapting a transform on title/w024. A frozen area inside that training winding can establish conventions; subsequent landmarks in separate physical support must test registration rather than fit it. A claim of future generalization still needs a prospectively frozen physical split and checkpoint/teacher exposure inventory.

## What supervision can support

Nieuwlaar's `native_bench.py` reads the released 78 keV model map and applies `cv2.INTER_AREA` to the native array's H,W. It derives high/low confidence classes inside native CT support where teacher>0. The ignored band is appropriate uncertainty handling, but teacher-derived “no ink” is not an independently authenticated absence of ink. The training builder uses threshold 47/255 by default (calibrated against five manual native sets), with manual labels taking precedence on w040/w041. Its version note identifies the 19 August native-label transfer rerun.

`native_common.py` claims teacher-to-manual IoU 0.96–0.99 with zero shift, but does not supply corresponding local CT landmarks/residuals. The separate build report records thresholded teacher/manual pixel agreement 0.7899 and0.818 on w040/w041. These are different claims/metrics and must not be conflated into a geometric alignment certificate.

The second 59 keV model map is useful corroboration across scan and teacher, with independent valid support and its own alignment uncertainty. It remains model output. The released benchmark's `student_over_ceiling` threshold 1.02 is a heuristic warning, not a theorem or calibrated bound on teacher cloning. Positive teacher agreement alone cannot prove a new detector sees physical ink.

Shift/rotation controls are worthwhile but insufficient: the author reports treatment misregistration AUC 0.70–0.77, so substantial nonlocal/texture agreement remains. Include the signed margin over these controls, reference disagreement, blank-response behaviour and spatially separated regional results alongside main agreement. Avoid pixel-count confidence intervals that treat millions of correlated voxels as independent samples.

## The prior art leaves a narrower, useful causal question

Hecate already distils both 2D and 3D fine-model outputs into its 9.6 µm model. Its model card says it introduced aligned real coarse patches after initially downsampling fine CT, used blur/contrast augmentation, and retained fine-derived inputs where native alignment failed. The card/runtime do not provide a segment-level training-exposure inventory or numerical local-registration thresholds. Generic paired-acquisition distillation therefore is not a new Plinius contribution, and Hecate must not be called independent of all candidate labels or regions.

Nieuwlaar's matched 20k-step continuation establishes the benefit of **adding his native-dense mixture** over continued training. However, resolving the physical aliases shows that only w030,w040,w041,w043 of its ten additions already appear in the aligned PHerc0139 training inputs. Six additions—w032,w033,w034,w046,w049,w050—also add physical tissue coverage. This control does not isolate acquisition quality from added tissue/supervision coverage. That is a legitimate opening for a matched-physical-region comparison, not a failure of the published experiment.

After correspondence qualification, the discriminating study is to hold physical regions, supervisory targets, initialization, exposure and test acquisition fixed, and compare real native versus fine-derived coarse observations. Only then test uncertainty-aware teacher transfer against a capacity/exposure-matched native learner. Any gain must survive region-level assessment and a reference not identical to its training teacher. This avoids spending a training pilot on an effect already explained by coverage or teacher-family matching.

## Admission decision

**Proceed now with metadata/coordinate qualification; proceed next with a bounded CT-only local-correspondence test once its coordinates and byte budget are frozen.** Retain raw depth information, physical units and the original source/teacher versions. Do not choose crops by ink response. Stop qualification where alignment is ambiguous or overlapping sheets cannot be distinguished; do not silently replace failures with synthetic fine-derived inputs in a comparison intended to test real native CT.

The unresolved blockers are local fine/native CT correspondence, exact valid support/depth/orientation conventions, physical split overlap, and teacher/model exposure. They block a causal paired training or pixel-accuracy claim. They do **not** block replaying the public benchmark as explicitly labelled reference agreement, or measuring the saved affine residuals above.

Data in the retained volume catalog are CC BY-NC 4.0; source code/model cards use their separately declared licences. A research export should preserve both provenance and attribution. This review does not establish prize eligibility or amend the frozen study52 protocol.

Primary code links: [Nieuwlaar pinned repository](https://github.com/Nieuwlaar/ink9um-dense-native/tree/217e7521670db49cde97872ea6a8bdfb80c449f1), [native benchmark](https://github.com/Nieuwlaar/ink9um-dense-native/blob/217e7521670db49cde97872ea6a8bdfb80c449f1/native-eval/native_bench.py), [training config](https://github.com/Nieuwlaar/ink9um-dense-native/blob/217e7521670db49cde97872ea6a8bdfb80c449f1/training/configs/train_dense_native.json), [Hecate pinned model card](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/README.md).
