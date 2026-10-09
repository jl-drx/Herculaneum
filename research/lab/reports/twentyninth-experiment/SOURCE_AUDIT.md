# Public fragment-depth benchmark: source audit

The useful contribution is the separation of ink-map agreement, localization relative to an estimated surface, and clear-papyrus versus ink response. These answer different questions. The last two are diagnostics relative to a gradient-derived surface, not independent ink-depth truth. We can reuse this distinction without adopting the full training/downloading pipeline or its published performance claims.

## Identity and scope

Repository: https://github.com/kartoun/vesuvius-fragment-ink-depth . The retained complete Git tree is191126e0e677b4cf17131a1f7bbaefb1fa8f92e1; this is a tree identity, not a commit identifier. Ten selected blobs were fetched by exact Git blobSHA1, checked for decoded length and Git identity, and separately SHA256-bound in acquisition-v1/SOURCE_RECEIPT.json. Ten HTTP200 responses returned75,837 body bytes;1,310,730 bytes reserved. No source changed upstream or locally. Files have .source suffixes to avoid implicit imports.

README, downloader and label-builder bytes from the sealed phase28 public audit were reused. UPSTREAM_LICENSE.txt preserves the actual MIT license, copyright 2026 Uri Kartoun. Code license does not establish data-license or prize-submission eligibility. No current prize claim or submission is being made. requirements.txt is an unpinned package list, not a reproducible CUDA environment; scikit-learn is absent from our current venv. No package installation was needed for this audit.

Selected safe AST definitions and assignments were executed against tiny synthetic fixtures only. run-v2/EXTRACTIONS.json records their original file identities and line spans. Full upstream modules, training, scoring CLIs, downloaders and models were not run. Static findings below are distinguished from targeted reproductions; none prove that an upstream published run encountered the edge case.

## Scoring and coverage

| Finding | Evidence | Consequence / treatment |
|---|---|---|
| Exported held-out predictions fill a full-sized volume with zeros and record rows metadata; Volume reads z_offset but not rows | export_predictions.py53–56; score_3d.py59–89; actual local syntheticZarr reproduction | Default full-row scoring can count unwritten rows as zero predictions. Explicit --rows can avoid this upstream. Require separate coverage, never infer it from zeros. |
| label_value=1 converts every other voxel, including ignore2, to zero | score_3d.py85–89; syntheticNPY reproduction | Carry unknown support separately. Binary labels with other background conventions require their own explicit encoding. |
| dtype-key normalization misses opposite-endian uint16 | score_3d.py80–82; opposite-endian NPY reproduction gave65535 instead of1 | Normalize by unsigned kind/width, accept both byte orders explicitly. This does not show that their actual predictions had that dtype. |
| 1024-bin ranking is exact for uint8 ordering/ties, approximate for fine floats | score_3d.py107–130;100 uint8 cases matched scalar oracles; constructed .5002/.5001 pair tied into one bin | Keep uint8 claim. Our float adapter computes exact rankings; example changes AUC/AP1 to.5 after upstream quantization. |
| AP with no positive examples becomes0 through nansum | score_3d.py122–130; synthetic reproduction | Mark binary discrimination undefined when either class is absent. Upstream AUC becomesNaN, creating different degenerate-bootstrap handling. |
| In-band mean accepts partial band coverage | score_3d.py175–180; six available layers of requested[-4,5] yielded a mean | Require every requested band layer and its validity. A complete narrow band can coexist with an incomplete wider profile; expose both masks. |
| Zero-mass columns are excluded from depth summaries; flat positive profiles get first-index argmax | score_3d.py170–174,212–239; static trace, local adapter cases | Expose zero-mass and peak-tie counts. A centroid of a flat profile is mathematically defined but not proof of localized ink. |
| Bootstrap tile space includes empty IDs and areas outside evaluation support | score_3d.py153,197–209;512×512 tile formula yielded5 IDs but4 occupied tiles | Do not inherit these CIs as parent-level uncertainty. Any future bootstrap must declare active blocks and empty-class replicate policy; spatial blocks still do not establish independent parents. |
| bestF0.5 optimizes on evaluation labels | score_3d.py133–140,207 | Descriptive best-on-this-reference only, not calibrated operating performance. Adapter supplies no threshold search. |
| Empty depth/ink cases are not a total reporting API | percentile on empty arrays; None surface_share formatted as float at plot/print | Future reporting must retain explicit undefined statuses and strict JSON. Adapter summaries use null/reasons; masked numerical arrays can containNaN. |

The scorer asserts fragment dimensions against the volume but does not authenticate every reference shape, source frame, coordinate units, depth sign, finite/range values, annotation support or original optical registration. A directory input stacks all TIFF layers in memory. Zarr row reads do not themselves guarantee bounded network/range behavior. Our adapter is only a bounded in-memory column-block primitive; it is not a qualified production dataset loader or a full external-memory exact-rank reducer. Its size guards cap elements, not process RSS.

## Labels and preprocessing

The retained step3 builder creates a new gradient-surface band policy from existing images/masks. It does not supply missing original annotation author/review history or optical warp/residuals. Unmarked fragment pixels are used as negative support; this must be called the **supplied-negative convention**, not exhaustive confirmed absence of ink. Clear columns remain negatives when unreliable ink columns are ignored. Require the same explicit convention or an independently reviewed negative mask in any comparison; no silent reclassification of unknown pixels.

step2 assumes exactly N_SAMPLE sampled ink columns when constructing is_ink, while step1.sample returns min(requested,available). Executing those exact assignments with requested 5, actual 2 ink and 5 clear mislabeled 3 clear columns as ink. With 2 ink + 2 clear, the negative-size allocation raised ValueError. This identifies a conditional small-sample defect, not its occurrence on their full Frag1 data. Our next panel code must construct class vectors from actual sample lengths, or avoid sampling labels at all.

step4.load_inputs estimates mean/std using the entire fragment, including the held-out rows, before training. predict_fragment explicitly estimates statistics on the target fragment. step5 uses each fragment's own0.5/99.5 percentiles to quantize32 layers to uint8, then applies (u8−127.5)/64. These are transductive input-statistics exposure, not direct held-out-label gradient leakage. They must be declared and held fixed. They are not seed42's robust MAD per-patch input contract and must not be substituted for it. Zero std/range and too-small sample sets lack explicit guards. Tiling/padding/overlap also differ across scripts; these can change results even with the same model.

## Splits, exposure and reproducibility

step4 chooses a band containing the middle 20% of annotated ink, using cumulative row counts. It excludes training patch centers within 96 rows around that band;64×64 training patches therefore have a real spatial buffer. Do not describe it as random pixel splitting or claim there is no buffer. Normalization still reads the held-out inputs. A complete physical-support audit would also bind inference context, rendering rays, pooling and aliases; we did not execute training or establish that independence here.

step5 excludes only the test fragment's exact name. Under our retained conservative grouping, Frag1/Frag2=Paris2, Frag3/Frag4=Paris1, Frag5=PHerc1667 and Frag6=PHerc51. Thus four of six ordinary LOFO folds retain a same-group sibling in training; six folds are not six independent parent tests. The local group-fold helper yields four groups, excluding all members of the test group. This is declarative metadata validation, not permission to train on reserved candidates or proof of disjoint voxel support. Equal fragment sampling also gives the two Paris groups more aggregate weight than singleton groups.

Existing seed42/Russo metadata already establishes PHerc1667 parent exposure relevant to Frag5; it does not prove use of those exact exposed pixels. Frag6 upstream history remains unknown, not affirmatively independent. Public aggregate results were read in phase28; no claim of blindness to those results is made.

step5 skips a fold/arm whenever its score JSON exists. That does not bind seed, steps, data, code or checkpoint to the current request. Our retained manifest/completion approach should remain the resume contract. Upstream requirements are unpinned; the scripts load whole stacks/all fragments and use CUDA. We did not run them, load weights or validate their published model artifacts. Reusing a full pipeline would require a separately pinned environment and complete input/output manifest.

## Reuse decision

Reuse the three diagnostic questions, explicit supplied-label baseline and parent-group comparison idea. Use the tested local exact-rank/support adapter for small panels. Do not import upstream full-stack I/O, implicit zero/unknown handling, partial-band means, CI interpretation or score-file-exists resume logic. Published results remain third-party reports; this audit neither confirms nor invalidates them.

The missing historical rendering/annotation/registration records are still absent from the inspected files, not proven unavailable publicly. A maintainer response can improve provenance but is not a universal prerequisite for explicitly limited supplied-reference research. The physical perturbation experiment still needs known input correspondence and physical depth; the scorer cannot supply these facts.
