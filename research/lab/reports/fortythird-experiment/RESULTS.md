# Phase43: concrete reference leads and a public profilometry release

The bounded two-target search did not establish an independent CT ink-reference packet. It did find an accessible official profilometry release with paired image/label paths and an explicit spatial-calibration warning. This changes our earlier availability assessment and supplies a useful supporting CPU research route. No image body, CT chunk, label array, model weight or model output was read, and no training was performed.

## Findings that change the next decision

| Lead | Established from the inspected sources | Still required |
| --- | --- | --- |
| Paris4 directly visible ink | The paper describes expert-selected visible ink voxels, followed by DINO/intersection supervision and self-distillation. Public native CT and a volumetric ink prediction are listed. | Exact figure/seed coordinates and separately supported positives/non-ink/unknown labels. The dense prediction is not an independent reference. |
| PHerc500P2 front surface | The paper names a specific warped infrared photograph and a 65-layer orthographic composite. Public segment `20250628074500-500P2_front` has CT-linked meshes and a 118-plane surface volume. | The paper-product-to-release mapping, actual aligned optical reference/control points and label ancestry. Matching specimen names does not equate these products. |
| PHerc500P2 profilometry | Official release metadata lists paired brightfield images, normalized height maps, calibrated-height text and ink annotations, including four regions from this papyrus. | Actual-file/reference QC, valid/unknown support, physical-overlap checks and clarification of the reported calibration. No inspected native-CT mapping accompanies these local samples. |

## Paris4: physical motivation, not a ready test set

The paper's [volumetric ink methods](https://arxiv.org/html/2606.29085v1#Sx4) distinguish direct expert ink seeds from the dense model-assisted training signal. The latter uses similarity to DINO embeddings of selected ink voxels, intersection with a separately trained detector, iterative student guidance and self-distillation. Human involvement does not make the complete dense product independent of those models.

The current catalog lists the 2.4 µm reconstruction `20260411134726` and prediction `20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr`. The prediction's inspected attributes contain multiscale geometry, not the original expert seed set or a negative/unknown annotation contract. Eight transferred surface-label entries are also present; each says `annotated_on: unknown`. Their mapping coverage is not label-authenticity evidence. The exact paper-code commit's guided-DINO example names a developer-local checkpoint and generic input directory, not a released coordinate/reference packet. We did not run that code or retrieve weights.

The six phase42 centers remain unchanged and stopped. This search neither sampled a seventh center nor concluded that ink is absent from the volume. It establishes a specific missing artifact: a coordinate-linked visible-ink reference with supported negative papyrus and unknowns. The [unsent request](REFERENCE_REQUEST_DRAFT.md) asks for that small packet and cites the relevant figure/volume.

## PHerc500P2: useful source association, unresolved product mapping

[Extended Data Figure 5](https://arxiv.org/html/2606.29085v1#Sx12) identifies a front-surface CT composite at 2.215 µm and the Naples M1050 infrared photograph #017, manually warped using control points. This is stronger specimen-specific evidence than a generic claim that fragment labels come from infrared. It does not identify the current public mesh or every label version.

The public `500P2_front` segment has mesh transforms and surface volumes tied to the 2.215, 4.317 and 9.362µm reconstructions. The inspected2.215µm surface attributes specify 118 planes and a 15060×26440 canvas, whereas the paper describes a 65-layer orthographic composite. This difference need not be a defect: it means we have not established an exact rendering/optical-coordinate equivalence. The segment catalog and its complete top-level prefix listing contain geometry, surface volumes and detector outputs, with no ink-label entry. The one cataloged PHerc500P2 label set belongs to another segment, `20250825181859--1`, and declares transferred/unknown annotation provenance. It cannot be assigned to the front figure by name alone.

The original Hugging Face `ink/0500p2` folder **does exist**. Its contents were not inspected within this request budget. The initial recursive bucket request returned only the first 1000 files, all under `ink/0009b`; it is not a complete inventory. The subsequent nonrecursive listing established the target folder. Therefore this report does not claim that the required original optical/annotation files are unavailable publicly. The search stopped at its frozen 24-request cap, not at proof of global absence.

## Profilometry: availability correction and calibration caution

The official [profilometer dataset](https://huggingface.co/datasets/scrollprize/profilometer/tree/a806bead2f3b9100c20e19de37814fb285cfeefd) is accessible at pinned revision `a806bead2f3b9100c20e19de37814fb285cfeefd`. The card and manifest describe 14 regions containing 16 letters across PHerc248, PHerc250 and PHerc500P2, with 5, 5 and 4 regions respectively. The manifest names all paired files and the card states common dimensions and a top-left origin. Actual array compatibility and optical annotation agreement remain untested. The PHerc500P2 records have local sample coordinates, but no mapping to the CT front surface was found in the inspected manifest/configuration. These named regions are not the phase42 CT panel or an admitted substitute for it.

This updates phase41/42's limited observation that the paper promised a future release. The release is now evidenced by live metadata and a versioned file manifest; we did not download the image corpus or verify every listed file body.

The [paper](https://www.nature.com/articles/s41598-026-58467-1) reports 0.34µm native lateral sampling. The [pinned dataset card](https://huggingface.co/datasets/scrollprize/profilometer/blob/a806bead2f3b9100c20e19de37814fb285cfeefd/README.md) and `configs/sampling.json` report 0.68793625µm/pixel—about 2.023× the paper value. A September 3, 2026 notice explicitly acknowledges the discrepancy and says it is under investigation. We cannot choose the correct instrument calibration from these sources.

The 14 manifest dimensions/pixel counts and nine factor-derived scales check numerically. Reported physical extents approximately match `(pixels−1)×0.68793625`, with maximum residual 0.000175µm. The initial audit's overly strict 0.0001µm check failed for four of 28 extents; the corrected audit retains every residual rather than changing the source. This tiny rounding-scale issue is separate from the factor-of-two paper/release difference and does not resolve it. See [audit correction](AUDIT_CORRECTION.md) and [extent arithmetic](EXTENT_ARITHMETIC.json).

A scale-free native-versus-4× degradation experiment can still ask about label learnability after actual-file QC. It must not report a physical micrometre cutoff, use disputed units for height quantization, or infer that profilometry simulates CT. The paper already studied matched-quality and native-to-degraded behavior; a small replay/control is supporting methodology, not a novel result. A subsequent teacher comparison would need its own justification and reserved-parent evidence.

## Scope and verification

The search made exactly 24 successful GET requests, 429,661 retained response-body bytes, no network retries and no media/CT/model reads. It concluded within nine minutes of the first request. All request bodies and frozen pins were checked; three fetched code/documentation blobs match the paper's retained Git tree, two Hugging Face text blobs match its pinned tree, and both target catalog extracts match the compressed official catalog. Nine inspected S3 prefix pages were complete at their stated prefix; this does not establish complete recursive coverage. [SOURCE_VERIFICATION.json](SOURCE_VERIFICATION.json) records the scope. No GPU/R2/account action or external message occurred. GPU spend remains £0 and the first-pilot total cap remains £10.

## Recommendation

Keep paired-acquisition learning as the main direction. The immediate CPU task should be a bounded intake/QC of the pinned profilometry release, followed—only if paired files and reference support qualify—by a conventional native-versus-4× learning control with physical-parent separation. Use relative degradation factors while calibration is unresolved. In parallel, the [ready-to-review Paris4 request](REFERENCE_REQUEST_DRAFT.md) could obtain the native CT reference needed for the main study; it remains unsent. The [next-study draft](NEXT_STUDY_DRAFT.md) states what the supporting control can establish and what it cannot.
