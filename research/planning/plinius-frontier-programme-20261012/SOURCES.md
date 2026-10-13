# Current methods, releases and source audit

Primary sources checked 12 October 2026. This is a targeted research review,
not a systematic literature search or an independent reproduction of every
release. Model cards, repository reports and awards establish useful leads;
they do not establish our own performance result. Public URLs below identify
the sources; SOURCE_REQUESTS.json and source_snapshot.zip preserve the exact
retrieved bytes, timestamps and hashes. No weights, scan payloads or new
inference were fetched for this review.

## Sources and implications

**S01 — Angelotti et al., Complete virtual unwrapping and reading of a rolled
Herculaneum papyrus (27 June 2026, preprint).**
[Full paper v1](https://arxiv.org/html/2606.29085v1).
The abstract page still lists v1 as the latest version at access. Reviewed the
results, surface/ink methods, review criteria, data/code availability and training
tables. The paper combines recto prediction, explicit corrected meshes, low
distortion flattening, rendering, fragment training, iterative pseudo-labels and
papyrological review. The ink ablation uses 12,396 scheduled steps, effective
batch 8, LR up to 3e-4 and a reported H100 run, unlike our old 512-update,
batch-two, LR1e-6 continuation. The paper distinguishes general surface-conditioned
ink enhancement from directly visible volumetric deposits in Paris4. It uses no
OCR/lexical training targets. Code is pinned by the authors at
`e583fb67468f483fadd73d52e06f0ab0fe5ba813`.

**S02 — Official open problems and current Villa.**
[Open problems](https://scrollprize.org/2026_open_problems),
[Villa snapshot](https://github.com/ScrollPrize/villa/tree/47111d1a31381a5ff2c05046ed3e311b0dbeec64),
[VC3D guide](https://scrollprize.org/segmentation).
The open-problems page is dated July 10; the fetched Villa head is 12 October
2026 06:39 UTC. It explicitly identifies sheet topology, meshing, fibre tracing,
winding labels, training label quality and uncertain causes of ink failure.
Coarse and fine acquisitions differ in energy/propagation as well as resolution.
Current code, not the archived segmentation guide alone, must set any new
implementation contract. The full fetched tree was not truncated.

**S03 — Surface predictors.**
[m7 card](https://huggingface.co/scrollprize/surface_m7_nnunet/blob/cd83ccf7e1d58b70bd9074a9a7e158cd2c894f20/README.md),
[recto U-Net](https://huggingface.co/scrollprize/surface_recto_3dunet).
m7 is a single-fold residual nnU-Net component of the first-place solution,
with 192³ 3D plans; it is not the full winning ensemble. The paper's recto
predictor uses 256³ patches and Medial Surface Recall plus Dice/CE, trained from
approximate voxelized surfaces. The Kaggle writeup URL returned only a page shell
in this environment, so the full competition solution was not independently
audited. Do not replace it with an invented winning configuration.

**S04 — Ben Kyles and collaborators, Socratic Method C3-F0.**
[Pinned release](https://github.com/Hob3rMallow/socratic_method/tree/7f86cbf8d38155b37fdd39449c6775ec50344b6c),
[recipe](https://github.com/Hob3rMallow/socratic_method/blob/7f86cbf8d38155b37fdd39449c6775ec50344b6c/recipes/f0/recipe.json),
[24 September comparison](https://github.com/Hob3rMallow/socratic_method/blob/7f86cbf8d38155b37fdd39449c6775ec50344b6c/recipes/f0_benchmark_20260924/README.md).
Ordinary single-m7 architecture, CE + Dice + function-space M7 KL, wider
anti-aliased corpus; selected 200k-exposure checkpoint from a 250k-exposure run.
Released weights and a hash-bound recipe are advertised. Reported macro Dice
0.7130 versus published m7 0.5607 is on **fine-teacher targets**, with 442 real
native and 247 downsampled-fine rows from two scrolls. Recipe selection explicitly
uses that validation corpus, including checkpoint and operating-point choice.
It is not an untouched human-reference test. The reported human-labelled
PHerc0500P2 comparison is essentially level and was seen by m7 in training.
The no-TTA matched operating-point comparison also favours F0, so TTA alone
does not explain all reported gains. Small-gap/curve repair does not certify
global sheet identity. The recipe retires several earlier topology losses;
blindly adding such a loss would repeat known unproductive ablations.

**S05 — William Stevens, patch-based unwrapping.**
[Pipeline9 at reviewed revision](https://github.com/WillStevens/scrollreading/tree/62cbc21bdafbe0313fee11d781d729256a14262e/pipeline9).
Use its pipeline9 documentation, not the older root README alone. It generates
overlapping patches, checks incompatible relationships, removes bad patches/
bridges, assigns placements and flattens. The official August summary credits
approximately 365 cm² on PHerc1667; this is reported coverage, not our replay.
Whole-scroll processing is documented at 16–32 GB host RAM. Conflict filtering
and patch-graph assembly are established baselines, not new Plinius inventions.

**S06 — ScrollFiesta geometry pipeline.**
[Reviewed revision](https://github.com/Hob3rMallow/scrollfiesta_public/tree/bce8a65f97652316efb084171a17d2f1f8d3ae1a).
CPU `grid_pipeline -> scroll_whole -> scroll_unroll`; September `sheet_assemble`
adds flattened-sheet assembly and audit. It consumes an existing surface field;
Socratic is the separate field producer. Crucially, the current README says
the default 100-cube fixture completes its audit but is **not geometry-qualified**:
47.4% area coverage, 171 overlapping triangle pairs. Do not equate a successful
build/default run with the curated result. Its own code is MIT, while some
vendored components have different terms.

**S07 — Multi-sheet fitting, winding constraints and consistency.**
[Lasagna](https://github.com/ScrollPrize/villa/tree/47111d1a31381a5ff2c05046ed3e311b0dbeec64/lasagna),
[spiral fitter](https://github.com/ScrollPrize/villa/tree/47111d1a31381a5ff2c05046ed3e311b0dbeec64/spiral-fitting),
[windaudit](https://github.com/sergeievland/windaudit),
[Thomas Kelly's winding assignment](https://github.com/tf-kelly/scrolls),
[official winding guide](https://scrollprize.org/open_problems/winding_annotations).
Current Lasagna includes normal/density-based multi-sheet optimization, correction
points, distance-transform/flow gates and flattening. Current winding code has
cycle construction, minimum-edge repair and explicit coordinate transport.
windaudit already provides exact pre-fit integer-cycle certificates; Kelly
already assigns global winding numbers and reports where constraints fail.
Kelly's documentation warns that high fit satisfaction did not detect excessive
turn count, and that some independent fixture inputs are absent. A consistent
graph/spiral is an internal property, not proof of the right physical sheet.

**S08 — Henderson, Diffeomorphic Spiral Fitting (WACV 2026).**
[Official paper entry](https://openaccess.thecvf.com/content/WACV2026/html/Henderson_Virtually_Unrolling_the_Herculaneum_Papyri_by_Diffeomorphic_Spiral_Fitting_WACV_2026_paper.html).
Reviewed the official abstract/bibliographic entry and current released fitter
documentation, not the PDF's numerical tables. The method fits a global deformed
spiral to neural surface evidence, guaranteeing a continuous sheet representation.
That representational guarantee does not authenticate interpolation through
unknown tissue or eliminate the need to represent tears/missing components.

**S09 — Neural sheet/fibre tracing.**
[Released copy-displacement model](https://huggingface.co/scrollprize/copy_displacement_latest),
[current sheet tracer](https://github.com/ScrollPrize/villa/tree/47111d1a31381a5ff2c05046ed3e311b0dbeec64/vesuvius/src/vesuvius/neural_tracing/tracers/sheets),
[current fibre tracer](https://github.com/ScrollPrize/villa/tree/47111d1a31381a5ff2c05046ed3e311b0dbeec64/vesuvius/src/vesuvius/neural_tracing/tracers/fibers),
[automated-fibre viewer integration](https://github.com/ScrollPrize/villa/pull/1930).
The released copy model predicts neighbouring-winding displacement from a
conditioning mesh and was trained at 4.8 µm. Current sheet-tracer loss code
already includes on-sheet confidence, frame-specific position errors and a
dense curve loss to discourage between-point sheet jumps. Therefore neither
confidence-aware tracing nor neighbour contamination is an unaddressed concept.
The latest sheet-tracer OPERATIONS.md points to local/H100 checkpoint paths,
not a verified downloadable public release; we cannot call that exact model
ready to reproduce. PR1930 is a merged viewer/data-format integration, not by
itself an accuracy evaluation of automatic fibre extraction.

**S10 — Houjing Huang, PLatS (October release).**
[Pinned repository](https://github.com/huanghoujing/PLatS/tree/86050f0531dd99d33989ec7c1e0d1e7d028c8493).
Point-conditioned transformer predicts a single-sheet autoencoder code;
clustering supports automatic instances. Portable inference weights, example
arrays and 106 released-test-case predictions are advertised in release assets.
These are formerly hidden, now exposed test cases. Prompted/automatic modes and
ignore policies differ; compare like with like and equal annotation budgets.
Training configurations are archival, not a one-command portable retraining
recipe. We checked documentation, not the large release assets or inference.

**S11 — Domenico Russo, Reader v2.**
[Pinned repository](https://github.com/DomRusso2/reader-v2/tree/02c7b46987813c78f56dd99f38c5a84b65a9cd2a),
[model revision](https://huggingface.co/domenicor046/reader-v2/tree/72633d8aa6d40e9e737ccea4c9928fbc624dab45),
[actual evaluation/target code](https://github.com/DomRusso2/reader-v2/blob/02c7b46987813c78f56dd99f38c5a84b65a9cd2a/scripts/v2_run.py).
The strongest newly important native-coarse baseline in this review: released
hybrid architecture, 40k updates, batch 64, eight source quotas including the
old corpus, dense native labels from six objects. Author-reported 0009B
teacher-map AUC is 0.834 versus 0.725 for Hecate/dense-native, with useful human
endpoint gains; Hecate still leads on reported 0841 human labels (0.855 vs 0.824).
These are different endpoints, not a universal ranking.

Two concrete code findings narrow the claims we should borrow:

- `align_all` (lines 221–250) runs an ink model, correlates its prediction with
  a resized teacher map on one selected segment per object, tests cyclic shifts
  in 15-pixel steps, and accepts a peak within ±30 pixels at correlation >0.03.
  This is an operational prediction-agreement gate, **not independent CT-based
  registration or per-pixel same-sheet verification**. `build_labels` uses the
  resized reference. We should not adopt that gate as physical truth.
- `train` (lines482–499) uses 0841 human and 0009B teacher-map summaries for a
  10k early stop decision; `best_ckpt` (lines506–517) selects by those summaries
  for later workflow use. We do not assert that this function selected the
  released fixed 40k file. We do conclude that the score sets are used for
  development, even if excluded from gradients. They are not pristine final
  confirmation of our next method.

The release's number-checker verifies consistency with retained JSON summaries,
not inference correctness, annotation independence or registration. We inspected
it but did not run it or fetch the result maps. One training seed, teacher ancestry,
other segments of 0009B in training, and 0841 exposure during development remain
limitations. The value is a strong reproducible comparator and broader training
recipe; copying dense native distillation is not itself novel.

**S12 — Erwin Nieuwlaar, dense-native and PHerc343.**
[Dense-native revision](https://github.com/Nieuwlaar/ink9um-dense-native/tree/217e7521670db49cde97872ea6a8bdfb80c449f1),
[PHerc343 recipe](https://github.com/Nieuwlaar/pherc343-first-letters/tree/e8370f1f88a211166056df62ea6fcfffe4a1063e).
Both revisions remain current at retrieval. Our existing audits establish the
native-supervision/coverage confound, exact model contracts, and the bounded
PHerc343 reproduction. The reported 20k-step continuation/control pair is useful
budget precedent, not a current price quote. The full PHerc343 recipe has 496
maps, whereas our positive-control reproduction was a fixed eight-map image
recipe. Preserve that distinction.

**S13 — Hecate.**
[Pinned card/code](https://huggingface.co/scrollprize/hecate/tree/9cb86e500e944b11a06a7020403cde5dffb5bcb2).
Current revision unchanged. Fine 2D/3D teachers, learned depth attention,
degraded-fine then aligned native training, explicit neighbouring-sheet caveats.
9.6 µm must apply in XY and Z, with actual source spacing used in resampling.
Available inference is not a complete training/teacher-ancestry release. Its
volumetric output is ink, not a papyrus surface or guaranteed middle-sheet mask.

**S14 — Canonical and DINO-guided volumetric ink.**
[Canonical model](https://huggingface.co/scrollprize/ink_canonical_2um),
[3D DINO-guided model](https://huggingface.co/scrollprize/ink_3d_dino_guided).
Released models plus the paper describe conservative intersection of DINO
prototypes and volumetric teacher support, then distillation. DINO-guided
pseudo-supervision and 3D ink are established directions. Directly visible
Paris4 deposits are a valuable development reference, not universal independent
truth for all teacher outputs or coarse scans.

**S15 — Datasets and annotation refinement.**
[Curated datasets](https://scrollprize.org/data_datasets),
[surface labels](https://scrollprize.org/data/datasets/surface-labels-README.md),
[ink labels](https://scrollprize.org/data/datasets/ink-labels-README.md),
[fibre skeletons](https://dl.ash2txt.org/datasets/fiber-skeletons/README.txt),
[harmonized sheet instances](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels/README.txt),
[nnInteractive refinement](https://github.com/hammb/vesuvius-surface-detection-nninteractive-refinement).
Public geometry supervision exists; we do not need to invent it all. Harmonized
Paris4 instance IDs span cube boundaries, with removed small components documented.
Do not confuse its accompanying model-generated semantic masks with manual
instance labels. Some archived datasets use older acquisitions. The current
surface-label README still contains placeholder layout text; this review verifies
documentation, not the complete downloadable surface cohort. Ink labels include
manual plus iterative pseudo-label ancestry. nnInteractive refinement can reduce
placement errors while retaining topological errors; Dice against its own input
mask is not correctness. No new data payload was downloaded.

**S16 — Ink detection from surface topography (Scientific Reports, 18 June 2026).**
[Paper](https://www.nature.com/articles/s41598-026-58467-1).
High-resolution optical topography supports ink discrimination on that dataset;
no universal bulk relief/roughness cue is established, and cross-papyrus transfer
is heterogeneous. It motivates signal-scale questions but does not prove coarse
CT lacks recoverable information. Our completed supporting branch stays shelved.

**S17 — Dinovol and strong architectural baselines.**
[Current Dinovol](https://github.com/ScrollPrize/villa/tree/47111d1a31381a5ff2c05046ed3e311b0dbeec64/dinovol),
[nnU-Net Revisited, MICCAI2024](https://arxiv.org/abs/2404.09556).
Current code documents DINO/iBOT/KoLeo, Gram anchoring, higher-resolution adaptation
and sparse point supervision. Its general fine-tuning implementation is still
described as incomplete. Off-the-shelf features can be tested without promising
a complete training pipeline. The nnU-Net paper's abstract reinforces the need
for configured, adequately trained baselines and fair resource comparisons;
its numerical medical-data comparisons were not replayed here.

**S18 — Paulo Sergio Camillo, depth-window training.**
[Pinned study](https://github.com/pscamillo/ink9-depth-window/tree/999b912c1ce92b8f5ab63ef2cc90ea017dc6ad04).
17/13/9/5-plane retraining on a released-scale schedule; narrow depth and jitter
are coupled. Five slices with ±2 jitter failed; no-jitter five-slice maps looked
similar to the wide model under one repeated human rater, but some quantitative
endpoints worsened. One seed per arm and within-segment evaluation limit claims.
Reported throughput gain is about 1.6× on its GPU, not guaranteed here. Earlier
local audit found harness defects, so a reuse requires contract verification.
No change to our frozen full-depth study follows from this paper/report.

**S19 — Miika Kestilä, native supervision isolation.**
[Pinned correction](https://github.com/Mickey123123/vesuvius-native-supervision-isolation/tree/996f2a3ab1adefa715e3e3f3de1eded69620b6db).
Inspected the README and actual patch: excluded annotations could affect native
target/membership construction, with a reproducible counterfactual test and cache
changes. This establishes a relevant leakage failure mode and test design, not
that our running bespoke study or any released checkpoint is affected. Check
the exact adopted loader/version before future production training.

**S20 — Current milestones and prizes.**
[Milestones](https://scrollprize.org/winners), [rules](https://scrollprize.org/prizes).
The September list specifically identifies Reader v2, automated fibres, PLatS,
winding assignment/consistency and training-isolation work. The current prize
page is for 2027; prize fit remains secondary to research. Award summaries were
used to find primary repositories, not as proof of algorithmic accuracy.

**S21 — Alex Clark, inkbench.**
[Repository](https://github.com/Alex55288/inkbench).
Existing checkpoint soup/ensembles and failed teacher/student attempts are
important simple comparators. Its own limits include three repeatedly inspected
within-training-parent validation regions, teacher-map endpoints elsewhere,
unknown canonical-model exposure and two label chunks treated as empty after
zero-byte retrieval. We should not import its headline AP as our own benchmark
or interpret model agreement as reliable proof of ink/absence. Ensemble and
continuation controls are established, not new contributions.

**S22 — Topology-aware losses.**
[Skeleton Recall, ECCV2024](https://arxiv.org/abs/2404.03010),
[clDice, CVPR2021](https://arxiv.org/abs/2003.07311).
Abstracts reviewed as methodological prior art. These methods address thin
tubular/network topology; a papyrus-sheet loss must account for two-dimensional
manifolds, neighbouring sheets and real tears. The current recto model already
adapts Skeleton Recall to medial surfaces. Adding a generic connectivity term
is neither untested territory nor automatically a valid sheet-identity guarantee.

## Reproducibility assessment

| Method | Available and inspected here | Remaining before our own performance claim |
|---|---|---|
| PHerc343 fixed four-model image | Existing locally reproduced maps/receipts | Independent ink reference; full author sweep was not reproduced |
| Reader v2 | Public weights listing, config, loader/evaluation/label code and report | Download/hash/strict-load and our fixed-support inference; independent assessment |
| Dense-native / Hecate | Existing qualified contracts and locally used weights; current release rechecked | Full training ancestry; new external assessment |
| m7 / Socratic F0 | Public model/recipe identities and benchmark limitations | Actual selected ROI predictions and geometry comparison; separate confirmation |
| Stevens / ScrollFiesta | Current source/docs and documented caveats | Bounded native data bridge, geometry replay and CT-based correctness |
| PLatS | Portable inference release and input/prompt contract documented | Actual release asset verification and matched-prompt benchmark |
| Latest sheet/fibre tracer | Public source and losses | Exact accessible checkpoint/data contract; newest operations docs include private paths |
| Lasagna / spiral fitting | Current source/docs, explicit prior-art mechanisms | Target-specific inputs/constraints and independent geometry checking |

No method is called globally strongest based on incomparable public endpoints.
For each planned task, select the best *eligible* strong baseline on separate
development evidence, pin it, then compare on the reserved assessment. Document
when a comparator cannot be reproduced instead of silently using a weaker one.
