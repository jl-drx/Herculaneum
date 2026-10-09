# Domenico Russo: dense 9 µm ink supervision audit

Read-only primary-source audit, 2026-10-08 UTC. No weights, label archive, scans, figures, or prediction assets were downloaded; no upstream scripts were executed. This report audits the author's released source and claims, not model quality independently measured by this project.

## Source identity and pins

- Official ScrollPrize August 2026 contribution entry credits **Domenico Russo**, $1,000, for evaluating all 14 released 9 µm checkpoints and dense teacher-label retraining: [villa winners, lines 57–58](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/15_winners.md#L57). That entry has no author repository link. The current author model card supplies the repository link and the repository README signs the same author name.
- Author code: [DomRusso2/ink9um-dense](https://github.com/DomRusso2/ink9um-dense/tree/dd5bc8fb300c810c6ec042cb7e068be78829d086), pinned main `dd5bc8fb300c810c6ec042cb7e068be78829d086`. Untruncated recursive tree retained in `russo-git-tree.json`. No gitlinks/submodules, trainer implementation, vendor directory, dependency manifest, or dependency lock appears in that tree.
- Author checkpoints: [domenicor046/ink9um-dense](https://huggingface.co/domenicor046/ink9um-dense/tree/3e5e7ed44e04606cf4a3766cf83a016543a30d83), pin `3e5e7ed44e04606cf4a3766cf83a016543a30d83`; primary card and API metadata already retained by the parent in `../training/*russo-current-*` and read locally here.
- Author pseudo-labels: [domenicor046/ink9um-dense-labels](https://huggingface.co/domenicor046/ink9um-dense-labels/tree/185a80ae4b9e75417bbb831797e6ab9e9e836fbc), pin `185a80ae4b9e75417bbb831797e6ab9e9e836fbc`. Despite being labels, this is a Hugging Face **model** repository, so metadata came from `/api/models/`, not `/api/datasets/`.

Primary source URLs, response lengths, SHA-256, retrieval times, ETag/Last-Modified where available, and local filenames are recorded in `sources.json`. Key retained hashes: GitHub README `5f162e9974b95dabb58b18f3d6f8e9fa422d9ddc8fc92b275fc39b0e8cf1e332`; recursive tree `f21f8fc7b51f4055072a5721d399661d18a6dd1e5195286840704ff4d21e706b`; pseudo-label card `ba43676d4b82b3823466a5a82d563e93b92c7a4c57684d7b32751e620fa6c330`.

## What the intervention actually changes

The three retained [training configs](https://github.com/DomRusso2/ink9um-dense/tree/dd5bc8fb300c810c6ec042cb7e068be78829d086/configs) were parsed and compared locally. Their only changed top-level fields are `out_dir` and `datasets`. Flattening the dataset groups by representation key confirms the same **29 representations** in all three arms, with identical input paths, physical keys, representation keys, and scroll assignments. All seven teacher segments change label roots in the all-7 arm; six change in the recommended `train_pseudo_ex016_config.json` arm.

**“w016 excluded” means excluded from dense pseudo-label replacement. It remains in training with its original sparse manual labels.** The config generator's `--exclude-pseudo` help explicitly says “segment(s) kept on REAL labels,” and [lines 80–106](https://github.com/DomRusso2/ink9um-dense/blob/dd5bc8fb300c810c6ec042cb7e068be78829d086/scripts/make_train_config.py#L80) only change the label root when a segment is not excluded. It does not create a wholly unseen w016 test segment.

All arms specify:

| Contract | Released source value |
|---|---|
| Model | `vesuvius_unet_3d_stem_2d`, flat output, one input channel, 16 stem channels |
| Input patch D × H × W | **17 × 128 × 128** |
| Z sampling | `flat_z_window_jitter`, 17-layer window, offset ±2, probability 1, padding forbidden |
| Aligned parent volume | 21 planes at nominal 9.6 µm (2.399 µm ×4 = 9.596 µm); aligned label plane z10 |
| Other representations | Five native 9.362 µm PHerc0139 representations; annotation plane z14 per inventory script |
| Normalization | `robust_mad`, configured percentiles 1 and 99 |
| Training | Seed 42; batch 64; 78,125 configured steps; SGD lr 0.01, weight decay0.00003; warmup 1,000; cosine scheduler; gradient clip 1; fp16 |
| Loss | `bce_label_smoothing=0.5`, `dice_label_smoothing=0.0`; description says smoothed BCE + Dice |
| Sampling | `fixed_scroll_prior_stratified`, fixed batch quotas 0139:29, 1667:22, Paris4:11, 0814:2 |
| Saving/online validation | Every 5,000 steps; best metric `val_balanced_accuracy`; `val_steps=32` |

The aligned 21-plane parent and the 17-plane network input are different quantities. Jitter moves the entire 17-plane window within the parent; it does not turn the model into a 21-plane network. The parent center z10 remains inside every permitted aligned window. Labels in the dense archive are written at z10 only, so depth collapse of the label stack yields that same 2D target. The builder cites the upstream flat-mode `amax` binarization, but the exact trainer version used for these runs is not released or pinned here.

The README reports sampling uniformly per physical segment within each scroll, independent of patch count. The config preserves the same representation identities and fixed quotas; the trainer/sampler implementation needed to independently verify observed sampling frequencies is absent. Dense supervision enlarges the pool of supervised patches, so unchanged sampling weights do not mean identical patch draws or an intervention that changes label density alone: label coverage, teacher targets, negative supervision, and patch availability all change together.

## Teacher, volume pooling, and label semantics

The [README method section](https://github.com/DomRusso2/ink9um-dense/blob/dd5bc8fb300c810c6ec042cb7e068be78829d086/README.md#L214) attributes the teacher to `scrollprize/ink_canonical_2um` on public 2.4 µm surface volumes. [teacher_seg.py](https://github.com/DomRusso2/ink9um-dense/blob/dd5bc8fb300c810c6ec042cb7e068be78829d086/scripts/teacher_seg.py) implements:

- 62 centered layers (`z0=(Z-62)//2`), tile 256, stride 128, batch 8, float16 CUDA;
- clipping CT intensity to 0–200 and dividing by 200;
- forward depth direction, with optional reversed pilot scored on manual supervision only;
- Gaussian tile blending, sigmoid output, and 4×4 XY mean pooling of the **2D teacher predictions** to the coarse label raster;
- threshold search over 0.02–0.90 in 0.01 increments on that segment's supervision pixels only, maximizing balanced accuracy.

The student CT volume pooling is a separate operation. [make_iso9.py, lines20–41](https://github.com/DomRusso2/ink9um-dense/blob/dd5bc8fb300c810c6ec042cb7e068be78829d086/scripts/make_iso9.py#L20) reads level0, takes 84 centered Z planes with `ceil((Z-84)/2)`, averages groups of 4 in Z and 4×4 in XY, rounds/clamps to uint8, and emits 21 planes with chunks 21×128×128. The README describes matching the official public level2 XY / Z-mean recipe; this code uses direct block averaging from level0 rather than reading the upstream pyramid's level2 array. Bitwise equality to the upstream prepared inputs was not independently verified in this audit. Teacher output itself has no Z axis to mean-pool.

Thresholds are hard-coded in [build_pseudo_labels.py, lines45–53](https://github.com/DomRusso2/ink9um-dense/blob/dd5bc8fb300c810c6ec042cb7e068be78829d086/scripts/build_pseudo_labels.py#L45): 0139-w016 0.15, w017 0.33, w028 0.27, w029 0.25; 0814-46527 0.53; 1667-w028 0.45, w029 0.36. The teacher is saved as uint8 probability×255; the builder thresholds those stored values against t×255. The saved maps and calibration outputs were not downloaded, so these values are source statements rather than independently recomputed calibration.

Actual [builder logic, lines127–157](https://github.com/DomRusso2/ink9um-dense/blob/dd5bc8fb300c810c6ec042cb7e068be78829d086/scripts/build_pseudo_labels.py#L127):

```
pseudo_ink = manual_ink where manual_sup, otherwise teacher >= t*
render_valid = student_CT[z10] > 0
pseudo_sup = render_valid AND NOT validation
```

Both manual positive and manual negative labels override the teacher. The new supervised negative class outside manual annotation comes from teacher predictions below threshold, not known absence of ink. The render-valid proxy is positive CT intensity on the central slice, not a separately authenticated paper or observed-data mask. Unsupervised pixels remain unknown even when `inklabels` contains a value there. The builder writes 0/255 uint8 at z10 and copies the old validation mask verbatim, preserving Zarr shape/chunks/dtype/fill/compressor. Its verification checks shape, broad density limits, zero overlap with that old validation mask, and empty first/last label planes. No asset-level checks were independently rerun.

The label card explicitly says: **“These are model-derived labels, not human annotation.”** It reports 288,718,152 supervised pixels over seven aligned segments versus 5,774,947 original manual-supervision pixels. It warns that w016's permissive 0.15 threshold labels 28% of the canvas as ink and that all-7 training merges held-out letters into blobs.

## Evaluation, mask revision, and leakage limits

[step9_eval.py](https://github.com/DomRusso2/ink9um-dense/blob/dd5bc8fb300c810c6ec042cb7e068be78829d086/scripts/step9_eval.py) runs full-canvas inference on three fixed aligned regions: 0139-w016, 0814-46527, 1667-w029; overlap 0.5, Hann blend, batch 8, `--no-compile`. [score_ink9um.py](https://github.com/DomRusso2/ink9um-dense/blob/dd5bc8fb300c810c6ec042cb7e068be78829d086/scripts/score_ink9um.py) thresholds at 0.5 and masks pixels; it converts uint8 output back to probabilities and then logits for the team's metric, alongside a direct threshold cross-check. `step9_eval.py` AUC averages tied ranks correctly. The records are author-produced scores against annotation masks; this audit does not authenticate ink or transcriptions.

The source config trains 78,125 steps, but the released/evaluated “final” weights and result paths are **step75,000**, the last regular 5,000-step save. Do not describe those weights as the78,125-step final model. Retained raw online logs end at 75,000 and contain 15 validation records per arm.

The README table uses **2026-08-14 labels**, and its +0.077/+0.100/+0.026 improvements describe the **all-7** arm. Recommended ex016 gains over the matched control are +0.0480/+0.0790/+0.0068. Its w029 AUC is lower than the control despite the small balanced-accuracy gain.

The author retained a corrected 2026-08-18 rescore after the official masks changed. The following values were parsed from `results/step9_results_v20260818.jsonl`; they were not independently measured:

| Step75,000 arm | w016 BA / AUC |0814 BA / AUC | w029 BA / AUC |
|---|---|---|---|
| Manual control |0.701862 /0.872512 |0.753879 /0.832307 |0.781396 /0.894519 |
| Dense all-7 |0.778563 /0.894542 |0.853878 /0.934955 |0.806965 /0.904431 |
| Dense w016-excluded |0.749900 /0.908212 |0.832891 /0.929764 |0.788178 /0.885284 |

Material limits:

1. **Pixel-mask holdout is not whole-segment holdout.** The three scored regions are within segments retained in all student training arms. The teacher's complete original training inventory is not provided here, so student validation exclusion does not prove teacher independence from those regions.
2. **The old zero-overlap guarantee does not hold for all-7 against the corrected w016 mask.** `results/label_version_diff.json` reports 2,924 new w016 validation pixels overlapping its old pseudo supervision, out of 178,146 corrected validation pixels (~1.64%). It reports zero such overlap on the other two regions. The README says the recommended ex016/control/released models are unaffected by this particular correction because their w016 training used original manual supervision. This does not remove their other selection/holdout limitations. The pseudo-label archive was last modified August17 and should not be presented as excluding the subsequently corrected mask without rebuilding/checking.
3. **Alias duplication is real; the reported reserved-case leak is the author's claim.** Both aligned 0139-w028 and native w044 use the public physical segment `20260115000000-w044_2026011522`, as confirmed by the already retained official dataset README. All author configs train both and assign different physical keys `0139:w028` and `0139:w044`. The author's README says the released reserved-validation split leaks through this alias. That is not, by itself, proof of contamination in the three scored masks above; w028 is not one of those three metric regions. Whole-physical-segment splitting must resolve the alias before future experiments.
4. **The evaluation regions were reused for model/method choice.** The author acknowledges that online-best checkpoints and the quoted ensemble were selected on these same three regions. The recommended ex016 arm was also selected after seeing all-7's w016 output and metric disagreement. Step75,000 comparisons avoid online checkpoint selection but do not constitute a fresh blind test of the already selected method.
5. **Online scores use different effective sample populations.** Raw logs show 17 validation batches for control, 22 for all-7, 20 for ex016; pixel confusion counts differ. The full-canvas fixed-mask scorer supports the final comparison more directly than these online scores. The absent trainer prevents auditing why the online populations differ.
6. **One training seed; three regions.** The author's reported released-model same-step seed spread is 0.061–0.130 on w016, 0.013–0.021 on 0814, 0.066–0.072 on w029. The README limits robust improvement evidence to 0814. Do not equate BA gain, AUC gain, legible writing, or authentic ink: w016's ranking differs between BA and AUC, and recommended w029 AUC declines.

The scorer's resume logic reuses predictions by tag/segment path without verifying checkpoint/input hashes. A new comparison should bind cache entries to content hashes and record exact mask versions rather than reuse these paths blindly. The scorer reports an `agree` boolean for its metric cross-check but does not abort when false.

## Availability, compatibility, hardware, and licences

The author advertises unchanged loading via:

```
python -m koine_machines.inference.infer <segment_9um_iso.zarr> \
  dense9um-w016excluded-step075000.pth out.tif \
  --overlap 0.5 --blend-mode hann --batch-size 8
```

and training via `python -m koine_machines.training.train configs/train_pseudo_ex016_config.json` after unpacking the archive and editing paths. These are author commands, not commands run during this audit. The cards say checkpoints preserve `model/config/step` format. Actual loading compatibility with the project's pinned current inference stack still needs a separate controlled check; no checkpoint bytes were fetched here. The dataset/config layout targets approximately isotropic 9 µm surface inputs with the trained preprocessing/normalization and 17-depth contract. It does not authorize feeding an arbitrary native 31-layer stack or a different geometry/raster directly.

Reproduction is incomplete from this repository alone:

- `score_ink9um.py` imports `ink9um_env`, and `fetch_labels.py` imports `fetch_parallel`; neither module is in the pinned untruncated tree.
- Teacher code imports `model_resnet3d_3d_decoder` from a separate local `vendor/inkdet` and loads an unpinned local canonical checkpoint. The teacher implementation/revision is absent.
- The training config generator reads two external `vendor/villa-ink` configs; exact upstream revision used is not recorded. The training implementation and its package environment are absent.
- Configs, teacher paths, interpreter paths, and PYTHONPATH conventions contain the author's Windows filesystem assumptions. Moving to this Linux cloud environment requires explicit path/environment adaptation.
- No requirements/pyproject/lock, container, training runtime log, or measured training-cost invoice was found. The released online-validation logs have scores and step numbers, no timestamps.

**Author-reported hardware/runtime**, not independently benchmarked: README says one RTX 5070 Ti 16 GB, 8.6 GB VRAM at batch 64, torch 2.11.0+cu128, zarr 3.2.1, Windows 11; 9.5 hours per run at 2.3 iterations/s. It gives no monetary compute cost. This is a training claim, not full preparation/teacher-generation/inference time or a quote for cloud GPUs.

Expected downloadable objects from metadata only:

| Object at pinned HF revision | Bytes | Expected SHA-256 (HF LFS) |
|---|---:|---|
| `control-manuallabels-step075000.pth` |138,366,975 |`e60cee6b2c08a021ddfa9683869dab91787531cef762d833cc1f21fdee7fa114` |
| `dense9um-all7-step075000.pth` |138,365,469 |`4e23f3cd6e5742a2911431145e825fe4523f8abdcb8790615e209dfc20be7977` |
| `dense9um-w016excluded-step060000-best.pth` |138,369,043 |`a92875559e654a35a0702e3563a95914131692ae8b832a0f86ac8e239947897d` |
| `dense9um-w016excluded-step075000.pth` |138,367,629 |`1d7b99bc641b459736a762b2a1d8c357f61ada4f0117c24c7327b6a1293ba1eb` |
| `ink9um-dense-pseudolabels.tar` |70,615,040 |`093cdb7dd03d881ba280be2b9b9fcb6ecf4f36594399a5a99e89538ad836736f` |

The checksums are expected metadata values, not hashes of locally downloaded binary objects. Four weights total 553,469,116 bytes; all four plus the label archive total 624,084,156 bytes before any input data.

[Repository LICENSE](https://github.com/DomRusso2/ink9um-dense/blob/dd5bc8fb300c810c6ec042cb7e068be78829d086/LICENSE): MIT, copyright 2026 Domenico Russo; preserve copyright/permission notices in copies or substantial portions. HF frontmatter declares MIT, but the detailed cards and GitHub README explicitly qualify derived weights/labels: **“remain subject to the terms of those sources”** (`scrollprize/ink_9um`, `scrollprize/ink_canonical_2um`, open-data S3). It is incorrect to turn the MIT code/card declaration into unrestricted relicensing of scans, upstream labels, or derived artifacts. These sources do not resolve each upstream data licence or award/publication permission; the parent terms/rules audit remains applicable. Private research reuse and award eligibility are separate questions.

## Practical use for this project

This is a credible **candidate ink-training method and comparison release**, with explicit controls and useful disclosed failures. A bounded experiment can compare the pinned recommended step75,000 model, its matched manual control, and a named released baseline on identical frozen CT patches/raster/depth/preprocessing, with original outputs retained and fixed display scaling. The student config contains no PHerc0800 representations, but absence from this student config is not proof of complete upstream teacher/model independence or target success.

An ink-map comparison on our detector-selected PHerc0800 regions would remain a transfer diagnostic. It cannot validate manuscript reading, mesh placement, or ink without independent evidence. A new training claim needs physical-segment splits with aliases resolved, frozen mask versions, a control that matches the proposed intervention, multiple seeds or an explicit uncertainty limit, and new evaluation material held out before tuning. Density, teacher choice, smoothing, and render/preprocessing changes need separate ablations if causal attribution is desired. Keep translation/speculative reading out of model-selection labels.

## Audit accounting

This Russo subtask added **267,723 response-body bytes in 25 requests**, including main-ref discovery, untruncated tree, docs/config/scripts/results, and pseudo-label metadata/card. Cumulative own frontier audit: **571,266 bytes in 47 requests**, below the 3 MiB cap. The parent's current checkpoint card/API files and pre-existing official dataset README were read locally and add no network bytes here. No figure, weight, tar archive, scan, or prediction was downloaded; no install, infrastructure change, publication, feedback, or outreach occurred.
