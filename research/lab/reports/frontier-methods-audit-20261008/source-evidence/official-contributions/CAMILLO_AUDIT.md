# Camillo depth-window contribution: source and reproducibility audit

Read-only source review, 8 October 2026. No training, inference, installation,
weights, scan payloads, prediction images or figure assets were acquired.

## Identity and official recognition

The official Vesuvius Challenge August 2026 Progress Prizes entry names
**Paulo Sergio Camillo / @pscamillo**, awards $1,000 for "How many depth
slices?", and links to `pscamillo/ink9-depth-window`.

- [Official pinned winners source](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/15_winners.md#L64)
- [Author repository pinned for this audit](https://github.com/pscamillo/ink9-depth-window/tree/999b912c1ce92b8f5ab63ef2cc90ea017dc6ad04)
- The fetched main-ref response identifies that commit and reports
  Last-Modified 20 August 2026. This is the repository-ref metadata, not a
  independently verified experiment execution date.

## What the published experiment reports

[README](https://github.com/pscamillo/ink9-depth-window/blob/999b912c1ce92b8f5ab63ef2cc90ea017dc6ad04/README.md)
reports matched training arms with depth 17, 13, 9 and 5 pooled slices, retaining
jitter of +/-2, plus a 5-slice arm with jitter zero. Reported blind judgements:
13/9 slices each 60 ties against the 17-slice baseline; 5 slices with +/-2
loses 58 to zero with two ties; 5 slices with zero jitter has 60 ties. The
retained `results/b05_score.json` and `results/b05j0_score.json` record those
counts. These are published author results; this audit did not reproduce them.

The useful hypothesis is coupling between depth-window width and depth jitter.
The result does not establish a universal optimal depth, statistical equivalence,
cross-scroll accuracy, or better target ink detection. The author's own README
reports worse balanced accuracy for the 5-slice/zero-jitter arm on PHerc1667/w029
(-0.133), despite visual ties.

## Input geometry, training and targets

- `scripts/prepare_arm.py` lines 64-68 explicitly generates training patches
  **[D,128,128]** and sets `flat_z_window_jitter.window_depth=D`; the optional
  `--max-offset` changes jitter separately. The described depth choices are
  D=17/13/9/5; these are retrained models, not an instruction to crop the old
  17-slice model's input without changing its architecture/checkpoint.
- The author reports 9.596 micrometre isotropic inputs from public **2.399
  micrometre** surface volumes, using the official 4x pooling recipe. The
  published harness consumes `<segment>_mean.zarr` without implementing the
  acquisition/pooling step or retaining its source hashes. The existing official
  dataset README describes XY pyramid level 2 and 4x Z mean pooling to 21 slices,
  with annotation only at Z=10. That establishes the advertised upstream
  representation, not the exact bytes used by the author's historical runs.
- B13 preregistration reports surface=10 and half-open canonical Z selections
  2:19 for D=17 and 4:17 for D=13; +/-2 jitter shifts the entire window. It states
  that both windows keep the annotated center slice. Exact historical target
  aggregation, normalization and loss implementation are delegated to
  `koine_machines` and are absent from the repository.
- README reports the `aligned21_hybrid_3d2d` recipe, batch 64, SGD, fp16,
  78,125 iterations, seed 42, and scroll quotas 29/22/11/2. It declares removal
  of the five native 9.362 micrometre representations, leaving the 24 aligned
  representations across four scrolls. It also reports `robust_mad`
  normalization and BCE label smoothing 0.5. The full baseline/config and
  historical package revision are missing, so these are author-reported settings,
  not fully audited resolved runtime parameters.
- All five depth arms reportedly used pre-18-August label-correction data. The
  later `m17L` arm reportedly used corrected labels. Exact label bytes/hashes and
  versioned acquisition manifests are absent. Ink-positive labels and supervision
  masks must remain distinct; zero labels outside supervision are not negatives.

## Evaluation selection, inference and leakage limits

- The retained `windows/b13.json` contains 60 windows, each **256x768** pooled
  pixels, across **24** segment aliases. All these segment aliases belong to the
  reported training corpus. B13 preregistration says surface coverage >=90%,
  **ink-label coverage >=3%**, and at most four windows per physical segment.
  This is annotation-stratified selection, not geometry-only selection.
- The README calls validation region-held-out, not segment-held-out. The
  published window manifest does not record training/validation exclusions, the
  sampling implementation, or physical-coordinate overlap. Full actual split
  independence therefore cannot be verified from this package.
- Three pairs of published windows overlap even in render XY:
  K11/K19 and K19/K46 on pherc0139-w028; K20/K34 on pherc1667-w023. Distinct
  aliases or nonoverlapping render coordinates would not prove physical
  independence either. Window counts are not 60 independent biological samples.
- `make_pairs.py` uses identical last-checkpoint steps, not validation-selected
  checkpoints. It invokes `koine_machines.inference.infer` with overlap 0.50,
  Hann blending, batch size 4, four workers and `--no-compile`. Its described last
  checkpoint is step 77,500 (save every 2,500), not the configured final 78,125.
- Display transformation is fixed `(uint8_score-64)/128`, clipped to [0,1]; it
  is a contrast display, not calibrated target confidence.
- One seed, one judge, reused windows and memory/expectation bias are explicitly
  declared. The second seed planned for a significant result was not run.
  The generic binomial vote test does not establish an equivalence margin or
  account for overlapping windows/segment clustering. A 17-slice/zero-jitter
  arm and a 5-slice/+/-1 arm were not run.

## Reproducibility gaps and actual harness defects

The untruncated pinned Git tree contains no `configs/`, `configs/m17.json`,
environment lock, upstream `koine_machines` revision, runtime log, or checkpoint
payloads. Although the README says checkpoints are here, no weight artifact or
author model-card URL is present in the inspected tree/README. A GitHub pin makes
these reviewed scripts immutable; it does not recover missing experimental inputs.

1. **Default window conversion is broken.** `prepare_arm.py` lines 119-123
   retains Portuguese keys `segmento`, `altura`, `largura` plus X/Y, then sets
   English fields from those keys. The shipped default `windows/b13.json`
   already uses English `segment`, `height`, `width`. They are discarded, then
   become null. Correct this before any harness reuse.
2. **The integrity check can pass with no predictions.** `check_integrity.py`
   lines 53-55 skip missing predictions; lines 68-81 only fail on identical
   existing pairs and return zero when no rows exist. Require complete pairs
   before treating it as a gate.
3. **B5J0's special grey-zone rule is absent from the generic scorer.** The
   preregistration requires a significant baseline win with fewer than 40 decided
   windows to be a grey zone. `score.py` classifies significant baseline wins
   generically. This does not change the published 60-tie B5J0 result, but must
   be reconciled before using the scorer for a new experiment.

No upstream scripts were executed or modified by this audit.

## Runtime, cost and licensing

- `scripts/README.md` reports roughly five hours on one **RTX 5070**, batch 64,
  for the example 9-slice arm. B13 preregistration dated 16 August estimates six
  hours on an RTX 5070. Main README reports **5.20 iterations/s versus 3.24**
  for five versus seventeen slices, and **4h10 versus 6h06** wall times.
  These are author-reported measurements/estimates; retained runtime logs,
  measured memory and monetary cost are unavailable.
- [LICENSE](https://github.com/pscamillo/ink9-depth-window/blob/999b912c1ce92b8f5ab63ef2cc90ea017dc6ad04/LICENSE)
  is MIT, copyright 2026 Paulo S. Camillo. The README says code, figures and
  results are MIT, while checkpoints remain subject to their upstream data/model
  terms. This does not relicense the scans or training annotations.

## Evidence and network ledger

All downloaded source bodies, exact requested/final URLs, response metadata,
SHA256 values and lengths are retained beside this file and in `sources.json`.
This agent's full contribution-source research used **303,543 response-body
bytes across 22 requests**, below its 3 MiB cap. The cap covers this agent's
official winners/community sources and both author READMEs as well as Camillo
code/metadata. No weights, scans or image assets were downloaded.
