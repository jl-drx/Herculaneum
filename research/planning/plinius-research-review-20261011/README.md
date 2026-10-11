# Research decisions while the PHerc0800 depth run continues

11 October2026. **The useful next direction is a matched-physical-tissue
acquisition comparison, conditional on local CT correspondence.** A generic
fine-scan teacher or another depth adapter is not a sufficiently new hypothesis.
We completed three independent scientific reviews, a saved-output positive-control
assessment, new coordinate-metadata checks and a small physical CT experiment.
Study52's worker, inputs, protocol and review gate were unchanged; partial shifted
predictions were not inspected. No GPU or GitHub Actions job was launched.

## Findings that change the next decision

| Finding | Evidence | Consequence |
|---|---|---|
| A new teacher/student architecture is premature. | [Hecate already uses fine2D/3D targets, degraded inputs and aligned native scans](novelty/REVIEW.md). Averaging BCE over uncertain transformations is exactly BCE against averaged targets. | Generic paired distillation or a label blur cannot be presented as novel. A geometry-aware method must beat support-matched and smoothing controls. |
| Existing native training mixes acquisition and physical coverage. | [The exact source/alias audit](paired-data/REVIEW.md) finds six of Nieuwlaar's ten native-dense additions introduce additional physical windings relative to its aligned fine inputs. | Its continuation control estimates the value of that training mixture. A same-tissue, equal-exposure comparison still addresses an unresolved question. This qualifies scope, not the validity of the published result. |
| Published fine/native registration is not exact at its own landmarks. | The14 published fine78→native113 fit pairs have median3D residual2.0501 native voxels (19.193µm), maximum5.3585 (50.167µm). [Both scalar and NumPy replay](paired-data/LANDMARK_RESULTS.json) agree. | Measure local CT correspondence before pixelwise supervision. These fit residuals are not local ink-surface errors and authorize no recentering. |
| Reconstruction/export alone materially changes the native CT input. | [Nine streamed chunks,18MiB](reconstruction-control/RESULTS.md), compare three locations across three reconstructions of one scan. Sibling correlations are0.9857–0.9877; usual-versus-sibling correlations are0.7279–0.8316. Site2 has a nonzero phase-correlation peak. | Reconstruction mismatch is a concrete supporting factor. Neither sharper images nor high correlation select a better ink input. Exact local pairing remains unqualified at site2; no coordinate correction was fitted. |
| The current raw-map review can recognize strong positive-control structures. | [All18 PHerc343 boxes and matched-size contexts](depth-review/CALIBRATION_RESULTS.md) were inspected from the eight saved maps:11 broad-corroboration judgments,7 uncertain. All map hashes passed; no inference. | Keep study52's fixed gate. This cued qualitative assessment is not recall, ink accuracy or a test of fifteen-voxel depth sensitivity. |

The [32 fresh metadata reads](metadata/RESULTS.md) also close practical sampling
questions. PHerc0139's59-keV `-L1` render is2.258µm; fine78 level2 is9.596µm inXY
but still2.399µm inZ. Using a filename's nominal resolution or pooling onlyXY
would quietly produce the wrong model input. Surface metadata is available for
all four published benchmark regions, but metadata consistency is not tissue
registration.

## One next experiment, with a bounded entrance and exit

**Question:** at equal tissue coverage, targets and training exposure, does
learning from real native coarse CT improve native-scan reference agreement
compared with learning from deliberately degraded fine CT?

The immediate CPU work should be a **bounded correspondence qualification on
existing training windingw040**. It is already a training source, has both
acquisitions and supplied manual sparse native supervision, and avoids using the
benchmark reference to develop registration. Freeze geometry-selected areas and
all request budgets before reading their intensities; develop conventions on
separate structure from the landmarks used to check them. Local features must
identify the same sheet rather than merely maximize texture correlation. Keep
failure and unknown-support masks, and do not substitute a different crop because
its model response looks better. The original scan/model inputs and all48 views
in study52 remain its separate fixed experiment.

That qualification has a specific exit: establish usable same-sheet
correspondence on the frozen areas, or record why it cannot be established.
Do not turn an unsuccessful block into an unlimited search. If correspondence
is already sufficiently precise, proceed with the simple acquisition comparison.
If appreciable but bounded same-sheet uncertainty remains, evaluate its effect
on transferred targets before proposing uncertainty-aware learning. If competing
wraps cannot be distinguished, stop pixelwise transfer on that support; a mixture
of different sheets is not a valid substitute.

The subsequent learning comparison has two matched arms:

- **Baseline:** fine78 CT resampled/degraded to the selected coarse input
  convention, using the exact same physical training patches and teacher/manual
  target hierarchy as the treatment.
- **Treatment:** real native113 CT from those same patches, mapped into the same
  input convention. Architecture, initializer, mixed-scroll retention exposure,
  optimizer, update count, augmentation draws, physical support and endpoint are
  held fixed. No additional tissue or reduced PHerc0139 exposure is hidden in the
  treatment.

Use a released, credible training budget such as the published20,000-step
continuation as a starting hypothesis, with measured throughput/cost before any
paid launch. This is a prospective design, **not a GPU launch plan or claim of
statistical power**. One arm pair supports a pilot; it cannot establish
between-training-seed variability or a frontier-performance result.

Evaluate both arms on both observation types with paired per-region results.
Reproduce the published native reference-agreement benchmark unchanged, then
report any separately qualified-support view with a different name. Retain the
59-keV/model comparator where valid, author-designated blankw053, fixed
misregistration controls and mixed-scroll retention diagnostics. Their model
references are not independent ink truth; Hecate's exact region exposure remains
unresolved. Use physical segment identities and overlap groups, not misleading
training aliases, to prevent cross-acquisition leakage. Already exposed regions
stay development material.

**Stopping criterion:** one predeclared learning endpoint and one matched
comparison, with a practical effect/retention/blank-response criterion set on
training/development evidence before scoring the reserved assessment. No
checkpoint, teacher threshold, region or depth selection by the best resulting
score. This packet does not invent a numerical power threshold before the
qualified support and endpoint precision are known, or reuse ParisAP thresholds
for a different nativeAUC task.

**Result that changes the research decision:** a reproducible native advantage
that survives the alternate reference/control checks would justify focusing
Plinius on acquisition-matched learning before adding architecture. An apparent
advantage only against its own teacher or caused by changed support/coverage
would not. No useful difference at an informative matched budget would reduce
the priority of native exposure alone; it would not prove absent ink. Only a
subsequent advantage beyond matched smoothing/coverage controls would justify
crediting local correspondence uncertainty as a new mechanism.

## How to interpret the running depth study

The [independent decision review](depth-review/REVIEW.md) supplies a complete
post-completion table. A new nonzero-depth candidate meeting the existing
family/padding criterion goes to independent geometry/image review. A negative
result closes this three-position screen as inconclusive and makes PHerc0139
qualification the next priority. It cannot rule out ink, all other depth centres,
weak-signal detection or reconstruction mismatch. The saved PHerc343 calibration
supports strong-lead visibility, not universal sensitivity.

## Evidence and portability

All reports, source ledgers, exact new CT objects,36 PHerc343 calibration panels,
three reconstruction panel sets, code and checksums are preserved in the private
research packet. The calibration includes its stopped initial mask-encoding
attempt and correction. Original sealed studies and their cumulative seal
remain unchanged. See the repository's `BUNDLE.json` and `PORTABILITY.md` for the
selective GitLFS download and replay instructions.

The main reviewer replayed the landmark arithmetic byte-for-byte, checked fresh
metadata contracts, inspected all three CT panel sets and spot-reviewed saved
PHerc343 boxes4,13,17. The independent CT reviewer separately recomputed every
comparison, including phase-correlation via a full-complex FFT. Its [review](reconstruction-control/REVIEW.md)
records the code255-versus-clipping distinction and the local alignment limit.
These are scientific/engineering checks, not papyrological authentication.

GPU spend remains£0; the existing£10 total pilot cap remains intact. No spare-VM
assignment, external outreach, public release or repeated PR was needed.
