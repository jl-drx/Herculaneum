# Reference and target comparison

The prepared pipeline closely reproduces the supplied PHerc1667 ink annotations. The PHerc0800 hotspot remains a diffuse response, and three geometry-selected contexts provide no supported letter reading. **No target transcription or translation is established.**

## What the reference establishes

A bounded public PHerc1667/w013 scan crop was paired with its released ink labels and supervision mask. The source-derived input contains 21 genuine depth layers at 9.596 µm sampling; the documented centered 17-layer window was used without padding or a fitted offset. Selection used annotation support before any detector output. Acquisition retained about 7.12 MiB of source payloads and metadata, with original chunks, URLs, hashes and preprocessing checks.

Both pinned checkpoints passed strict state matching. In the fixed central 128×128 analysis area, 2,861 pixels have supplied ink labels and 13,523 have supplied background labels. At the predeclared raw-score midpoint of 0.5:

| Checkpoint | Agreement precision | Agreement recall | Annotation-ranking ROC AUC |
| --- | ---: | ---: | ---: |
| Seed42 | 97.43% | 96.64% | 0.99944 |
| Seed43 | 97.05% | 98.95% | 0.99979 |

These are **descriptive agreement with current annotations on a training-overlapped region**. They establish that this implementation responds to the released labeled pattern. They do not establish accuracy on PHerc0800 or independently authenticate ink. The physical segment and representation are listed in both models' training configurations. The acquired label objects postdate the pinned model, so exact historical training-label byte identity is unproved. Intact-scroll labels include human strokes and iterative pseudo-labeling; the supervision mask defines supplied coverage rather than independent physical truth.

See the [source audit](reference-source-audit.md) and [comparison design](comparison-design.md) for source alignment, annotation semantics and limits.

## What happened on PHerc0800

Three disjoint 256×256 contexts were frozen using geometry and observed-data coverage, excluding the initial pilot and hotspot crops. Their selection code did not read predictions. This small field was already scored in the first experiment, so the contexts are not unseen or held-out validation regions.

All rows below use the central 128×128 area, restricted to valid/observed pixels evaluated by both models. The reference additionally requires supplied supervision. A shared score above 0.5 is a diagnostic response under smoothed training targets, not an ink label or calibrated confidence.

| Context | Analyzed pixels | Both seed scores >0.5 |
| --- | ---: | ---: |
| PHerc1667 supplied-label control | 16,384 | 2,790 (17.03%) |
| PHerc0800 original post hoc hotspot | 16,384 | 1,856 (11.33%) |
| PHerc0800 geometry context 1 | 16,384 | 0 |
| PHerc0800 geometry context 2 | 16,384 | 0 |
| PHerc0800 geometry context 3 | 12,796 | 830 (6.49%) |

The source and detector panels differ in structure: the reference response tracks a curved supplied stroke; the original target hotspot remains broad and diffuse. In context 3, a separately recorded geometry diagnostic places all 830 shared elevated pixels within 30 source pixels of invalid geometry. That association motivates checking boundary effects; it proves neither an artifact nor ink. Quiet regions likewise do not establish absence of ink.

The two models share training data. These contexts share one scroll segment and acquisition, and neighboring pixels are dependent. No statistical generalization or target accuracy follows from these counts. Native target spacing is 8.640 µm, 9.96% smaller than the pooled reference; no resampling was fitted to improve visual similarity.

## Review and reproducibility

- [Side-by-side source, annotation and raw-score comparison](comparison-reviewed/context-comparison.png)
- [Machine-readable masked measurements](comparison-reviewed/comparison-summary.json)
- [Frozen target selection, coverage and boundary analysis](independent-regions.md)
- [Predeclared comparison rules](PREDECLARED_COMPARISON.md)
- [Separate assistant visual review](joint-visual-review.md)

The reference runner preserves the original target runner and uses identical checkpoint construction, normalization, inference and blending functions. It accepts genuine 17–31-layer inputs and requires an explicit window. Every compared source, frozen provenance file, output and mask passed checksum verification. The target crops' central scores match the prior full-field scores bit for bit. These are pipeline checks, not accuracy validation.

An early reference run preceded final provenance additions. The comparison checksum guard rejected the changed sidecar; that run is preserved and excluded. The reference was rerun against a frozen finalized provenance snapshot, as recorded in [the run note](earlier-reference-run-note.json). Original source arrays remained unchanged.

## Scientific decision

Keep the hotspot unresolved and untranscribed. The labeled control makes a broken scoring implementation less likely, while the target's diffuse and boundary-associated responses remain insufficient for letters. The next useful acquisition is a larger, separately selected mesh-backed PHerc0800 surface region with more interior area and broader character context. Preserve the verified depth convention, predeclare the region from geometry, and use fixed depth and boundary diagnostics before selecting any purported text. Specialist reading comes after image-supported strokes.
