# Feasibility assessment

Assessment: 8 October 2026.

**A rigorous attempt to recover a small passage is feasible. A new reading or award cannot yet be predicted from our results.** Virtual reading of Herculaneum papyri has succeeded publicly, and our local setup now reproduces a released scan/annotation pattern. The uncertain part is finding authentic, legible ink on this target and validating a reading.

## Evidence available here

- Public PHerc0800 CT, surface volumes and associated coordinate meshes are accessible. Bounded reads preserve original source payloads and hashes.
- At the initial hotspot and two neighbors, the archived renderer settings reproduced all 93 tested CT-derived surface values exactly. This establishes local geometry and depth ordering, not ink.
- Both verified detectors closely reproduced current supplied PHerc1667 annotations on a training-overlapped region. This supports pipeline execution; it is not independent target accuracy.
- The initial PHerc0800 field and additional geometry-selected contexts supplied no coherent target letterforms. Model hotspots were diffuse or associated with invalid-geometry boundaries.

## Resources and practical scope

This environment has four CPU cores and no available GPU. Prior target inference measured roughly 0.16–0.20 seconds per scored 128×128 tile. A fully scored 1536×1536 field schedules 529 tiles per model, making the two-model run about three to four minutes including startup on this instance. Three fixed central-region depth controls add roughly two to three minutes. Source acquisition remains bounded to tens of MiB, with no need for a whole-volume download.

Those measurements support local inspection and candidate screening. A full-scroll search, new segmentation, broad model training or reconstruction across many surfaces would require a larger compute and geometry workflow. Translation becomes a tractable task once a defensible transcription exists, with specialist assessment of damaged Greek and alternative readings.

## Decision

Proceed with a larger geometry-selected context and fixed depth/boundary checks. Keep unsuccessful and uncertain results. If broader regions repeatedly yield only material texture, reassess geometry, resolution transfer and target selection rather than turning model outputs into words. Seek papyrological review when a reproducible stroke sequence appears.

The immediate deliverable is source-linked evidence suitable for review. We have no supported target transcription, translation or prize-qualification claim yet.

## Results and the June 2026 method audit

The actual two-model larger-context run took 95.17 and 84.28 seconds of inference, and four fixed seed42 controls took about 40–42 seconds each. The broader source and diagnostic closeups still provide no coherent target letterform. This supports the operational budget while leaving ink recovery unresolved.

The [primary-source method comparison](../methods-review/PHERC1667_METHODS.md) shows that the successful June team used approximately 2.4 µm data, a different 3D ResNet/2D U-Net detector, five target-adaptation rounds, substantial manual geometry correction and eight papyrologists. Its reported approximately 25 hours of manual annotation per wrap across 31 wraps illustrates the geometry effort; it is not a forecast for this target.

The current public PHerc0800 inventory lists only 8.640 µm data. All 22 matched First Letters eligible scan pairs in the bounded metadata screen are 8.640 or 9.362 µm. No finer eligible replacement is verified here. Our later official 9 µm baseline therefore fits the available resolution, but it does not reproduce the full June achievement. Enlarging a coarse image cannot add the missing observations.

The next proposed learning milestone is a bounded native-resolution PHerc1667 methods-control with the documented ablation/validation split, after exact artifacts and preprocessing are audited. Target training and papyrological reading require credible, independently reviewed signal first. Current results support a rigorous attempt, not a promise of a new passage or a full-scroll reading.
