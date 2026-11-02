# Engineering roadmap

[Overview](../README.md) · [Project tour](project-tour.md) · [Quickstart](quickstart.md)

Herculaneum has working research infrastructure and unresolved scientific questions. The next useful software work should make experiments easier to reproduce and evaluate while preserving the original evidence.

The tasks below are proposed, bounded engineering projects. They are suitable for implementation, review and coding-assistant comparison; they are not completed features or authorization to run new scientific experiments.

## Portable experiment execution

**Problem:** many historical workers retain absolute lab paths and assume separately stored inputs.

**Next task:** adapt one selected historical worker to a configuration-driven entry point, with explicit input, output and cache paths.

**Acceptance:** reproduce its retained outputs from the same inputs in a fresh directory; record source/data identities; reject an existing output directory; document any numerical differences. Preserve the original worker and sealed evidence.

Start with the [research portability notes](https://github.com/jl-drx/Herculaneum/blob/9f6fe45ab20eb0961d5e06aba5c3038088941180/research/README.md), the working [experiment-39 replay](https://github.com/jl-drx/Herculaneum/blob/9f6fe45ab20eb0961d5e06aba5c3038088941180/research/tools/replay_phase39.py) and the [artifact contract](artifacts.md).

## Faster bounded acquisition

**Problem:** the current cache deliberately serializes downloads for correctness.

**Next task:** design bounded concurrent acquisition while retaining shared-input pins, hash checks, disk limits and explicit missing-object semantics.

**Acceptance:** test interruption, competing owners, low disk space and corruption under concurrency; verify that no partial object is admitted as complete; measure throughput and peak storage against the existing implementation.

Start with [chunk_cache.py](../src/herculaneum/chunk_cache.py) and its [tests](../tests/test_chunk_cache.py).

## Artifact portability and larger outputs

**Problem:** the R2 implementation has a conservative single-object cap of 5 GB, and research artifacts need selective restoration.

**Next task:** design independently immutable chunks and a manifest-driven restore path for one representative large artifact.

**Acceptance:** reconstruct the original byte length and SHA-256 exactly; resume after interruption; reject missing, reordered or altered chunks; publish completion only after readback. Do not assume multipart ETags are content hashes.

Start with the [artifact guide](artifacts.md) and [publication implementation](../src/herculaneum/artifacts.py).

## Clearer comparison reports

**Problem:** aggregate metrics can hide regional regressions, model sensitivity and reused evaluation data.

**Next task:** build a compact report from an existing frozen result, showing per-region metrics, worst regressions, reference provenance and the original decision criteria.

**Acceptance:** reproduce every displayed value independently; retain failed and one-class cases; distinguish display similarity, reference-label agreement and physical ink accuracy; do not change thresholds after seeing outputs.

The [experiment-39 comparison](https://github.com/jl-drx/Herculaneum/blob/9f6fe45ab20eb0961d5e06aba5c3038088941180/research/lab/reports/thirtyninth-experiment/RESULTS.md) and [experiment-47 sensitivity study](https://github.com/jl-drx/Herculaneum/blob/9f6fe45ab20eb0961d5e06aba5c3038088941180/research/lab/reports/fortyseventh-experiment/RESULTS.md) provide concrete fixtures.

## Research direction

The [research plan](https://github.com/jl-drx/Herculaneum/blob/9f6fe45ab20eb0961d5e06aba5c3038088941180/research/CURRENT_PLAN.md) at this snapshot prioritises correct-sheet recovery, stronger native-resolution learning and explicit reference/exposure accounting. Historical plans and negative results remain part of the record.

Geometry qualification, data availability, supervision quality and controlled comparisons are still substantive constraints. A successful infrastructure task improves the ability to investigate them; it does not itself establish a reading of a scroll.

New scientific work follows the active research protocol. Paid compute has its own concrete workload approval and budget requirements. Dataset and model terms remain separate from ownership of the project's original code.
