# Herculaneum

**Research software for exploring ink detection in carbonised ancient scrolls.**

A personal project by James Lee, built around the [Vesuvius Challenge](https://scrollprize.org/). Herculaneum combines Python infrastructure, machine-learning experiments and reproducible evaluation to investigate what CT scans can tell us about fragile papyrus without physically opening it.

The engineering problem is practical: acquire only the data an experiment needs, compare model outputs fairly, and preserve enough evidence for someone else to check the result.

[Take the project tour](docs/project-tour.md) · [Try it locally](docs/quickstart.md) · [Explore the research](https://github.com/jl-drx/Herculaneum/blob/9f6fe45ab20eb0961d5e06aba5c3038088941180/research/README.md) · [See the roadmap](docs/roadmap.md)

## What this project demonstrates

| Area | Working capability | Start reading |
| --- | --- | --- |
| Data engineering | Bounded downloads, checksum verification, persistent cache pins and recovery from interrupted transfers | [Cache implementation](src/herculaneum/chunk_cache.py) and [tests](tests/test_chunk_cache.py) |
| Reliable research infrastructure | Immutable run plans, artifact readback and a completion manifest published only after verification | [Artifact implementation](src/herculaneum/artifacts.py) and [tests](tests/test_artifacts.py) |
| ML evaluation | Frozen comparisons, independent metric reconstruction, documented negative results and portable saved-output replay | [Research tour](docs/project-tour.md) and [replay instructions](docs/quickstart.md#replay-a-real-experiment) |

The infrastructure uses **Python 3.12**, Requests and an optional Botocore backend for private R2 storage. The scientific work uses **NumPy, SciPy and PyTorch**, with retained third-party model and data provenance. Development includes AI coding assistance; results are supported by source, tests and recorded checks.

## Selected results

- **Reproduced a published visual result.** The PHerc343 four-model composite matches the published display with RGB mean absolute error **0.0181 on a 0–255 scale** and **99.83% yellow-display overlap (IoU)**. These measure reproduction of an image, not ink-detection accuracy. [Experiment 46](https://github.com/jl-drx/Herculaneum/blob/9f6fe45ab20eb0961d5e06aba5c3038088941180/research/lab/reports/fortysixth-experiment/RESULTS.md)
- **Caught an implementation error before extending an experiment.** An exact-baseline check exposed an in-place tensor mutation in a prediction recorder. The corrected run reproduced all **262,144 baseline pixels** before testing three fixed grid shifts. [Experiment 47](https://github.com/jl-drx/Herculaneum/blob/9f6fe45ab20eb0961d5e06aba5c3038088941180/research/lab/reports/fortyseventh-experiment/RESULTS.md)
- **Made a comparison independently replayable.** A **12.3 MB** capsule contains saved predictions, labels, masks and reference evidence for reconstructing eight supports and 64 ranking metrics without model weights or CT volumes. [Experiment 39](https://github.com/jl-drx/Herculaneum/blob/9f6fe45ab20eb0961d5e06aba5c3038088941180/research/lab/reports/thirtyninth-experiment/RESULTS.md)

**Research status:** no new readable ink, authenticated letters, transcription or translation has been established. The broader aim remains recovery of useful evidence from the scrolls; the results above demonstrate reproducibility and evaluation work toward that aim.

## Try it locally

From a checkout, inspect the command interface with Python 3.12:

```bash
PYTHONPATH=src python3.12 -m herculaneum --help
```

The [quickstart](docs/quickstart.md) walks through a small offline demo: publish a synthetic result, verify it, then confirm that altered bytes are rejected. It uses temporary files and the Python standard library. A separate, pinned research checkout provides the real experiment replay.

## Find your way around

- `src/herculaneum/` — the cache, artifact stores and command interface.
- `tests/` — failure, recovery, integrity and CLI checks.
- `baseline/` — retained detector adapters, upstream source and notices.
- `docs/` — setup, architecture and operational contracts.
- **Research branch** — the experiment archive and replay fixture live on [`codex/plinius-pherc343-reproduction`](https://github.com/jl-drx/Herculaneum/tree/codex/plinius-pherc343-reproduction), separately from the infrastructure on `main`. The selected results above are pinned to snapshot `9f6fe45`; they are not a live progress feed. [Active research plan](https://github.com/jl-drx/Herculaneum/blob/9f6fe45ab20eb0961d5e06aba5c3038088941180/research/CURRENT_PLAN.md).

For development, use the [full CPU setup](docs/quickstart.md#full-cpu-development-environment), [architecture](docs/architecture.md), [recorded validation](docs/validation.md), [artifact guide](docs/artifacts.md), [baseline guide](docs/baseline.md), [reference specification](docs/reference-experiment.md), [GPU policy](docs/gpu-pilot.md) and [cloud prerequisites](docs/cloud-connections.md).

## Attribution and access

The project's contribution is its research workflow, infrastructure, comparisons and verification around existing datasets and models. Upstream model architectures, checkpoints, scan data and published results retain their own attribution and terms; see the [third-party notices](baseline/vendor/villa-ink-baseline/THIRD_PARTY_NOTICES.md).

This repository is private; a reviewer needs repository access. Bulk artifacts have separate storage and access requirements. No blanket license is asserted for original project code or bundled research data. Preserve each asset's terms when reusing it.
