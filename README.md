# Herculaneum research

Reproducible tooling for bounded Vesuvius Challenge experiments. GitHub holds code, configurations and provenance definitions; scan chunks come from public Vesuvius storage, and private artifact storage preserves results and checkpoints.

The project has not established new readable ink, a transcription or a translation. The existing approximately 9 micrometre detector is a baseline; native PHerc1667 method reproduction and transfer to eligible PHerc0800 are separate scientific tasks.

## Local setup

Python 3.12 and `uv` are required. The reviewed CPU environment is hash-pinned:

```bash
bash scripts/setup_cpu.sh
source /workspace/.venvs/herculaneum/bin/activate
herculaneum --help
```

This installs the package, verifies retained sources and runs local infrastructure tests. It does not download scans or checkpoints, connect to R2, or start paid compute. The CPU lock must not be used as a CUDA installer.

## Workflow

- [Architecture and current scope](docs/architecture.md).
- [Chunk cache](docs/cache.md): bounded, verified source acquisition and reuse.
- [Artifact storage](docs/artifacts.md): local validation and a private R2 backend with completion manifests.
- [Existing detector](docs/baseline.md): unchanged pinned CPU adapters and their scientific limits.
- [Reference specification](docs/reference-experiment.md): native PHerc1667 control and unresolved prerequisites.
- [GPU pilot policy](docs/gpu-pilot.md): the approved £10 aggregate cap and launch gates.
- [Validation evidence](docs/validation.md) and [cloud connection prerequisites](docs/cloud-connections.md).

Large arrays, checkpoint binaries, caches, credentials and generated reports are excluded from Git. Preserve per-asset terms when acquiring or retaining data. Third-party source notices and licenses accompany the focused upstream snapshot; no blanket license is asserted for the original project code.
