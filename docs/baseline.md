# Existing CPU detector

`baseline/` preserves four reviewed lab scripts without modification. `baseline/source-manifest.json` records their SHA-256 values. The focused upstream modules retain their own pinned manifest, original license and additional source notices. `scripts/verify_sources.py` verifies the retained snapshot.

The approximately 9 micrometre `scrollprize/ink_9um` checkpoints are pinned to revision `7109667e2607db1b90c37c8b09cb876ea7fe7bb1`. Only their public metadata is stored in Git. Downloading both checkpoint binaries is optional and about 264 MiB:

```bash
python baseline/fetch_ink_models.py
```

The adapters consume bounded `.npy` arrays, immutable provenance and explicit checkpoint paths. Inputs and generated results belong outside Git:

```bash
python baseline/run_ink_large.py /path/to/surface.npy \
  --provenance /path/to/provenance.json --mask /path/to/geometry-mask.npy \
  --seed42-checkpoint baseline/data/models/ink9um-seed42-step075000.pth \
  --layer-start 7 --max-total-tiles 1200 --output-dir /path/to/new-results
```

The large adapter expects uint8 `[31,H,W]`, 128–2048 pixels in XY, and caps scheduled checkpoint tiles before mask skips. Source and checkpoint hashes, strict model loading, robust-MAD normalization, FP32 CPU inference and Hann blending are retained. Reference inputs with 17–31 depth planes use `run_ink_reference.py` instead; never pad them to satisfy a different adapter.

Historical acquisition orchestration and frozen experiment analyses are retained in the lab snapshot, not promoted as portable tools: some depend on exact absolute paths and experiment hashes. Moving or changing a frozen sidecar does not preserve its provenance.

The current reference annotation agreement overlaps training. Target predictions have not supported coherent letters. Raw scores and two-seed agreement do not establish ink authenticity, reading accuracy or independent validation. Native 2.4 micrometre PHerc1667 inference needs a different audited recipe.
