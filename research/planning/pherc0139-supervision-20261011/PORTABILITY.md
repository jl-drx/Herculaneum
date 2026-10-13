# Private restoration and offline replay

Use private branch `codex/plinius-pherc343-reproduction` in jl-drx/Herculaneum.
The ordinary Git text directory retains the protocol, results and code. The
bounded LFS archive retains compressed source tiles, decoded crop, all transferred
arrays, panels and provenance. It contains no credentials or model weights.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch codex/plinius-pherc343-reproduction https://github.com/jl-drx/Herculaneum.git
cd Herculaneum
git lfs pull --include='research/workspace/bundles/pherc0139-supervision-20261011.tar.gz' --exclude=''
python3 research/planning/pherc0139-supervision-20261011/restore_packet.py --bundle research/workspace/bundles/pherc0139-supervision-20261011.tar.gz --destination /workspace/review-supervision
```

The restorer checks archive and every file SHA256 and rejects unsafe archive
members or an existing destination. It requires Python3.11+ standard library.
To READ results and panels, no correspondence packet, CT download, inference or
GPU is needed. To RECOMPUTE geometry-linked target transfer, restore the prior
correspondence packet using its PORTABILITY.md; download only that additional
168MB archive. The following assumes it is at `/workspace/review-correspondence`.

Use NumPy2.5.3, the version verified here (Python3.12 runtime). The primary replay
needs NumPy only; the prior packet's optional CT score replay additionally needs
SciPy. Run the target replay in a new output directory:

```bash
cd /workspace/review-supervision
python3 acquire_teacher.py replay
python3 analysis.py analyse --packet /workspace/review-correspondence --candidates CT_CANDIDATES.json --teacher teacher/crop.npy --teacher-metadata teacher/CROP.json --output /workspace/replayed-supervision
python3 review/independent/recheck.py --study /workspace/review-supervision --packet /workspace/review-correspondence --results /workspace/replayed-supervision --output /workspace/rechecked-supervision
```

`acquire_teacher.py replay` verifies the four retained compressed tiles and
reconstructs the identical crop without network. **Do not use `fetch` for replay.**
The independent checker reads source tiles and retained CT/geometry, imports no
study analysis, and reports the preserved threshold-boundary arithmetic caveat.
It requires no models or torch. Historical source-fetch scripts/receipts contain
original absolute paths and are archival provenance; they need not be rerun.

The older draft `SYNTHETIC_FIXTURES.json` and interpreter caches are omitted;
`SYNTHETIC_CHECKS.json` is the authoritative pre-acquisition fixture receipt.
The design review is a recommendation; PROTOCOL.md records the final choice of
operator, w040 threshold48, and conditional2D interpretation before acquisition.
All four supports, both operators and every candidate remain in the packet.
