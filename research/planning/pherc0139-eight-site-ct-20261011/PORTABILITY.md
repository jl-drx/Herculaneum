# Private download and offline replay

The self-contained bundle contains the selected public coordinate/metadata
inputs, 72 raw CT chunks, geometry queries, fixed protocol/code, source receipts,
numerical results and all 16 review panels. It contains no model weights,
teacher/annotation values or credentials. No other workspace is required.

From an authorized checkout of private branch
`codex/plinius-pherc343-reproduction`:

```bash
git lfs fetch --include='research/workspace/bundles/pherc0139-eight-site-ct-20261011.tar.gz' --exclude='' origin codex/plinius-pherc343-reproduction
git lfs checkout research/workspace/bundles/pherc0139-eight-site-ct-20261011.tar.gz
python research/planning/pherc0139-eight-site-ct-20261011/restore_packet.py \
  --bundle research/workspace/bundles/pherc0139-eight-site-ct-20261011.tar.gz \
  --destination /tmp/plinius-eight-ct
```

The standard-library restorer verifies the archive and every member's length
and SHA256 against BUNDLE.json, rejects unsafe/duplicate paths, and refuses an
existing destination. Retain the repository's BUNDLE.json and CHECKSUMS.sha256
alongside the archive. Original sources retain their upstream terms; see
`research/THIRD_PARTY_NOTICES.md`. No blanket relicensing is asserted.

Python3.12.14 with the pinned requirements was used. NumPy, SciPy and tifffile
are sufficient for numerical replay; Matplotlib is only for optional figure
regeneration, and Requests only for the already completed acquisition.

In the restored directory:

```bash
python verify_preflight.py
python verify_results.py
python analyse.py --output /tmp/plinius-eight-ct-replay
```

Both verification receipts should pass. Compare the replay's RESULTS.json and
per-site arrays/reports against the restored `results/` files. OUTPUTS.json
contains location-relative paths and is not a scientific numeric result.
Relocated replay in the original software environment is tested for exact
bytes; cross-platform floating-point comparisons can use the independent
checker tolerance of 1e-9. That numerical tolerance changes no scientific gate.

The scripts use one numerical CPU thread and need no GPU, credentials, new
downloads, CT server or neural model. `acquire.py` deliberately refuses an
existing acquisition. Do not run it to replay. `prepare_geometry.py` similarly
refuses an existing geometry directory. For optional full geometry regeneration,
use a disposable restored copy, rename its `geometry/` directory, run the script,
and compare every regenerated file against the saved directory before proceeding.

Figures are already included. If regenerating them in a disposable copy, first
rename its existing `panels/` directory, then run:

```bash
MPLCONFIGDIR=/tmp/plinius-mpl-cache XDG_CACHE_HOME=/tmp/plinius-cache python make_panels.py
```

No figure contrast is fitted. The temporary cache settings accommodate this
environment's read-only home directory. These are recovery instructions, not
a new spare-VM assignment or an Actions workflow dispatch.
