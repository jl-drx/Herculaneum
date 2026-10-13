# Private download and offline replay

The self-contained packet needs only this single LFS object. Do not download all
historical bundles. No inference, weights, CT service, R2 credentials or GPU is
needed. Python 3.12.14 and the versions in requirements-replay.txt were used here.
Memory stays below the spare VM's 9GB limit; the archive and restored files need
well below 1GB before any optional duplicate replay directories.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch codex/plinius-pherc343-reproduction https://github.com/jl-drx/Herculaneum.git
cd Herculaneum
git lfs pull --include='research/workspace/bundles/pherc0139-six-site-supervision-20261011.tar.gz' --exclude=''
python3 research/planning/pherc0139-six-site-supervision-20261011/restore_packet.py \
  --bundle research/workspace/bundles/pherc0139-six-site-supervision-20261011.tar.gz \
  --destination /workspace/review-six-site-supervision
```

The standard-library restorer checks the archive and every file's SHA256 and
length, rejects unsafe archive members, and refuses an existing destination.
The report and panels can then be read without installing any scientific code.
For replay, create a Python environment with the pinned requirements and run
from the **restored packet**, whose relative inputs are self-contained:

```bash
cd /workspace/review-six-site-supervision
python3 acquire.py replay
python3 analyse.py --output /workspace/replayed-six-site-supervision
python3 verify_results.py \
  --results /workspace/replayed-six-site-supervision \
  --output /workspace/rechecked-six-site-supervision.json
```

`acquire.py replay` uses the retained compressed tiles. Do not run `fetch` or
`metadata.py` for replay. `prepare.py` can regenerate the target-free alternatives,
source bounding rectangles and full teacher footprint checks offline; it should
produce the identical ALTERNATIVES.json and PROVENANCE.json. It overwrites those
two generated files, so use a disposable restored copy when testing this step.

The original catalogue and earlier geometry qualification are retained as pinned
evidence. Fine/native coordinates and saved inverse queries are included. The
packet reproduces the new sensitivity audit and its teacher input-footprint
checks; recomputing the original CT inverse solution or its NCC landscapes from
raw CT is the scope of the separate eight-site CT packet. Prior structural
failures and the declared teacher ancestry limitations remain unchanged.

To regenerate the fixed-scale primary panels, remove/rename the existing panels
folder in a disposable copy and run make_panels.py with a writable Matplotlib
cache, for example MPLCONFIGDIR=/tmp/plinius-mpl-cache. Do not tune display
contrast or use the figures to choose new regions.

The short audit ran in this chat. These are portability instructions, not a
request to dispatch another VM or a paid job. Checkpoint retrieval for the active
native-learning worker uses its separate documented branch and packet.
