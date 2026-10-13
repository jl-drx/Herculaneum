# Study49 retention and replay

The Git export contains exact source/report copies; the separate PHASE49 LFS
supplement contains study49's raw qualification chunks, renderer outputs, fixed
model inputs/probabilities, support masks and execution receipts. It does not
transfer secrets or install a running worker. The source cache is14MiB; duplicate
renderer cache entries are retained so the complete study directory is recoverable.
Old checkpoint containers, the pinned renderer distribution and upstream source
remain in their earlier independently indexed workspace groups/receipts.

For saved-result validation, run `verify_saved.py` from its restored study directory
with NumPy installed. It is read-only and performs no inference or download.
The exact root path is required only by historical write-once producers and the
cumulative full-lab verifier; never rerun those producers inside sealed directories.
The new qualification used `/workspace/.venvs/pherc343` (Python3.14, Torch2.13 CPU);
source fetching/geometry preparation used `/workspace/.venvs/herculaneum`.

The fullfield source/chunk/runtime/output-support JSON files are forecasts, not
receipts for an acquired or inferred full field. Study49 contains20 tile forwards,
zero completed full target maps. Virtual canvases have explicit unknown placeholders;
only their fixed qualification crops may be treated as observed input.
