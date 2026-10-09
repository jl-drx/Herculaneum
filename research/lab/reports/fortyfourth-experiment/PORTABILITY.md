# Phase44 input recovery and scope

Git preserves selected protocols, scripts, reports, all per-arm numeric results and compact review figures. It does not hold the large raw-height row bodies, crop arrays, score arrays or model weights. The retained local chain remains the authoritative full artifact set. R2 was not used.

`CROP_REPLAY_PLAN.json` pins six immutable public ranges, three raw-height row blocks and three complete label PNGs. `research/tools/materialize_phase44.py` reconstructs the nine required height/label/base-support arrays in a new directory, verifies their exact recorded SHA-256/length, and never starts training. The source whole-file LFS digest is verified for labels; partial raw-height downloads are verified against the retained range digest, not represented as whole-file verification. Transfer is bounded by the plan and requests fail closed on ignored or changed ranges. Signed redirect queries remain in memory and are never saved.

Example from the selected research checkout, using an environment with NumPy, SciPy, Pillow and requests:

```bash
python research/tools/materialize_phase44.py \
  --plan research/lab/reports/fortyfourth-experiment/CROP_REPLAY_PLAN.json \
  --output /tmp/plinius-phase44-inputs
```

A local offline-cache test reconstructed all nine arrays byte-identically at `/tmp/plinius-phase44-reconstruction-final`; see `INPUT_MATERIALIZATION_VERIFICATION.json`. The separate relocation/public-network replay path was not executed and no fresh-machine training claim follows. The original public range download itself passed the phase44 intake checks.

Archived training workers retain write-once GO checks and original paths. Do not run them inside sealed evidence. A new run must bind its own paths/inputs/environment and the same design before execution. Materialized `common-support.npy` is the initial pixel-QC mask; the training worker additionally excludes interpolation neighborhoods and32px crop borders. Both steps are prescribed and verified, and final support statistics are retained in `training/input-stats.json`.

Data, labels and derivative figures: Vesuvius Challenge (2026). *Optical profilometry of opened Herculaneum papyri*. https://doi.org/10.1038/s41598-026-58467-1. CC BY-NC4.0. Preserve source license and attribution; no blanket project license is asserted. Absolute lateral calibration remains unresolved. No CT observations or model-derived pseudo-labels are included in this experiment.
