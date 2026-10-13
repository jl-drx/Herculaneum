# Independent study52 completed-map verification, version 2

**PASS** at 11 October 2026, 08:52 BST (07:52 UTC).

This version changes only manifest scope enforcement. The original manifest is verified against an independently enumerated exact set of 71 generated evidence files: `assess_complete.py`, `ASSESSMENT_PLAN.json`, `METRICS.json`, 16 full-strip PNGs, 48 cell PNGs and 4 contact-sheet PNGs. Each of the three image folders must contain exactly its expected files. Every original manifest entry still passes its retained size and SHA-256 checks.

Later recovery notes, visual-review documents and independent-verification reports are outside that original manifest. Their coverage belongs to the parent’s whole-packet portable manifest; this verifier makes no claim that the original output manifest covers them. This avoids treating legitimate subsequent documentation as a failure of the frozen generated output set.

All source hashes, map checks, factorial pair enumeration, support, statistical calculations and image-pixel checks are unchanged from v1. The original verifier, successful results and report remain intact in `/workspace/scratch/phase52-completion-independent-20261011` and its retained copy. `V1_TO_V2.diff` records the complete script changes.

The full rerun verified:

- 48 saved maps and all 168 distinct comparisons: 24 padding, 24 order, 72 cross-model and 48 depth.
- All 672 full-strip/cell scores. Maximum Pearson discrepancy remains 3.33e−16; maximum MAD discrepancy remains 1.42e−14 uint8-equivalent levels.
- 64 exact panels, covering 192 depth-region views with 1:1 source pixels, fixed nearest-even display and RGB (139,90,43) unknowns.
- 4 contact sheets and all 48 reduced tiles.
- 786,240 common-support pixels and 192 unknown pixels.

No frozen generator, plan, metrics, source map, mask or PNG was modified. No inference, network activity or paid compute occurred. This is a reporting-reproducibility check, not evidence of ink accuracy or authenticated text.

Replay with the existing runtime:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/.venvs/herculaneum/bin/python /workspace/scratch/phase52-completion-independent-20261011-v2/verify.py
```

The script uses the original explicit source paths and writes only its own JSON reports beside itself. `VERIFICATION.json` contains every score and image check; `SUMMARY.json` contains the concise proof. `MANIFEST.json` covers the files in this v2 evidence directory when created.
