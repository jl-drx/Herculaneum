# Independent result review

**PASS for the frozen finite operator study. No learning or ink-accuracy conclusion follows.**

All 64 cases were independently recomputed from the retained teacher crop and original correspondence query field, without importing production code. Native eligibility was independently reconstructed from the 16 retained CT chunks using all 28 planes. Input hashes, crop decoded-pixel hash, geometry-query and support hashes, and the production input manifest were checked. The verifier then rebuilt A/B/C from saved candidate transfers and independently recomputed all reported scalar summaries, masks and descriptive decisions.

- 1,536 candidate transfer arrays independently sampled from source, plus all four target arms in each case.
- 8,000 scalar comparisons and 1,024 array assertions passed; scalar replay error was exactly zero.
- 60 exact rational mean/covariance checks passed.
- Independent nested bilinear arithmetic differed from production by at most 4.2633e-14 original source levels.
- All case/query hashes and the result checksum manifest passed.

The arithmetic difference produces one retained, nonprimary threshold discrepancy: seven candidate values in `point_bilinear_s6_j11` are exactly 48 under the independent expression and 47.99999999999999 under production. Common masks, active unions, target-arm classifications and all descriptive decisions agree. No rounding or tolerance was inserted into the declared threshold.

## Findings

Primary means below use original source levels, so divide by 255 for the protocol's target units.

| Support | Common pixels | Active-union pixels | B–C active MAE | B–D active MAE |
|---|---:|---:|---:|---:|
| 4 | 1,024 | 0 | undefined | undefined |
| 5 | 1,024 | 139 | 1.004080 | 0.173677 |
| 6 | 1,024 | 451 | 1.250221 | 0.271024 |
| 7 | 992 | 155 | 0.797706 | 0.051410 |

The global control needs source locations outside the retained crop for 32 pixels of support 7. The protocol correctly excludes that same column from every arm in every case. No extra source data were fetched. This changes the valid comparison domain relative to the earlier local-only sensitivity result; it is not a model failure.

B–D stays below one source level in all 48 cases with nonempty positives, with ranges 0.1681–0.2542, 0.2674–0.3289 and 0.0509–0.1777 on supports 5/6/7. In this retained score field, sampling at the local candidate centroid closely approximates averaging the candidate scores.

B–C is not identically zero. Its support-5 range crosses the declared descriptive scale (0.9825–1.0242); support 6 stays above it (1.2446–1.2542), and support 7 below (0.7718–0.8169). Treating the primary support-5 value of 1.00408 as a scientific success would overinterpret a descriptive threshold on an inspected packet. Neither small target differences nor this threshold crossing establish their effect on optimisation.

The evidence does not currently justify a special geometry-uncertainty learner. The result supports retaining an ordinary soft-target baseline, fixed smoothing and centroid controls in a future matched comparison if a defensible evaluation reference is qualified. It does not establish that C is a better target, or that local geometry is useless. C remains geometry-conditioned and differs from B in mean and spread as well as support association. See `DESIGN_REVIEW.md` for these limitations.

## Replaying the review

Standalone compact replay requires only this packet and NumPy:

```bash
python review/compact_verify.py --packet /path/to/packet --output /path/to/new-output-directory
```

This verifies source sampling at the **saved derived UV queries**, their declared checksums, target construction, masks, every scalar summary, and exact kernel moments. It explicitly does not independently derive those coordinates from meshes or recompute native eligibility from CT. Output is written to a new directory and the packet is read-only.

The stronger original-source replay also needs the prior correspondence and supervision packets:

```bash
python review/independent_check.py --root /path/to/packet --prior /path/to/pherc0139-supervision-20261011 --correspondence /path/to/pherc0139-correspondence-20261011
python review/compare_outputs.py --root /path/to/packet
```

That mode writes its independent arrays and receipts in `review/`, verifies original-source checksums and native max-28 eligibility, and re-extracts the candidate queries from the original correspondence field. It does not rerun the mesh inversion itself. A stale or altered original source fails rather than silently falling back to compact mode.

`COMPARISON.json` is the stronger original-source comparison receipt. `compact_verify.py` was also exercised successfully against the self-contained input copy; its output reported the same scalar and source-transfer tolerances. No training, model inference, new data or paid compute was performed for this review.
