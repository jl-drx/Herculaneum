# Independent supervision-sensitivity arithmetic review

11 October 2026. Reviewed the frozen protocol before target acquisition. After root explicitly reported completed acquisition, read only the four saved compressed TIFF tiles, prior geometry/CT, and fixed result arrays. No network request, inference, training, geometry fit or target-dependent selection was made. The independent checker never imports `analysis.py` or the acquisition implementation.

**Pass, with one preserved floating-point threshold caveat.** Coordinate selection, source sampling, native eligibility and reported metric arithmetic agree. The caveat does not change any finite-set counts or routing decisions, including the primary j14 results.

## Checked work

- Verified crop/origin and shift-sign contract before acquisition: comparison of native `(j,y,x)` against mapped fine at query `(j+dz,y+dy,x+dx)`, with the native eligibility mask kept at the unshifted destination.
- Independently reconstructed **880 transferred arrays / 901,120 score occurrences**, directly indexing the four original compressed TIFF tiles. Primary mean16 sampling uses independent integer sums of each complete 4×4 block, followed by nested one-dimensional interpolation; point sampling indexes full-resolution corners directly. No saved pooled crop or root sampling function was used.
- Compared against every saved array: maximum absolute score difference **2.842170943040401e−14** in 0–255 units.
- Rebuilt native `max(all28layers)>0` eligibility from the original retained CT chunks and checked exact equality.
- Independently recomputed **all 64 finite-set summaries and all 816 pairwise comparisons** from saved arrays. All checked numerical and categorical metrics match exactly; maximum metric difference **0**. The 816 comparisons include all CT alternatives plus the two separate geometry-control contrasts.
- Recomputed all 64 finite-set summaries using the independently resampled scores. Their positive-union, positive-common, common-negative, variable-envelope and pseudo-supervision counts all remain identical. Every routing decision is unchanged. Maximum continuous metric change is **2.1316282072803006e−14**.

Hashes, counts and replay details are in `RESULTS.json`, `TRANSFER_CHECKS.json`, `ENVELOPE_REPLAY.json` and `ALTERNATE_ARITHMETIC_ENVELOPES.json`. Frozen request-plan SHA256 is `29c9adae378c2bfea16321945f43bad7a74ea69516e5b2c8e7efcb9661ddd76e`; source result SHA256 is `71d31ab18bcf4f5878e5506f6746940d1471f4c4bf04ecc3c37da46aa21a7cdb`.

## The threshold boundary caveat

There are **six threshold disagreements**, all for the same unique full-fine query, repeated under support6 candidates00–05 at j11. The four interpolated full-fine corners are exactly `[48,48,48,48]`:

- root's four-product floating-point sum: `47.99999999999999`;
- independent nested interpolation: `48.0`;
- exact rational interpolation of these corners and the retained binary-float fractions: exactly48.

The frozen rule is `>=48`; thus the alternate implementation classifies those six occurrences positive. This is one constant-plateau arithmetic event repeated by the finite shifts, not six independent disagreements about geometry. No primary j14 classification differs. Across all 901,120 score occurrences, 197 lie within1e−9 of48; this count is supplemental arithmetic context, not an uncertainty probability or new threshold band.

`THRESHOLD_BOUNDARY_DISCREPANCIES.json` preserves every occurrence's native position, query, corners, root/alternate values and exact rational comparison. The frozen outputs and threshold were left unchanged. Do not claim every binary candidate map is bitwise invariant to algebraically equivalent interpolation. The finite-set interpretation is invariant in this replay: all 64 envelope counts and decisions are unchanged.

## Scientific reading within the frozen scope

All eight planes and both operators give the same qualitative routing: supports5,6,7 are conditionally sensitive (**48 combinations**), while support4 has no threshold-positive evidence and is inconclusive (**16 combinations**). At primary j14, threshold-variable fractions of the positive union are:

| Support | Mean16 then bilinear | Full-fine point bilinear |
|---|---:|---:|
| 4 | Undefined: zero positive union | Undefined: zero positive union |
| 5 | 32.35% | 29.81% |
| 6 | 32.37% | 32.37% |
| 7 | 22.22% | 22.22% |

This is robust to the two fixed resampling operators and the eight declared query depths within these supports. It means plausible CT-derived XY alternatives can materially change the transferred pseudo-label boundary under the declared numerical rule. It does **not** identify which transfer is right or establish physical ink accuracy. Support4's low scores cannot become a stability pass or an authenticated blank control.

The 2D teacher has no ink-depth annotation; projecting its XY score through a 3D query assumes it remains the same along the fine normal ray. The result does not resolve neighboring-wrap or ink-face uncertainty, identify the correct Z plane, reconstruct historical manual overrides, or clear the earlier CT gate. It supports designing explicit correspondence-uncertainty controls before training, without changing those limitations.

## Replay

```bash
/workspace/.venvs/herculaneum/bin/python recheck.py \
  --study /workspace/herculaneum-lab/reports/pherc0139-supervision-20261011 \
  --packet /workspace/herculaneum-lab/reports/pherc0139-correspondence-20261011 \
  --results /workspace/herculaneum-lab/reports/pherc0139-supervision-20261011/results \
  --output /workspace/scratch/new-independent-review
```

Inputs remain read-only. Python/NumPy and standard-library zlib suffice; no TIFF library, model, network or root analysis import is needed.
