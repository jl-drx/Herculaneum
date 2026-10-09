# Independent native synthetic compatibility review

**PASS on the first verification attempt.** Both exact source-bound factories strictly loaded all 508 named checkpoint entries, with 264 actual storage groups, 244 duplicate-name pairs and 34,546,498 unique parameters. Independent safetensors NumPy reads matched every loaded per-key hash before and after inference. Source and state remained unchanged.

The historical and current factories were bitwise identical over all nine frozen synthetic cases:

| Comparison | Exact values / total | Maximum absolute difference |
|---|---:|---:|
| Pre-sigmoid tile logits | 180,224 / 180,224 | 0 |
| Tile sigmoid probabilities | 180,224 / 180,224 | 0 |
| Final raw blended scores, including unevaluated zero case | 167,936 / 167,936 | 0 |

All selected depth indices, normalized tensor bytes, tile schedules and evaluated masks matched exactly. There were no cross-factory uint8 or clipped-author-decoding mismatches. The zero input produced no model call, no evaluated pixels and zero output. Each worker made exactly eleven model forwards; the independent checker made none.

Independent stable sigmoid reconstruction differed by at most `5.960464477539063e-8`. The analytic double-precision Hann formula differed from the worker FP32 window by at most `1.7881393432617188e-7`. Both were within the acceptance rules frozen before native outputs. Reconstructed accumulation, weight sums and raw blending were exact; each worker's quantization and clipped decoding matched its own recorded conversion arithmetic exactly.

The independent receipt is [verification-v1.json](verification-v1.json), SHA256 `973cf4f841e7c46dff30c03398a44fe21eb91a63825951ac5867870737677da4`. Verification took 0.713 seconds wall / 0.912 seconds CPU, with 307,654,656 bytes peak RSS. Preparation included 32 independent evidence tests, 23 source-bound preprocessing comparisons, nine exact historical Git-blob checks and 20 exact historical AST-extraction checks, plus the reviewed author's 94 passing toy checks. No model retry, checkpoint mutation, source edit or tolerance adjustment occurred after freezing.

This qualifies strict loading and the named fabricated-input CPU compatibility boundary. It does not establish historical CUDA equality, untested preprocessing branches, native scan correctness, benchmark independence, ink accuracy or an improvement over Plinius seed42. The author clipping `(uint8-64)/128` remains an output decoding convention, not calibrated probability. No real scan or label was read for this qualification and no GPU was rented.
