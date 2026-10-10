# Independent conservative-continuation numeric review

**PASS.** The sealed independent NumPy checker reconstructed all 64 immediate pre/post pairs, 256 fixed-risk batch evaluations and 64 cache-derived baseline batches, making 6,164 scalar comparisons. All normalized input/target/mask fingerprints, historical teacher-cache FP32 bit patterns, scheduled supports/counts, first-step sentinels, fixed-risk current-batch bit comparisons and published reductions/gates passed. The [verification result](completed-review/independent-verification.json) is 340,084 bytes, SHA256 `8ae053a9095b2d546f2401dcaa8779ed5d5a2cece954d5decaad3edb1b469bf8`.

Both arms pass the exact frozen final gate. The following values use the retained worker mean floats and the declared subtraction order, final equal-batch mean minus cache-derived initializer equal-batch mean. Both base and full deltas must be at most -0.0001, inclusively, without gate tolerance.

| Arm | Final base delta | Final full delta | Eligible |
| --- | ---: | ---: | --- |
| A_preserve | -0.010478727519512177 | -0.006830823261972707 | Yes |
| B_preserve | -0.013059169054031372 | -0.008859663339445989 | Yes |

Independent FP64 per-pixel reconstruction gives the same gate decisions. Maximum difference from a saved FP32 BCE/base/full/hard-BCE scalar was **1.5920×10⁻⁷**, within the unchanged absolute tolerance 2×10⁻⁶. Maximum KL or probability-drift scalar difference was **6.9389×10⁻¹⁸**, within 10⁻¹². Logged equal-batch means were additionally checked exactly against `math.fsum` of their 32 retained per-batch scalar values divided by 32. Counts, bit patterns and gate comparisons received no tolerance.

Actual same-batch immediate base/full increases were 0/0 for A and 1/0 for B across 32 updates; the retained worker precision and independent FP64 reconstructions agree on those counts. The checker verifies every update and fixed snapshot at updates 1, 8, 16 and 32. It selects no best snapshot, changes no cutoff and invents no final fixed risk for the older 1e-4 study.

The checker imports no worker, model, historical loss, preprocessing or augmentation helper. It reconstructs float32 robust normalization, independent D4 coordinate maps, manual/pseudo participation and sentinel target canvases. BCE uses stable log probabilities; KL explicitly uses the initializer-to-student probability/log-ratio terms. Manual and pseudo participant means remain separate. Fixed risk averages the 32 scheduled batch losses equally, preserving overlap and repeated exposures rather than pooling or deduplicating pixels. The pseudo hard labels remain the seed43 map, distinct from the seed42 preservation logits.

The [source freeze](code-freeze.json) and [final-plan source contract](FINAL_PLAN_CONTRACT.json) were recorded before actual retained-array review. Root subsequently authorized reconstruction after training and analysis both passed. The checker verified 238 source/input pins before and after reconstruction and excluded eight weight/checkpoint pin records from reads. Input pins include the final plan SHA256 `3151f0c5ef26bc0ef3c61266ab6ee45051506037773b8aa62f42a50d925748ac`, gate SHA256 `0e07252f2705e7905979a7c29dc90a7717133783d334ac3cd269711cf5351b7d` and summary SHA256 `9b48a9f187d68dc9ebf57b88cef295547d553b5a5d0a1f181c218d9371ae7284`. The preferred preparation fixtures passed 22 mathematical and four preprocessing/fingerprint cases, with no retained-array evaluation during preparation.

The reported 384 student forwards, zero teacher forwards, 64 backwards and 64 optimizer updates match the frozen source loops, raw-logit extents and completed worker/controller receipts. Model, optimizer, gradient, RNG and checkpoint validation are **worker-reported checks**: this reviewer checked their recorded consistency and lineage without independently executing them or reading weights. The stronger independent claims here concern retained numeric outputs, masks, fingerprints, losses, risk reductions, sentinel logits and gate arithmetic.

These are empirical adaptation risks on the same PHerc0139 training regions and schedule. The gate authorizes the separately frozen research retention comparison; it establishes no held-out performance, ink authentication, letters or translation. Root owns scientific interpretation and the retention decision.

Actual reviewer operations were retained NumPy/JSON reads and calculations only: **zero Torch imports, model calls, checkpoint reads, network calls, new predictions or training**. Earlier evidence remained unchanged. The final [review receipt](REVIEW_RECEIPT.json) and [manifest](MANIFEST.json) bind this package; no further edits are planned.
