Phase24 input preparation passed: 871 assertions across the six retained development contexts and all 150 tiles. No models were constructed or called, no weights were decoded, and no network requests were made. Prior experimental evidence was read without modification.

The authoritative execution contract is `MODEL_INPUT_CONTRACT.json` (SHA256 `7eef20e874722314b4c32c99acbe846e8ec63da7640cddd2d204028299b95259`). The unchanged forward adapter must normalize each original patch once. The capture wrapper must verify its original normalized fingerprint, reverse the normalized tensor's depth axis 2 in BCZYX layout, make it contiguous, verify the canonical reversed fingerprint, and only then call the model. The model receives depth indices 16 through 0, with no second normalization, raw-input reversal, new pooling, or padding. Both checkpoints use these same paired fingerprints. Depth direction means dataset raster ordering; anatomical surface orientation is not authenticated here.

`INPUT_CONTRACT.json` and `verification.json` preserve the first diagnostic preparation exactly. That file's proposed raw-reversal execution policy was superseded by the root's final normalized-tensor boundary before any model execution. `derive_model_contract.py` constructed the authoritative contract from the completed diagnostic JSON alone: zero scientific array reads, normalizer calls, or model calls. `MODEL_INPUT_DERIVATION.json` binds that derivation and its inputs. Consumers must bind the authoritative model contract, rather than the diagnostic file's obsolete execution policy.

All 150 recomputed original normalized fingerprints exactly match the retained phase16 seed42 and phase23 native forward captures. The six sources remain FP32 arrays of shape 17 by 384 by 384. Each context retains the same 25 tiles: 128 by 128, stride 64, row-major y then x, with coordinates 0, 64, 128, 192, and 256 on each axis. Per-plane hashes verify that depth reversal preserves plane values and reverses only their order. Source arrays remained unchanged by normalization.

The diagnostic comparison also found `normalize(raw[::-1])` byte-equal to `normalize(raw)[::-1]` for all 150 actual tiles. No scale fallback was used: scaled MAD ranged from 7.196778073883056 to 74.64866454391479. The unused preclip standard deviation differed on 42 tiles by 1.9073486328125e-6. These actual observations do not establish general commutation; reduction order can matter when fallback standard deviation supplies the normalization scale. The canonical operational boundary therefore defines reversal directly on the original normalized tensor. The diagnostic raw-reversal comparison neither defines nor gates that boundary.

Original primary support is retained exactly as known-label mask intersected with the central square `[32:352, 32:352]`; it is not the entire known-label or surface mask. Known pixels remain within the surface mask. The adapter receives no mask and the retained forward evaluated masks cover all 147,456 canvas pixels per context. Scoring must retain the original primary mask and these unchanged class counts:

| Context | Parent | Positive | Negative | Primary pixels |
| --- | --- | ---: | ---: | ---: |
| frag3-context1 | Paris1 | 9,434 | 82,951 | 92,385 |
| frag3-context2 | Paris1 | 5,771 | 83,490 | 89,261 |
| frag1-context1 | Paris2 | 19,400 | 69,159 | 88,559 |
| frag1-context2 | Paris2 | 12,633 | 70,681 | 83,314 |
| frag2-context1 | Paris2 | 19,049 | 61,660 | 80,709 |
| frag2-context2 | Paris2 | 20,248 | 66,318 | 86,566 |
| Total | | 86,535 | 434,259 | 520,794 |

The canonical contract binds the phase23 input contract, seed42 configuration and checkpoint metadata/state digest, native load-verification receipt, source arrays and masks, both models' saved forward output descriptors, and exact original/reversed tile fingerprint pairs. Retained forward logits, raw scores, and evaluated masks were checked against their byte pins, shapes, dtypes, finite-value conditions, and coverage. Forward output numerical reconstruction remains supported by the prior sealed audits; this input audit did not rerun reconstruction, inference, training, accuracy metrics, or any selection process.

The one scientific input-audit process completed successfully in 4.347738632997789 seconds. `CODE_FREEZE.json` binds the verifier source used for that run. The subsequent contract derivation and final metadata integrity review did not revisit scientific arrays. `RETENTION_MANIFEST.json` seals this directory, excluding itself. All six contexts remain reused development diagnostics; this work adds no unseen evaluation evidence and does not revalidate physical geometry or annotation alignment.
