# Independent native-reference comparison verification

**PASS on the first saved-output verification attempt, with unchanged frozen code and tolerances.** This is an execution/metric verification result; the protocol has no performance pass gate.

All 150 normalized input fingerprints matched both the retained seed42 audit and native capture. The original 520,794 primary pixels were preserved: 86,535 positive and 434,259 negative supplied-reference labels. Both outputs covered every primary pixel. All 508 loaded-state value hashes and 244 alias pairs matched the independently read qualified safetensors before and after inference. Exactly one construction, one strict load and 150 native model forwards were verified; the checker made no model call. Sources and inputs remained unchanged.

Every raw, uint8 and author-decoded ROC-AUC/AP, unique-score count, decoded endpoint fraction, paired difference statistic, and separate equal-context parent summary agreed with the author calculation within `1e-12`. Saved quantization and decoding arrays matched exactly for both models. Native probability sums, weights and raw blends were bitwise exact against reconstructed saved-tile arithmetic. Independent sigmoid maximum error was `5.960464477539063e-8`, analytic Hann error `1.7881393432617188e-7`, and retained seed42 blend reconstruction error `2.980232238769531e-7`, all within frozen arithmetic tolerances.

| Physical parent | Seed42 raw AUC | Native raw AUC | AUC delta | Seed42 raw AP | Native raw AP | AP delta |
|---|---:|---:|---:|---:|---:|---:|
| Paris1, two contexts | 0.784788 | 0.630308 | −0.154479 | 0.240483 | 0.203584 | −0.036899 |
| Paris2, four contexts | 0.657642 | 0.705446 | +0.047804 | 0.384286 | 0.487602 | +0.103316 |

| Context | Native minus seed42 raw AUC | Native minus seed42 raw AP |
|---|---:|---:|
| frag3-context1 | −0.021743 | +0.059380 |
| frag3-context2 | −0.287216 | −0.133177 |
| frag1-context1 | −0.015171 | −0.007672 |
| frag1-context2 | +0.004180 | +0.226094 |
| frag2-context1 | +0.126344 | +0.055517 |
| frag2-context2 | +0.075862 | +0.139326 |

The complete independent metrics, class counts and arithmetic errors are in [verification-v1.json](verification-v1.json), SHA256 `b318016263d852d1de7843a6b5dfe1540a2feb779d044ca500df64ed0396bd3e`. Verification used 1.830 seconds wall, 2.011 seconds CPU and 308,523,008 bytes peak RSS. The checker was frozen before native execution; no model replay or tolerance adjustment occurred. The preserved preparation-only fixture correction is documented in [ACCEPTANCE_RULES.md](ACCEPTANCE_RULES.md).

All six contexts remain reused development regions on two physical parents. These numbers measure supplied-reference agreement under the existing input/annotation contract; they do not establish teacher independence, accurate native physical alignment, a causal training-method effect, fresh holdout performance or decipherment. Scientific interpretation and any subsequent study design remain with root. No paid resources were used.
