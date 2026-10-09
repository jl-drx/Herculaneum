# Native reference comparison: execution and independent verification passed

The fixed comparison completed on its first attempt with one model construction, one strict native safetensors load and exactly150 full-model forwards. All150 normalized input fingerprints matched the retained seed42 inputs before their native calls. Every original primary pixel remained scored: **520,794 pixels**, including86,535 supplied positives and434,259 supplied negatives. No mask intersection, registration change, depth search, warmup, training or threshold classification occurred.

Both raw ranking metrics vary by context. These are descriptive supplied-reference agreements on six reused development regions; there is no retrospective performance pass gate.

| Context | Seed42 AUC | Native AUC | Seed42 AP | Native AP | Native−seed42 AUC / AP |
|---|---:|---:|---:|---:|---:|
| frag3-context1 | 0.870064 | 0.848321 | 0.298078 | 0.357457 | −0.021743 / +0.059380 |
| frag3-context2 | 0.699512 | 0.412296 | 0.182887 | 0.049710 | −0.287216 / −0.133177 |
| frag1-context1 | 0.446052 | 0.430881 | 0.185973 | 0.178302 | −0.015171 / −0.007672 |
| frag1-context2 | 0.903678 | 0.907858 | 0.500096 | 0.726190 | +0.004180 / +0.226094 |
| frag2-context1 | 0.501298 | 0.627642 | 0.269839 | 0.325355 | +0.126344 / +0.055517 |
| frag2-context2 | 0.779541 | 0.855403 | 0.581235 | 0.720561 | +0.075862 / +0.139326 |

Equal-context means were computed separately for the two physical parents:

| Parent | Seed42 AUC / AP | Native AUC / AP | Native−seed42 AUC / AP |
|---|---:|---:|---:|
| Paris1, two contexts | 0.784788 / 0.240483 | 0.630308 / 0.203584 | −0.154479 / −0.036899 |
| Paris2, four contexts | 0.657642 / 0.384286 | 0.705446 / 0.487602 | +0.047804 / +0.103316 |

The prespecified uint8 and author-decoded diagnostics are retained separately for **both** models, including AUC/AP, unique-score counts and decoded endpoint fractions. The full per-context diagnostic and paired probability differences are in `analysis-v1/metrics.json`; no diagnostic replaces the primary raw-score comparison. Across contexts, mean absolute probability differences range from0.073479 to0.167169. Quantization creates ties, and the author-decoded scores are not calibrated probabilities.

Strict loading verified508 FP32 named tensors,244 actual alias pairs,264 storage groups and34,546,498 unique parameters. All pre/post values and source hashes remained unchanged. The semantic state hash remained `83698399e8467f728e6cde2eed998603b82f0fa298f14f2533dcdbabe08b9b58`, matching the phase22-qualified artifact. The worker took14.154 seconds wall and50.601 seconds CPU, with838,111,232 bytes measured peak RSS. The external controller completed in15.447 seconds. Network and `torch.load` counters stayed zero; gradients and training steps were zero.

The frozen independent checker passed on its first attempt. It verified all150 input hashes and calls, all508 state hashes and244 aliases, unchanged source/input pins and the entire520,794-pixel primary support. Every raw/uint8/decoded metric, diagnostic array, endpoint fraction, paired statistic and parent mean matched within the fixed `1e-12` metric tolerance. Native sums, weights and raw scores reconstructed bitwise. Independent arithmetic errors remained within the frozen tolerances: sigmoid at most`5.960464477539063e-8`, Hann at most`1.7881393432617188e-7`, and seed42 blend at most`2.980232238769531e-7`. The checker made no model calls.

Key receipts:

- Native result SHA256: `c38ad03a287af2c20f365eb1a5edd63f40639b2a0c7dd61922a60abd1493ecf9`.
- Analysis metrics SHA256: `67dba0d21bded05a6166969f597053b89399430de3dee29f2fbd31357f2c65a5`.
- Independent verification: `../independent-review/verification-v1.json`, SHA256 `b318016263d852d1de7843a6b5dfe1540a2feb779d044ca500df64ed0396bd3e`.

This comparison does not establish a fresh holdout, teacher independence, a causal benefit from native pseudo-labels, historical CUDA reproduction or physical ink. No model promotion is encoded. Seed42 and all earlier evidence remain unchanged. No GPU was rented; no acquisition request was made.
