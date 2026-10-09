# First-step probe results

The historical learning rate **1e-4 overshot on the exact historical first batch** for both objective families: one fresh AdamW update increased the smoothed base and complete preservation objective. At 1e-6 both decreased; at 1e-5 the base decreased while the complete objective increased. These are same-batch first-step observations, not selection of a new learning rate or an explanation of the full 32-step runs.

A uses smoothed manual BCE; B adds 0.25 times the original smoothed pseudo BCE. Complete objective adds KL(initial teacher||post student) on the same participating masks, with the same 0.25 pseudo coefficient for B. Initial detached logits supply the teacher; unknown pixels never participate. Values below are stable float64 changes reconstructed by the executed worker from its actual FP32 logits; negative means descent.

| Objective | Learning rate | Base loss change | Full loss change | Weighted KL |
|---|---:|---:|---:|---:|
| A | 1e-06 | -0.006745 | -0.005870 | 0.000874 |
| A | 1e-05 | -0.018163 | +0.024002 | 0.042166 |
| A | 1e-04 | +0.014596 | +0.118165 | 0.103569 |
| B | 1e-06 | -0.007344 | -0.006496 | 0.000849 |
| B | 1e-05 | -0.021086 | +0.026476 | 0.047563 |
| B | 1e-04 | +0.020080 | +0.145220 | 0.125140 |

The signed linear term was negative in every branch. At 1e-4 the weighted KL was 0.103569 for A and 0.125140 for B, outweighing linear terms −0.088973and −0.105061. The identities`delta_base=KL+linear` and`delta_full=2KL+linear` explain the measured positive changes. At 1e-5 the KL was large enough to make the full objective increase while the base still decreased. Clearly secondary hard-target loss changes, exact numeric values and update summaries are in [first-step-results.csv](first-step-results.csv).

Matching and optimizer checks passed. Initial weights had semantic SHA256 `21c186000b9b3093835e185342668acd0d86147a719c37513e937924b77ea228`; normalized input/masks/hard targets matched fingerprint `571d10c58008cb6f9e581a717bbb067846dae6a2ee2d29bdf9eb141ca9a02ec3`. Initial logits matched archived tenth student/teacher values bitwise. Every branch reset all state keys, reused the unchanged clipped gradient and started a fresh empty optimizer at step 1. Global preclip gradient norms matched history: A 2.3968579769and B 2.5406050682.

The reviewed runtime first-Adam checks passed across all coordinates. Gradient-dot-actual-update was negative in all six branches and wrong-sign coordinate counts were 0. The largest absolute ideal-versus-FP32 update discrepancy was4.63246788e-08; every coordinate stayed inside the predeclared dtype-aware allowance, with maximum error/allowance ratio0.325919. Between 233 and 237 coordinates rounded to unchanged values; this was retained as FP32 quantization rather than a relative-error failure. The actual clipped gradients' float64 norms were 1.000005283 and 1.000005209, retaining the small difference introduced by PyTorch's FP32 norm reduction and clipping target 1.

Root alone executed **8 forwards,2 backwards and6 independent first updates**, with no retry, network call, checkpoint output, calibration or extra evaluation. Combined wall time was16.083s; highest worker peak RSS was1.136GiB, below 3 GiB. The controller recorded7,841,868output bytes before its small receipt, below the combined 64 MiB cap.

This supports finite-step overshoot at 1e-4 on this particular initializer, objective and batch, with a correct first-order descent direction. It does not demonstrate why later minibatches or all 32 updates transferred poorly, identify representation damage, establish a globally safe learning rate, or authorize a replacement training recipe. Existing specimen/teacher lineage limits remain unchanged.

This report reads **saved JSON results and parameter summaries only**, verifies their receipt hashes and loss-change identities, and does not rerun models or read raw arrays/checkpoints. Numerical raw-logit reconstruction belongs to the separate independent audit. The root execution receipt is [run/run-results.json](run/run-results.json), SHA256 `afae91cf22d96ab305e6f74505a9fc0848fd33bfc482c6e0a6494bacc5f2ee4c`; ready plan SHA256 `d57d1a9d17bc11d8dd6630de58deb73338674e91530dc522ff28184a7afc599d`. Frozen source/preparation/controller hashes are in [code-freeze.json](code-freeze.json).
