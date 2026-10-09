# Conservative continuation: fixed adaptation risk

Both 32-update controls at learning rate `1e-6` lower the fixed adaptation base and full objectives relative to the cache-derived initializer. Both pass the predeclared computational gate requiring each final difference to be `<=−0.0001`. This permits the separately frozen retention check; it does not establish generalization, an optimal learning rate or a promoted model.

| Arm | Initial base/full | Final base | Final full | Base change | Full change |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_preserve | 0.597077174 | 0.586598447 | 0.590246351 | −0.010478728 | −0.006830823 |
| B_preserve | 0.745954789 | 0.732895620 | 0.737095126 | −0.013059169 | −0.008859663 |

![Actual fixed-batch risk changes](../visualization/render/fixed-risk-change.png)

Each plotted point after initialization evaluates the **same 32 frozen batch exposures on one current state**, after updates 1, 8, 16 and 32. Risk is `math.fsum` of the actual per-batch loss scalars divided by 32: base BCE remains FP32; the full objective adds FP64 preservation KL. The baseline at 0 is explicitly derived from the retained frozen-teacher logits, with no new initializer forward. Repeated and overlapping exposures keep their original multiplicity and separate manual/pseudo denominators. They are not independent samples.

Both base and full risks are below their respective initial values at every measured checkpoint and decline across these checkpoints. Raw values and deltas are preserved in [fixed-risk.csv](../visualization/render/fixed-risk.csv). The two arms optimize different weighted objectives, so their absolute risks or delta magnitudes do not establish an arm ranking.

On each update’s own identical batch, immediate full-loss increases occur on **0/32** updates for each arm. Immediate base-loss increases occur on **0/32** for A and **1/32** for B. These immediate measurements are distinct from the fixed-batch risk and from the old changing-model, changing-batch online trajectory. The older `1e-4` run did not retain final fixed-batch risk; no such old final value is inferred here.

The supplied summary records exact agreement between each checkpoint’s current-batch fixed-risk logits and its immediate-post logits. Training used the same credited seed42 initializer, two PHerc0139 regions, manual/pseudo masks, seed43 hard pseudo targets, smoothing, preservation objective, 32-row schedule, transforms and AdamW settings; only the learning rate changed. No threshold calibration or retention inference is part of this training report. Plinius remains a working research-series name.

Root training and analysis passed. Independent numerical verification also passed: 64 updates, 64 cache-derived baseline batches, 128 immediate pre/post loss records, 256 fixed-risk batch evaluations and 6,164 scalar comparisons were checked. The reviewer reconstructed logits/supports/losses/risks/gates; parameter, optimizer and RNG isolation remain worker-reported receipts rather than independently executed model checks. This report uses saved outputs only, without raw-array or checkpoint reads or model calls.

## Separate conditional retention result

The separately executed retention stage passed both predeclared ranking screens. On the two reused Frag4 contexts, mean AUC/AP rise slightly while sensitivity at the **unchanged** baseline cutoff falls. The ranking screen does not require sensitivity preservation, and passing it does not promote a model.

| State | Frag4 mean AUC | Mean AP | Sensitivity at cutoff 0.7558947205543518 |
| --- | ---: | ---: | ---: |
| Frozen seed42 | 0.734809 | 0.186118 | 3.025% |
| A_preserve | 0.740985 | 0.188433 | 0.354% |
| B_preserve | 0.743636 | 0.190900 | 0.183% |

![Frag4 ranking and operating response together](../visualization/retention-render/frag4-ranking-and-response.png)

These are equal-context means of supplied-reference agreement on one specimen, with prior diagnostic history and unresolved upstream exposure. No new threshold, independent accuracy or orientation choice is introduced. Full context/domain interpretation is root-owned; this compact view preserves the difference between ranking and fixed-cutoff response.

Input receipts: risk-summary SHA-256 `9b48a9f187d68dc9ebf57b88cef295547d553b5a5d0a1f181c218d9371ae7284`; retention-gate `0e07252f2705e7905979a7c29dc90a7717133783d334ac3cd269711cf5351b7d`; independent training verification `8ae053a9095b2d546f2401dcaa8779ed5d5a2cece954d5decaad3edb1b469bf8`; retention metrics `4f1d8df047dc034cfaed3ddce397191d152674921baca6c7e97acd1c692737c3`. Figure/CSV identities are in `../visualization/render/figure-receipt.json` and `../visualization/retention-render/figure-receipt.json`. The frozen implementation remains unchanged.
