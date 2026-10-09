# Actual-backbone receipt review

**PASS.** The root's single actual-seed42 synthetic integration smoke matches the reviewed plan. This review read source and result text only; it performed no additional pretrained load, model forward, array read, gradient computation, network request or GPU operation.

`runtime-receipt-review-v1.json` records 78 checks covering the unchanged preexecution seal, all plan source records, command and plan hashes, controller-to-worker binding, observed calls, resource bounds, exact initial identity flags, first/second gradient sentinels and unchanged original state. The worker receipt is SHA-256 `698a7f804e6d7653fc1f9ced9ac866909b4fb1e8ad5315aba18a8b85766d7cc2`.

The controller completed in 5.100 seconds, with no retry. The worker completed in 4.355 seconds; worker peak RSS was 704,192,512 bytes and the controller sampled 704,802,816 bytes. Both are below 4 GiB. It loaded seed42 exactly once and observed one full-model forward, two original fusion forwards, five original 2D-network forwards, four adapter forwards, four backwards and two synthetic SGD steps. Full-model and component counts overlap and must not be added as distinct whole-model evaluations.

Both 5,296-parameter arms started with identical state hashes and exact feature/logit equality to the unchanged model. Their first projection gradients were finite and nonzero, their first hidden gradients were exactly zero, and their second hidden gradients were finite and nonzero after one projection update. All original parameters remained frozen, without gradients, and all modules remained in evaluation mode. Before/after state hashes match. This released model reported no named buffers; the separate independent toy included BatchNorm buffers to exercise preservation explicitly.

These checks establish consistency of retained receipts with reviewed code. They do not independently recompute actual-network tensor equality or gradients from raw tensors, and the smoke does not establish ink-detection efficacy. Synthetic loss changes must not be used to rank the feature arms. There were no scientific pixels or labels, saved weights, network calls, GPU calls or paid resources.

The original preexecution seal remains unchanged. `retention-manifest-v2.json` adds this runtime review and checker while binding that prior seal.
