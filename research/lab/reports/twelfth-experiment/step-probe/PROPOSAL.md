# Conditional first-step descent probe

This package fills the missing same-batch pre/post-update measurement in the retained ninth/tenth studies. It remains conditional on root's completed offline audits and a root-ready frozen plan. Preparation has read text, JSON metadata and source only; no real array, checkpoint, model, forward/backward or fixture execution has occurred.

Use the unchanged strict sixth model helper and seed42 initializer, matching semantic state SHA256 `21c186000b9b3093835e185342668acd0d86147a719c37513e937924b77ea228`. Retain the eighth/ninth/tenth adaptation contexts, manual participation excluding historical V, seed43 hard-pseudo support,17-plane robust-MAD input normalizer and exact historical step1 schedule/augmentations. The normalized batch/masks/hard-target fingerprint must be `571d10c58008cb6f9e581a717bbb067846dae6a2ee2d29bdf9eb141ca9a02ec3`. Saved tenth first student and teacher logits must match the new initial logits **bitwise**, not merely approximately.

Two objective families are fixed: A=`U_s`; B=`U_s+0.25P_s`, with epsilon0.5 targets`0.25+0.5y` and the original separate masked participant means. Each worker computes one initial forward and one backward. Unknown output pixels must have exactly zero gradient. The initial smoothed U/P, secondary hard U/P and global gradient norm must reproduce the historical sentinels before any update. Actual clipping is PyTorch global norm1, including its denominator epsilon1e-6.

The clipped gradient is reused only at the **identical** initializer. Before each of learning rates1e-6,1e-5,1e-4, restore all unique parameters in-place; then verify the semantic hash across all508 state keys, and verify the unchanged clipped-gradient hash. Create a new AdamW with empty state, betas0.9/0.999, optimizer epsilon1e-8, weight decay0 and no scheduler. Take exactly one update, then evaluate the same batch once without gradients. No parameter state, optimizer moment or gradient from an updated model carries into another branch. The shared initial gradient is mathematically valid because every branch has the same initialization, input, objective, deterministic modules and clipping.

The detached initial FP32 logits supply the initializer teacher on this batch, without constructing or forwarding a separate teacher. Measure original smoothed base BCE, KL(initial teacher||post student) and base+KL, plus clearly secondary hard-target BCE. Forward KL is evaluated in float64 from actual FP32 logits, temperature1, on M for A and meanM+0.25meanP for B. At initialization the preservation value and gradient are zero, so this first update is shared by the matched smoothing/preservation objectives. Train/eval mode guards reject mutable/running buffers, active dropout/stochastic depth and output activations; RNG must remain unchanged by each forward.

For post logit displacement`dz`, initializer probability`q` and smoothed target`s`, retain the stable per-support identities

`delta_base = KL(q||p_post) + mean[(q-s)*dz]`

`delta_full = 2*KL(q||p_post) + mean[(q-s)*dz]`.

Do not hide small values by clamping, round loss changes to FP32 or classify a tiny non-improvement as an optimizer sign bug. Save both the actual FP32 BCE values and independently arranged float64 change terms. Decomposition tolerance is absolute1e-10; participating KL means may be no lower than−1e-6 under the inherited roundoff guard.

The ideal first fresh AdamW update at clipped gradient`g_c` is`−lr*g_c/(abs(g_c)+1e-8)`. Compare this against the actual parameter difference computed in float64 from FP32 before/after values. The predeclared absolute per-coordinate allowance is`4*ULP(initial_fp32)+8*eps_fp32*abs(ideal_delta)+1e-12`, where ULP is positive absolute spacing. Record absolute errors and error/allowance ratios; do not use relative error on near-zero updates. Require optimizer step1 for every participating parameter, unchanged no-gradient parameters and no coordinates moving opposite the gradient. A positive aggregate`g_c dot actual_delta` is an error; unchanged coordinates may legitimately round to zero in FP32. A negative first-order directional product does not guarantee that every finite learning rate decreases the measured loss.

Retain the normalized input, exact participant masks/hard targets, initial and three post logits, initialization/input/gradient hashes, pre/post losses, stable decomposition and per-parameter gradient/update summaries. Parameters and optimizer moments are ephemeral; no checkpoint or optimizer artifact is saved. Independent reviewers can reconstruct numerical loss changes from saved logits and masks; full raw gradients/weights are not retained, so parameter-update evidence consists of the reviewed executable checks and summaries.

The complete conditional scope is **two workers,8 model forwards,2 backwards and6 independent optimizer steps**. CPU only, four intra-op/one inter-op threads, deterministic algorithms, seed20261008, worker wall120s, total wall300s, CPU360s/worker, RSS acceptance3GiB, address-space cap12GiB and output cap64MiB. Root owns execution and the total wall/output checks. There is no new calibration, validation, Frag4 inference, sequence of updates, open learning-rate search, chosen winner, paid resource or network request.

The result can show immediate descent or finite-step overshoot on this single historical batch. It cannot identify the cause of all32-step transfer results, minibatch interference, internal representation damage or a generalizing learning rate. Root owns descriptive interpretation; the probe does not freeze a new training recipe.

Operational interface:

```sh
# Metadata/source-only pending proposal. This does not load binary inputs.
python prepare_probe_plan.py --output-plan proposal.pending.json

# Root only, after offline audits and technical review; preserve the pending file.
python prepare_probe_plan.py --ready --review TECHNICAL_REVIEW.json \
  --offline-audit OFFLINE_AUDIT.json --output-plan plan.ready.json

# Root only after final SHA/source freeze; run A then B once, no retries.
python probe_worker.py --plan plan.ready.json --plan-sha256 FROZEN_SHA256 \
  --objective A --output-dir FRESH_A_DIR
python probe_worker.py --plan plan.ready.json --plan-sha256 FROZEN_SHA256 \
  --objective B --output-dir FRESH_B_DIR
```

The ready plan pins root's`twelfth-experiment/DIAGNOSTIC_DESIGN.md`, historical plans/sentinels and all inputs actually used. Exact binary input hashes are verified by the guarded worker immediately before and after the authorized run, not falsely claimed as rechecked during text-only preparation. The worker rejects a pending plan and stops on a matching/finite/resource/identity failure; it does not retry or modify prior evidence.
