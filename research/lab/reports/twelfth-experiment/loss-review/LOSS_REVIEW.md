# Independent loss mathematics and source audit

**No sign, support-normalization, target-smoothing or update-order defect was found in the retained ninth/tenth implementations.** The tenth report's `original_objective` is already the smoothed BCE term. A hard-label versus smoothed-label mismatch therefore does not explain away its reported worsening against the initializer. This is a source and synthetic-mathematics review; the objective-audit package separately verifies actual stored trajectories.

## Exact objectives and output gradients

Let `z` be a student logit, `p=σ(z)`, `t` a detached initializer logit on the same normalized/augmented input, and `q=σ(t)`. Hard participant labels are `y∈{0,1}`. With epsilon 0.5 the target is `s=0.25+0.5y`, hence 0.25 or 0.75. Set `A(z)=log(1+exp(z))` and `B(z,s)=A(z)−sz`.

Manual support `M` and pseudo support `P` are disjoint. Each mean divides by the number of participating pixels **across the entire two-item batch**, not by full tensor size, two regions, or four class groups. Write those counts as `N_M,N_P` and use `c=0` for A, `c=0.25` for B.

`O(z)=mean_M B(z,s_M)+c mean_P B(z,s_P)`

`D(z,t)=A(z)−A(t)−q(z−t)=KL(Bernoulli(q) || Bernoulli(p))`

`K(z,t)=mean_M D(z,t)+c mean_P D(z,t)`

The ninth optimized `O`; the tenth optimized `F=O+K` with lambda 1. For a manual output, `∂O/∂z=(p−s)/N_M`, `∂K/∂z=(p−q)/N_M`, and `∂F/∂z=(2p−s−q)/N_M`. For a pseudo output replace the denominator by `N_P` and multiply all three derivatives by `c`. Every output outside the active supports has zero direct loss derivative. Shared parameters and spatial/instance normalization can nevertheless change predictions outside those supports.

The factor 0.25 weights the **pseudo mean**, not each pseudo pixel relative to a manual pixel irrespective of support size. Its per-pixel coefficient ratio is `0.25 N_M/N_P`. Neither A nor B is class-balanced. The separately logged `R` is the equal mean of four region-by-hard-class means; its member derivative would be `(p−s)/(4N_region,class)`, but **R is diagnostic only** in these arms.

Pseudo hard labels and preservation probabilities are different objects. The former threshold a retained teacher-score map at 0.5. The latter come from initializer logits on the exact current patch. They must not be substituted for one another or assumed numerically identical.

## Exact change decomposition

For one fixed input/support, write `Δz=z−t` and `L=mean_M[(q−s_M)Δz]+c mean_P[(q−s_P)Δz]`. Then:

`O(z)−O(t)=K+L`, and `F(z)−F(t)=2K+L`.

The initializer has `K(t,t)=0`; its full and base objectives coincide. Consequently a positive same-input smoothed-base delta plus nonnegative KL implies a positive full-objective delta as well, up to the recorded numerical tolerance. The preservation penalty cannot reverse that arithmetic.

For general nonnegative lambda, the per-output effective target is `r=(s+lambda*q)/(1+lambda)` and `B(z,s)+lambda D(z,t)=(1+lambda)B(z,r)−lambda H(q)`. At lambda 1, the unconstrained logit optimum has probability `(s+q)/2`. This is a **free-output oracle bound**, not a claim that one shared network can realize independently chosen optima across changing patches. At the initializer, preservation loss and its output gradient are exactly zero; the tenth first gradient equals the corresponding ninth smoothed gradient.

Hard BCE can rise while smoothing improves: a positive label predicted with probability 0.95 is above its smoothed optimum 0.75, so moving it slightly downward increases hard BCE while decreasing both smoothed BCE and, for a sufficiently small initial step, the full objective. That valid example is distinct from a positive delta in the already-smoothed `original_objective` field.

## Implementation and retained-observation checks

- [smoothing.py](../../ninth-experiment/implementation/smoothing.py) applies `epsilon/2+(1−epsilon)y` only after masked selection. Hard masks/class identities remain unchanged. Its hard-loss call also validates participant labels and nonempty class groups.
- [training_mechanics.py](../../eighth-experiment/implementation/training_mechanics.py:26) uses PyTorch BCE's default mean on Boolean-selected participants. Lines 48–51 supply the A/B coefficients; balanced R is not in either objective.
- [tenth run_worker.py](../../tenth-experiment/implementation/run_worker.py:87) forms smoothed `U/P`, calls the forward KL helper, and adds `D_M` or `D_M+0.25D_P`. `original_objective` is that smoothed `U` or `U+0.25P`, not hard BCE.
- [preservation.py](../../tenth-experiment/implementation/preservation.py) detaches the teacher, promotes finite FP32 logits to FP64 for stable `logaddexp`, preserves the unclamped value, and rejects mean KL below −1e−6. Its derivative direction is `p−q`. Teacher/student shape, mask disjointness and finite values are checked.
- [analyze_control.py](../../tenth-experiment/implementation/analyze_control.py:62) compares those smoothed student terms with smoothed initializer terms on the same scheduled input/support. Its last-eight summary means steps 25 through 32, each observed **before** that step's update. These are neither post-update measurements nor final-model empirical risk.
- Both workers clear gradients, compute loss, backpropagate once, clip the global parameter-gradient norm to 1, then call fresh AdamW (`lr=1e−4`, betas 0.9/0.999, epsilon 1e−8, weight decay 0). The tenth checks frozen teacher state, disjoint storage, RNG stability and train/eval-compatible modules. The initial gradient sentinel and parameter-update probe are consistent with that order.

Tenth raw arrays are FP32 `[32,2,1,128,128]`, indexed by `step−1`; logits are saved before backward/update. The two axes after step identify scheduled batch member and singleton output channel. The same step's augmented masks/targets and input fingerprint are required for reconstruction. Ninth retains scalar logs but no equivalent all-step raw logits. Neither package provides a same-first-batch post-update logit record or a step-32 post-update observation.

Changing minibatches, finite shared-parameter updates and Adam's adaptive history do not guarantee monotonic online loss, despite correct formulas. A positive late online delta is therefore evidence about that recorded optimization trajectory, not by itself an implementation bug, proof of final-state degradation on the adaptation population, or an unavoidable mathematical cost of preservation.

## Narrow first-step probe assessment, not authorization to execute

A later minimal probe could reproduce the exact frozen first batch, initializer, normalization/augmentation, fresh optimizer and gradient clipping; verify the existing input/logit/loss/gradient sentinels; and retain same-batch pre/post-update logits. Detached initial logits already supply the initializer teacher values for that single batch, so an additional teacher forward is mathematically unnecessary. Score the same fixed smoothed base, forward KL and full objective before/after; introduce no new region, calibration or model-selection metric.

With fresh zero Adam moments and zero weight decay, the ideal first parameter update is `d_i=−eta*g_clipped_i/(abs(g_clipped_i)+1e−8)`. Global clipping multiplies the original gradient by a positive common factor, so `g·d<0` whenever the gradient is nonzero. Preservation has zero first gradient, hence both base and full objectives share this initial descent direction. Sufficiently small steps descend locally in real arithmetic; the historical finite learning rate need not. FP32 rounding and the epsilon term must be preserved. Any comparison of learning-rate scaling must restart from the identical initial parameter/optimizer state, and remain a predeclared minimal diagnostic rather than an open learning-rate search. No such probe was executed in this review.

## Synthetic evidence and scope

[check_numpy_math.py](check_numpy_math.py) passed **13 NumPy-only cases**: BCE/KL finite differences; forward-KL orientation; A/B masked gradients and unknown-output invariance; unequal batch/class normalization; zero initial KL/gradient; weighted change identities; effective-target identities; hard-versus-smoothed counterexample; positive-base/full implication; fresh clipped AdamW direction; extreme-logit finiteness; and overlap rejection. Fixtures do not call the retained PyTorch loss functions and are not a new autograd execution test.

Exact source hashes and schema metadata are bound in `SOURCE_RECEIPTS.json`; [numpy-fixtures.json](numpy-fixtures.json) binds the executed synthetic checker. No network, real logit-array reads, checkpoint loads, Torch imports, or actual tensor/model forward/backward calls were made for this loss audit. Earlier evidence remains unchanged.
