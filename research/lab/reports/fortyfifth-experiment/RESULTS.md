# Plinius45: closing the native run and returning to the scroll

The approved native profilometry trajectory completed 8,192 updates. The source and target audits support redirecting the programme to a PHerc343 positive-control reproduction and then PHerc0800. Further profilometry studies and the unsent Paris4 reference request are shelved. A CPU AP gain is no longer a universal prerequisite for a properly designed GPU pilot; the £10 total cap and concrete launch controls remain.

## Native continuation

The first 2,048 updates reproduced every one of 42 original model tensors exactly. Training then continued for 6,144 additional updates without resetting AdamW or either random generator. Model, optimizer moments/counts and RNG state were retained and read back at 2,048, 4,096 and 8,192. Architecture, inputs, weighted BCE, sampling and learning rate were unchanged. This was one fold, training PHerc250/500P2; the PHerc248 development scores did not choose its endpoint.

| Updates | PHerc250 training AP | PHerc500P2 training AP |
|---:|---:|---:|
|512|0.168531|0.357926|
|1024|0.155218|0.468211|
|2048|0.372165|0.527950|
|4096|0.370534|0.515441|
|8192|0.551912|0.590452|

The 0.80 AP fit criterion failed, while both training APs still improved by more than 0.01 in the last interval. Optimization budget remains inconclusive for this setup. The result does not establish a lack of recoverable information; the strategic decision is to stop this supporting branch and return to the target.

Final-interval gains: PHerc250: +0.181378 AP; PHerc500P2: +0.075011 AP. The original positive-centre oversampling plus class weighting remains a probability-calibration caveat; AP is the fixed primary fit metric.

PHerc248 was already exposed development data. Its fixed final scores are native: AP 0.033097, AUC 0.521854; coarse4: AP 0.033760, AUC 0.529607. No untouched-test or CT-ink result is claimed.

On that reserved development papyrus, native-input AP changed from 0.041889 at 2,048 updates to 0.033097 at 8,192 (-0.008791). The greater training fit did not transfer in this fold. This supports closing the branch while keeping the distinction between underfitting and transfer failure explicit.

## Released-checkpoint variability

Four fixed released checkpoints—seeds 42 and 43 at 60k and 75k—were evaluated on all eight reused regions. The seed42/75k repeat exactly reproduced all eight archived baseline arrays. There were 800 new single-tile forwards and no fitting or checkpoint selection. One initial role-label validation failure occurred before construction/forward; the additive correction is preserved.

| Region | AP: seed43 minus seed42 at75k |
|---|---:|
|frag3-context1|+0.194971|
|frag3-context2|-0.058220|
|frag1-context1|+0.025113|
|frag1-context2|+0.217758|
|frag2-context1|-0.020411|
|frag2-context2|+0.105766|
|pherc0814-46527-fresh-rank01|-0.000129|
|pherc0814-46527-fresh-rank14|+0.001898|

The equal-parent Paris difference is **+0.075216 AP**; the maximum absolute regional difference is **0.217758 AP**, with all six Paris regions exceeding 0.01 in absolute AP. However, mean Paris AUC changes by **−0.038628**, including a **−0.235352** regional AUC regression on frag3-context2. Late-checkpoint Paris changes (75k−60k) are seed42: +0.025531 AP, seed43: +0.026434 AP.

These are descriptive released-checkpoint differences, not a calibrated noise floor. Two seeds and correlated temporal endpoints do not establish a reliable variance, statistical power or a twice-noise significance gate. The practical 0.01 margin and uncertainty must remain separate. The old Paris panel has only two physical parents and remains development/regression evidence. Earlier matched, deterministic arm differences remain real conditional effects for those exact trajectories; this new comparison does not relabel them as random numerical noise. It shows why general conclusions need broader data and paired training replications. No model is promoted from this comparison.

## Target geometry and new sources

A fixed sparse lattice measured nearest-run disagreement with the public M7 prediction on three previously used PHerc0800 meshes. Median absolute offsets were 0.5, 1.0 and 1.0 native voxels. Twenty of 102 measurable sites exceeded two voxels, two other sites tied between nearest sheets, and 57 of 104 usable profiles contained multiple predicted runs. None of the three medians exceeds the old engineering threshold. This argues against blanket recentering; local outliers and wrap ambiguity remain worth testing. M7 agreement is not an independent measurement of true surface placement. See [the full geometry report](GEOMETRY_RESULTS.md).

PHerc343 uses the target acquisition settings 8.64 µm / 116 keV and supplies a concrete 66-layer, ±15-depth, four-model recipe including Hecate. Its mesh concatenation reproduced all three coordinate files and metadata byte for byte; its published image hash was verified. CT rendering and ink maps have **not** been reproduced here. Hecate needs 9.6 µm sampling in all three axes and already implements paired fine-to-native supervision; that changes our prior-art baseline.

Eleven live metadata reads confirmed PHerc0139 fine/native paired renders and the author benchmark/blank-control assets. The references are model-generated and local registration still requires checking, but this is a direct supporting route that need not wait on the Paris4 request.

[Critical review and primary sources](STRATEGY_REVIEW.md) · [One next experiment and its stopping criteria](NEXT_STUDY_DRAFT.md) · [Overview](overview.png) · [Geometry lattice](geometry-offsets.png)

## Verification and persistence

Independent retained-output checks reproduced 80 ranking values (maximum difference 2.22e-16), reconstructed all eight panel supports and blended 4,718,592 pixels from 800 raw tiles (maximum difference 3.81e-07). Optimizer steps/moments and all three saved continuation states were checked. Separately, all 104 M7 profiles were reconstructed from 71 observed chunks and their run classifications independently reproduced. These checks establish computational fidelity, not physical ink truth or experimental power.

The native worker made 8,192 training minibatch forwards/backwards plus 216 evaluation minibatch forwards; 2,048 of its updates replay prior work. The checkpoint panel made 800 single-tile forwards. The actual CPU quota is four cores; simultaneous Torch workers caused severe contention, so the panel was paused with state intact and resumed after native training. No scientific trajectory was restarted to improve an outcome.

GPU spend: **£0**. No paid resource, R2 upload, maintainer message, new reading or translation. Selected code/reports/metadata/figures are exported to GitHub; bulk arrays, chunks, maps and checkpoints remain in the cloud lab. See [portability and exact limits](PORTABILITY.md).
