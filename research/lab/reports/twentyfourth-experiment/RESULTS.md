# Paired depth-order test: completed and independently verified

Reversing the normalized depth axis worsened both average ROC-AUC and average precision (AP) for both fixed models on each physical parent. Seed42 declined on all six contexts. The native checkpoint declined on five and improved on one previously weak context. This establishes substantial order sensitivity under the controlled intervention; it does not establish an incorrect physical surface normal or provide a general correction for the earlier native-model failures.

**Decision:** retain the existing forward seed42 baseline and keep the native checkpoint as a contrasting research reference. Do not select a direction per region, blend forward/reverse outputs, impose an invariance loss or rent a GPU on this evidence. The next priority is a reference with an authenticated input/annotation/exposure contract, before further outcome-guided model tuning.

## What was tested

The frozen protocol held the six development contexts, 17 selected planes, values, normalization, tile schedule, labels and primary masks fixed. Each original patch went through the unchanged forward normalizer; its exact captured input hash was checked, then only the normalized tensor depth axis was reversed (16 through 0) before inference. The adapter still read forward. Original and reversed input bytes were verified before and after each call. This isolates order without changing the normalized values. Raw reversal followed by renormalization was not the execution path.

A prospective sparse/MAD-fallback toy showed why that distinction matters: FP32 reduction order can change normalization (maximum normalized difference 7.15e-7 in that toy). Actual selected patches happened to commute byte-for-byte, but the test used the explicit normalized permutation regardless. No post-result tolerance change was made.

Two fresh sequential CPU workers each constructed and strictly loaded one fixed checkpoint, then ran 150 full-model forwards: **300 forwards total, 31.421 seconds**, no warmup, scientific retry, gradients or training updates. Prior seed42 and native forward outputs were reused after verifying their input/state/score contracts. Runtime: Python 3.12.14, NumPy 2.5.3, Torch 2.12.0+cpu, safetensors 0.8.0; FP32 deterministic evaluation, four intra-op threads and one inter-op thread. External measured peak RSS was 706,297,856 bytes for seed42 and 837,148,672 bytes for native.

All 520,794 original primary pixels were retained: 86,535 supplied positives and 434,259 supplied negatives. Primary support remains the known mask intersected with the central 320×320 region of each 384×384 context. These are six repeatedly reused development contexts on **two physical parents**, not six independent confirmations.

## Parent results

Equal weight per context within each named parent. Paris1 contains two contexts and Paris2 four. These descriptive means have no pooled-pixel significance test or performance pass gate.

| Parent | Model | Forward AUC | Reverse AUC | Change | Forward AP | Reverse AP | Change |
|---|---|---:|---:|---:|---:|---:|---:|
| Paris1 | seed42 | 0.784788 | 0.489187 | -0.295601 | 0.240483 | 0.088780 | -0.151703 |
| Paris1 | native | 0.630308 | 0.540326 | -0.089983 | 0.203584 | 0.084710 | -0.118874 |
| Paris2 | seed42 | 0.657642 | 0.477125 | -0.180517 | 0.384286 | 0.237717 | -0.146569 |
| Paris2 | native | 0.705446 | 0.564964 | -0.140481 | 0.487602 | 0.294696 | -0.192906 |

Difference of changes is (native reverse−forward) − (seed42 reverse−forward), not a causal estimate of the native training method:

| Parent | Difference of AUC changes | Difference of AP changes |
|---|---:|---:|
| Paris1 | +0.205618 | +0.032829 |
| Paris2 | +0.040036 | -0.046337 |

Native loses less AUC on both parents, but loses more AP on Paris2. That metric-dependent contrast does not establish that one model is universally more robust.

## Every context

| Context | Model | Forward AUC | Reverse AUC | Change | Forward AP | Reverse AP | Change |
|---|---|---:|---:|---:|---:|---:|---:|
| frag3-context1 | seed42 | 0.870064 | 0.453746 | -0.416318 | 0.298078 | 0.111603 | -0.186475 |
| frag3-context1 | native | 0.848321 | 0.526524 | -0.321797 | 0.357457 | 0.100865 | -0.256592 |
| frag3-context2 | seed42 | 0.699512 | 0.524628 | -0.174884 | 0.182887 | 0.065957 | -0.116931 |
| frag3-context2 | native | 0.412296 | 0.554127 | +0.141831 | 0.049710 | 0.068554 | +0.018844 |
| frag1-context1 | seed42 | 0.446052 | 0.231062 | -0.214990 | 0.185973 | 0.137537 | -0.048436 |
| frag1-context1 | native | 0.430881 | 0.296306 | -0.134575 | 0.178302 | 0.148483 | -0.029818 |
| frag1-context2 | seed42 | 0.903678 | 0.574188 | -0.329490 | 0.500096 | 0.201475 | -0.298621 |
| frag1-context2 | native | 0.907858 | 0.657498 | -0.250360 | 0.726190 | 0.290888 | -0.435302 |
| frag2-context1 | seed42 | 0.501298 | 0.394231 | -0.107067 | 0.269839 | 0.197189 | -0.072650 |
| frag2-context1 | native | 0.627642 | 0.539213 | -0.088429 | 0.325355 | 0.238755 | -0.086600 |
| frag2-context2 | seed42 | 0.779541 | 0.709019 | -0.070522 | 0.581235 | 0.414667 | -0.166569 |
| frag2-context2 | native | 0.855403 | 0.766840 | -0.088562 | 0.720561 | 0.500657 | -0.219904 |

The exception is native on frag3-context2: AUC rises from 0.412296 to 0.554127 and AP from 0.049710 to 0.068554. Both remain below forward seed42 on that same context (0.699512 / 0.182887). Meanwhile native AP on frag1-context2 falls from 0.726190 to 0.290888. A retrospective direction choice would hide these failures and reuse evaluation information.

## Raw score drift

Reverse minus forward on the unchanged primary support. These sigmoid-derived model scores have not been established as calibrated probabilities. The 95th percentile uses the frozen linear interpolation rule.

| Context | Model | Mean signed | Mean absolute | 95th percentile absolute | Maximum absolute |
|---|---|---:|---:|---:|---:|
| frag3-context1 | seed42 | -0.056723 | 0.153341 | 0.445001 | 0.585137 |
| frag3-context1 | native | -0.132154 | 0.154055 | 0.329228 | 0.420865 |
| frag3-context2 | seed42 | -0.033773 | 0.094140 | 0.373261 | 0.517198 |
| frag3-context2 | native | -0.075595 | 0.111800 | 0.312990 | 0.443251 |
| frag1-context1 | seed42 | -0.023340 | 0.136732 | 0.442924 | 0.558659 |
| frag1-context1 | native | -0.052145 | 0.108375 | 0.273826 | 0.431717 |
| frag1-context2 | seed42 | +0.033616 | 0.142690 | 0.416074 | 0.543663 |
| frag1-context2 | native | +0.041825 | 0.136770 | 0.376189 | 0.476907 |
| frag2-context1 | seed42 | +0.029475 | 0.158859 | 0.436082 | 0.534941 |
| frag2-context1 | native | +0.095190 | 0.140418 | 0.421433 | 0.515879 |
| frag2-context2 | seed42 | -0.089239 | 0.101998 | 0.412035 | 0.548697 |
| frag2-context2 | native | -0.222562 | 0.225311 | 0.472183 | 0.538609 |

## Independent verification and visual review

The separately frozen checker passed on its first actual-output verification without executing either model. It checked 300 input pairs (150 unique pairs shared across models), 1,016 state tensor hashes, 488 expected alias pairs, exact schedules and original support. All raw metrics, score drifts, per-model changes and differences of changes matched within 1e-12. Reverse sums, weights and raw mosaics reconstructed bit-for-bit; independent sigmoid arithmetic differed by at most 5.96e-8, Hann weights by 1.79e-7 and retained forward-control reconstruction by 2.98e-7, within the prospective FP32 tolerance. The technical PASS is not a performance success gate.

The root reviewer viewed all four fixed-scale whole-context figures (all six contexts per model). Reversal frequently removes responses along supplied strokes or increases broad background responses. The native frag3-context2 exception includes reduced upper background response rather than a convincing general recovery of its lower strokes. Visual impressions do not identify physical normals or authorize letter interpretation.

- [seed42-paris1](visuals/render-v1/seed42-paris1-depth-reversal.png)
- [native-paris1](visuals/render-v1/native-paris1-depth-reversal.png)
- [seed42-paris2](visuals/render-v1/seed42-paris2-depth-reversal.png)
- [native-paris2](visuals/render-v1/native-paris2-depth-reversal.png)

[Root visual observations](ROOT_VISUAL_REVIEW.json), [independent verification](independent-review/verification-v1.json), [complete metrics](implementation/analysis-v1/metrics.json), [technical report](implementation/RESULTS.md) and [frozen protocol](PROTOCOL.md) preserve the underlying evidence.

## Implications and limits

We can reject the narrow practical proposal that uniformly reversing these inputs repairs the native checkpoint comparison: both parent summaries worsen. We cannot reject every possible physical orientation issue, infer the correct ink-bearing side, attribute checkpoint differences to one training technique, or claim new-parent accuracy. The intervention is a computational counterfactual on already selected normalized inputs, not a physically re-rendered surface. No p-values or confidence intervals are inferred from correlated pixels.

No new model, transcription or translation is claimed. PHerc0009B, PHerc0500P2, PHerc0841 and Frag6 remain unscored; Frag6 Stage C stays closed. Published model configuration omissions do not settle teacher ancestry. More tuning on these six regions would not cure that evidence gap. [The next-study recommendation](NEXT_STUDY_DRAFT.md) specifies the missing source and evaluation contracts and a prospective fixed-reference comparison.

GPU spend is **£0**, with the **£10 total first-pilot cap** unchanged. There were no scientific network requests, candidate acquisitions, credential operations, R2 uploads, outreach, publication or prize submission in this phase. All earlier sealed evidence and the checkout at 49fabf3e890d5b880f391b27d581f1e7a1ebdbcc are preserved. Reusable startup guidance and local integrity retention are recorded separately; saving a configuration draft is not an external backup, publication or verified fresh-task restoration.
