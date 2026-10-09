# Independent numerical review: phase 20 fusion adaptation

**PASS for execution consistency and published arithmetic; FAIL for the predeclared scientific improvement screen.** Both scientific training arms and both final development arms were reviewed on the first frozen checker attempt. No checker failure, numerical tolerance change, source repair after outputs, training replay or additional model execution occurred in this review.

The explicit-depth arm beats the matched raw-input control on the five training/Paris2 improvement components, but fails five of the twelve baseline-retention components. This is a relative benefit inside an adaptation recipe that damages several reused development regions. It is not evidence that the adapted model improves upon the original seed42 reference overall.

## Verified training evidence

The independent implementation uses NumPy and the retained arrays, without importing the producer numerical code or Torch. It reconstructed all 32 PHerc0139 batches, D4 transforms, robust normalization, manual support and hard targets, raw/depth feature bundles and their hashes. Historical pseudo arrays were used only to authenticate old complete-batch fingerprints; they did not enter the objective or derivative.

For both arms, all 32 saved actual original-model outputs and all 32 initial zero-adapter outputs are bitwise identical to the historical cached teacher on the exact normalized inputs. The original fused-feature cache receipts agree across arms. Student-minus-teacher diagnostic arrays match their declared arithmetic exactly; they are not treated as an output-residual architecture.

The checker accepted 2,812 scalar loss/summary comparisons, 1,355,776 optimizer scalar comparisons and 985,056 exact adjacent parameter/moment scalars. Every one of the 5,296 ordered adapter parameters is represented at every update. Initial tensors match across arms, initial first/second Adam moments and projection parameters are exactly zero, and saved trace vectors reconstruct the reported initial/final semantic state hashes. The first step has zero stem gradients and nonzero projection gradients; the next step activates stem gradients. All original-model state/mode/gradient receipts remain unchanged, with no original parameter update.

| Numerical comparison | Maximum observed error | Frozen bound |
| --- | ---: | ---: |
| Smoothed BCE or full loss | 9.8021e-8 | 2e-7 |
| Hard BCE | 5.4236e-8 | 2e-7 |
| FP64 KL | 3.4694e-18 | 1e-12 |
| Manual probability drift | 6.9389e-18 | 1e-12 |
| Manual output derivative | 1.0382e-11 | 2e-10 |
| Reported FP64 gradient/update norms | 1.1103e-16 | 1e-12 |
| FP32 global gradient norm | 1.1921e-7 | 1e-6 + 2e-7 × expected magnitude |
| Clipping coefficient | 1.1921e-7 | 2e-7 |
| AdamW parameter update | 7.4506e-9 | 2e-8 + 8 corresponding FP32 ULPs |

The raw control clipped once and the explicit-depth arm did not clip. Both clipping branches were independently exercised by pre-output synthetic fixtures. All 64 manual output-gradient arrays have exactly zero unsupported-output derivatives. The independently reconstructed output derivative is `[2 sigmoid(student) − sigmoid(teacher) − (0.25 + 0.5 target)] / manual_count`.

All immediate post-update and fixed-risk values at updates 1, 8, 16 and 32 were retained, including unfavorable values. The current batch at each fixed measurement is bitwise identical to its immediate post-update output. Fixed risk is an equal mean of the 32 scheduled batch risks, including overlap; it is not a unique-pixel estimate.

The independently reconstructed final full risks are 0.5940976294035865 for raw control and 0.5938324328028228 for explicit depth, versus the initial 0.5970771768957628. The depth arm's advantage over raw is about 0.00026520, clearing the frozen 0.0001 training margin. These FP64 reference reconstructions differ slightly from the producer's FP32 BCE values within the predeclared tolerance; report comparison separately validates the producer's exact declared scalar arithmetic.

## Verified development evidence

Both final adapters were reviewed on all six fixed contexts, with 25 tiles per context. The reviewer independently reconstructed normalized tile input/feature hashes, sigmoid probabilities, Hann weighting, placement, coverage, masks and label participation. All 300 new tile outputs and the retained baseline maps were checked. Maximum sigmoid/blend disagreement was 2.980232238769531e-7, below the frozen 2e-6 bound. Diagnostic logit differences are exact saved student minus cached baseline values.

All 18 context-by-model AUROC, AP and fixed-cutoff summaries were independently calculated. The original cutoff remains 0.7558947205543518, without fitting. Root metrics match the independent per-context ranks, confusion counts, equal-context/equal-fragment summaries and all 17 screen components. Both numerical reconstructions give the same pass/fail decisions, with no gate-margin ambiguity.

| Paris2 equal-fragment mean | AUROC | AP |
| --- | ---: | ---: |
| Unchanged seed42 | 0.6576421878 | 0.3842856729 |
| Raw control | 0.6285177340 | 0.3049922231 |
| Explicit depth | 0.6470743106 | 0.3130374990 |

The candidate passes twelve screen components and fails five. The failed baseline-retention components, expressed as candidate minus unchanged seed42, are:

| Component | Observed difference | Required minimum |
| --- | ---: | ---: |
| frag1-context2 AUROC | −0.0991631231 | −0.02 |
| frag1-context2 AP | −0.2017177124 | −0.02 |
| frag2-context2 AP | −0.0884798600 | −0.02 |
| frag3-context1 AUROC | −0.0418728508 | −0.02 |
| frag3-context1 AP | −0.0520055512 | −0.02 |

All five improvement components pass: candidate base/full training improvement, full-risk advantage over raw, and Paris2 macro AUROC/AP advantages over raw. The seven other context-retention components pass. The complete favorable and unfavorable arithmetic and distances to every boundary are preserved in `development-v1/verification.json` and the root metric-comparison receipt.

## Execution scope and limitations

The producer performed real original-model work. Across both training arms, retained hook receipts count 64 full-original calls, 64 original depth-fusion calls, 512 original 2D-network calls, 448 adapter calls, 64 backwards and 64 adapter updates. Across development, counts are 300 full-original calls, 300 depth-fusion calls, 600 2D-network calls and 300 adapter calls, with no backwards or update. Full-model events overlap their component events and must not be added as independent whole-model passes. Four original checkpoint loads occurred across the four producer workers. Original parameter updates were zero. The reviewer itself performed zero model calls, Torch imports, checkpoint decodes, training replays or network calls.

The numerical check does not independently prove convolutional autograd, hidden-feature-to-logit computation or execution history. It verifies retained output derivatives, parameter-gradient consequences and semantic state linkage. Feature-cache, residual and decoder-input-gradient hashes/norms are execution receipts with checked schema and lineage; the 32-channel hidden arrays were not independently reconstructed. Development actual-original-versus-cache equality is a source-reviewed producer receipt because separate new original development arrays were not retained. Exact checkpoint bytes were not decoded by this role; adapter trace state hashes match strict-reload receipts. The complementary technical source review is separately pinned by the root.

Both stages passed with the checker source and numerical contract frozen before new scientific outputs. A prior root prelaunch metadata assertion confused preflight READY with PASS and was corrected before scientific worker execution, as recorded in `../ROOT_PRELAUNCH_STOP.json`. That event was not a scientific retry or a numerical-checker failure. Earlier immutable phase 18/19 results were not changed.

The review recorded 52 training and 96 development array-read events, with peak RSS approximately 181.3 MB and 67.9 MB respectively. These read events include repeated historical/result files; they do not denote independent observations. The retained `publication-review` path names a local report-arithmetic comparison and involves no external publication.

All Paris1/Paris2 contexts are reused development evidence. None is a fresh confirmation set, and correlated pixels do not supply independent replication. The depth coordinate is a nominal stack index, not authenticated physical ink depth. This result authorizes no baseline replacement, new text-reading claim, GPU launch, upload, outreach, prize submission or external publication. The existing GBP 10 total pilot cap is unaffected; this review spent GBP 0 on GPUs.
