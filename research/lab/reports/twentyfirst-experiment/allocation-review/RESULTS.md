# Independent review: preservation-budget allocation

**Numerical execution and report arithmetic pass. All five training mechanism criteria pass, but the practical development screen fails five of fourteen criteria.** Every checker stage passed on its first frozen attempt, without source repairs or tolerance changes after scientific outputs. The numerical findings do not authorize replacing seed42 or renting a GPU.

The candidate moved one unit of teacher consistency from manual M to explicitly retired supplied-reference V. It reduced teacher disagreement on V in the training domain, while failing the required development improvement and three regional baseline-retention criteria. This establishes the declared empirical training change; it does not establish a causal explanation for regional generalization failures or independent scientific confirmation.

## Scope and input identity

The governing protocol is `PRESERVATION_ALLOCATION_PROTOCOL.md`, SHA256 `498bf89ca0944ca26f596bdc8041951a82365667ccddf4eb88a8a1f0b33998b1`. The final executable plan is SHA256 `2e1cc8c46d5f7aaade31916cdc0304cd6cb8268e49d8aea73927c730dd535f74`. The checker and numerical contract were frozen before scientific outputs in `CHECKER_FREEZE.json`.

Independent NumPy reconstruction verified the exact 32 PHerc0139 batch-two inputs, historical fingerprints, manual/retired masks, source indices and explicit-depth feature tensors. Both arms receive the same features and begin with identical adapter tensors. All scheduled actual-original and initial zero-projection logits are bitwise equal to retained seed42 teacher logits. The original model's fused-cache receipts agree across arms. The supplied-reference masks are the previously sealed participation masks, not authenticated geometry-valid masks.

Historical ink/pseudo arrays were read only to reconstruct old complete-batch fingerprints. Retired ink-label values did not enter objectives, teacher construction, support selection or retired-label scores. The candidate preservation support excludes annotation-unknown outputs. Broader geometry-valid preservation remains blocked independently of this executed, narrower allocation study.

## Objective and gradient verification

For each batch, A (`manual_double`) optimizes B_M + 2 K_M and B (`manual_retired`) optimizes B_M + K_M + K_V. BCE is averaged only over manual hard targets smoothed to 0.25/0.75. Each FP64 Bernoulli KL is a separate mean over its named, nonempty stratum. Full-schedule measurements equally average the 32 batch risks and retain repeated/overlapping exposures.

The reviewer independently reconstructed all nine saved terms: base, manual KL, retired KL, objective_A, objective_B, the arm's own full objective, manual hard BCE and probability drift on both strata. It checked that own full equals the correct shared formula and did not rank unlike own objectives against each other. Both common formulas remain available for like-for-like diagnostics.

With p = sigmoid(student), q = sigmoid(teacher) and y_s = 0.25 + 0.5y, A's manual output derivative is `(p−y_s + 2(p−q))/|M|`, with zero elsewhere. B's manual derivative is `(p−y_s + p−q)/|M|`, plus `(p−q)/|V|` on retired V and zero outside the union. All unsupported derivatives are exactly zero under the appropriate arm-specific mask. Hidden-feature gradients can spread beyond these output masks through the decoder; that does not change output-level objective participation.

All 5,296 ordered parameters, clipping and AdamW transitions, moment continuity and semantic state hashes were checked at every update. Initial projection weights/biases and optimizer moments are zero; initial stem parameters are random and matched. Both arms activate the expected projection/stem learning path. Original model parameters, buffers, modes and parameter-gradient absence agree with the frozen execution receipts.

| Verification | Observed maximum or count |
| --- | ---: |
| Loss/summary scalar comparisons | 5,108 |
| Optimizer scalar comparisons | 1,355,776 |
| Exact adjacent parameter/moment scalars | 985,056 |
| BCE/full/shared-objective error | 1.1878710515222934e-7, within 2e-7 |
| Manual KL error | 1.734723475976807e-18, within 1e-12 |
| Retired KL error | 3.469446951953614e-18, within 1e-12 |
| Hard BCE error | 7.808371565998229e-8, within 2e-7 |
| Either probability-drift error | 6.938893903907228e-18, within 1e-12 |
| Supported output-gradient error | 1.0042895443214406e-11, within 2e-10 |
| Reported FP64 gradient/update-norm error | 1.1102230246251565e-16, within 1e-12 |
| FP32 global-gradient-norm error | 8.940696716308594e-8, within its frozen mixed bound |
| Parameter-update error | 7.450580596923828e-9, within the frozen absolute-plus-ULP bound |

Neither scientific arm clipped a gradient. Both clipping branches and deliberate optimizer corruptions had already been exercised by the frozen synthetic fixtures. Immediate post-update and fixed-schedule current-batch outputs match bitwise at updates 1, 8, 16 and 32; all scheduled measurements are retained.

## First-step diagnostic

First-step parameters, parameter gradients, Adam moments and pre/post-update logits agree bitwise across arms. The saved output-gradient arrays differ in 4,444 float32 bit patterns, with maximum absolute difference 6.776263578034403e-21 and L2 difference 1.8977319038844038e-19. These tiny supported-output differences are compatible with finite-precision KL cancellation. They did not survive into the saved first-step parameter-gradient or parameter vectors.

This was reported under the predeclared diagnostic-only contract. No inter-arm tolerance was fitted, and exact first-step equality was not an acceptance requirement. Each optimizer trajectory passed its own independent arithmetic checks.

## Training measurements and all five mechanism decisions

Both arms start at B_M = objective_A = objective_B = 0.5970771741122007, with zero manual/retired KL and drift. The final declared producer means are:

| Shared diagnostic | A: manual double | B: manual plus retired |
| --- | ---: | ---: |
| Manual smoothed BCE | 0.5937815066426992 | 0.5937956906855106 |
| Manual KL | 0.0014933086691720845 | 0.000902447629463882 |
| Retired KL | 0.0012055920111736758 | 0.00056186354989094 |
| objective_A = B_M + 2 K_M | 0.5967681239810434 | 0.5956005859444384 |
| objective_B = B_M + K_M + K_V | 0.596480407323045 | 0.5952600018648655 |
| Manual hard BCE | 0.32057489920407534 | 0.32020418159663677 |
| Manual absolute probability drift | 0.01007561112517518 | 0.007516071126244586 |
| Retired absolute probability drift | 0.009569570538326513 | 0.006543241101060464 |

A's own optimized scalar is objective_A; B's is objective_B. Their own-full improvement tests compare each arm with its own initial value, not with the other arm's unlike objective.

| Mechanism component | Value | Minimum | Outcome |
| --- | ---: | ---: | --- |
| A own-full improvement | 0.00030905013115734015 | 0.0001 | PASS |
| B own-full improvement | 0.0018171722473352414 | 0.0001 | PASS |
| B manual-BCE improvement | 0.0032814834266901016 | 0.0001 | PASS |
| Retired KL(A) − retired KL(B) | 0.0006437284612827358 | 0.0001 | PASS |
| Manual BCE(A) + 0.001 − manual BCE(B) | 0.0009858159571886072 | 0 | PASS |

Independent reconstructed values produce the same five decisions. The candidate's manual BCE exceeds A by approximately 0.0000141840, within the declared 0.001 allowance. These are empirical training diagnostics; they neither authenticate the teacher nor establish external efficacy.

## Development reconstruction and all fourteen practical decisions

Both final arms were checked on all six fixed contexts, with 25 tiles each. The independent implementation reconstructed normalized inputs, shared explicit-depth feature hashes, sigmoid probabilities, Hann blending, placements, coverage, masks and supplied-label participation. Maximum score-map discrepancy across all three models was 2.980232238769531e-7, within 2e-6. The baseline is included in that maximum.

The reviewer independently calculated all 18 context/model rank and fixed-cutoff summaries, equal-context/equal-fragment aggregations and all screen components. The unchanged literal cutoff is 0.7558947205543518. Published metrics, confusion counts, all gates and the separately identified historical phase20 reference agree with the frozen comparison. There are no producer-versus-independent pass/fail ambiguities.

| Paris2 equal-fragment mean | AUROC | AP |
| --- | ---: | ---: |
| Unchanged seed42 | 0.6576421877658194 | 0.3842856729084452 |
| A: manual double | 0.6682161475002897 | 0.36636404347957197 |
| B: manual plus retired | 0.6407270327643089 | 0.36445057469770126 |

The candidate practical screen passes nine components and fails five. Values below are candidate minus seed42, with the declared minimum shown explicitly.

| Practical component | Difference | Minimum | Outcome |
| --- | ---: | ---: | --- |
| Paris2 macro AUROC gain | −0.01691515500151053 | 0.005 | FAIL |
| Paris2 macro AP gain | −0.019835098210743918 | 0.005 | FAIL |
| frag1-context1 AUROC | −0.02106592301946375 | −0.02 | FAIL |
| frag1-context1 AP | −0.006895727972040411 | −0.02 | PASS |
| frag1-context2 AUROC | −0.08621151467898824 | −0.02 | FAIL |
| frag1-context2 AP | −0.12805345068077761 | −0.02 | FAIL |
| frag2-context1 AUROC | 0.05644490861584117 | −0.02 | PASS |
| frag2-context1 AP | 0.023575231439782185 | −0.02 | PASS |
| frag2-context2 AUROC | −0.016828090923431516 | −0.02 | PASS |
| frag2-context2 AP | 0.03203355437006006 | −0.02 | PASS |
| frag3-context1 AUROC | 0.0018298705361192713 | −0.02 | PASS |
| frag3-context1 AP | −0.0007557817317699334 | −0.02 | PASS |
| frag3-context2 AUROC | 0.02702088259827895 | −0.02 | PASS |
| frag3-context2 AP | −0.0015442839029573274 | −0.02 | PASS |

The results are heterogeneous. An AUROC gain for A or a favorable region for B does not replace the failed prespecified candidate screen. The historical phase20 adapter uses a different objective and remains a descriptive reference, not a new matched trial arm.

## Evidence boundaries and resources

Training review recorded 69 array-read events and approximately 195.5 MB peak RSS; development review recorded 96 array-read events and approximately 68.7 MB peak RSS. Read counts include repeated retained files and are not independent observations. The reviewer made zero Torch imports, model calls, checkpoint decodes, training replays and network requests. The local `publication-review` directory contains report-arithmetic comparison only, with no external publication.

Producer hook counts remain those frozen in the protocol: across training, 64 original full-model, 64 original fusion, 512 original 2D-network and 448 adapter forwards, with 64 backwards/decoder-input backwards/updates. Across development, 300 original full-model, 300 fusion, 600 2D-network and 300 adapter forwards, with no backwards or update. These are overlapping full/component events, not additive independent whole-model passes. Four producer workers loaded original seed42 once each; original parameter updates were zero.

Saved gradients and hidden tensor hashes are producer execution evidence. Independent checks establish masked output derivatives, clipping/optimizer consequences, trace continuity and reported arithmetic; they do not replay convolution/autograd or independently prove hidden execution history. New original-versus-cache development equality relies on reviewed code and its pinned receipt rather than independently retained original development output arrays. Trace state hashes agree with strict checkpoint reload receipts, but checkpoint bytes were not decoded here.

Every Paris1/Paris2 context is reused development evidence. Retired PHerc0139 V is an explicitly revised teacher-consistency role, not fresh validation. Unknown geometry and annotation/teacher ancestry remain qualified. The observed training improvement does not establish frontier performance, readable new text, independent confirmation or a reason to scale this recipe. Root retains scientific judgment. GPU spend in this review was GBP 0; the existing GBP 10 total first-pilot cap is unchanged.
