# What our own findings do and do not close

Reviewed 12 October 2026. These are retained results, not new computations of
historical predictions. Exact local report hashes are in REVIEW_RECEIPT.json.
The native-learning evaluation is excluded from this review and remains locked.

| Evidence | Supported conclusion | Research consequence / what would reopen it |
|---|---|---|
| [Studies 18–20](../../lab/reports/eighteenth-experiment/RESULTS.md): small explicit-depth/residual and fusion designs | Those bounded adapters did not establish useful transfer. Small updates plus preservation constraints restrict the tested change. | Do not repeat them. Independent sheet-instance inputs and measured neighbouring-wrap contamination would be a different mechanism; a renamed depth head would not. |
| [Study 35](../../lab/reports/thirtyfifth-experiment/RESULTS.md): 284/284 historical render samples reproduced; zero accepted optical/CT landmarks | Input mechanics can be exact while optical registration remains unverified. The historical interpolation defect is distinct from physical alignment. | Frag6 cannot be an independent optical ink endpoint until correspondence is actually supported. More exact renderer replay does not settle that. |
| [Study 36](../../lab/reports/thirtysixth-experiment/RESULTS.md): texture-depth transitions, search-edge censoring | Repeatable texture boundaries do not identify a reliable physical ink face. | Keep full depth; no automatic recentering by that proxy. Reopen only with independent geometric reference and an estimator addressing the specific failure. |
| [Study 37](../../lab/reports/thirtyseventh-experiment/RESULTS.md): +0.004960 AP after 512 updates, with -0.025821 worst regional AP regression | Longer narrow continuation improved some fit/transfer but failed the declared development gate. Dice improved fit without useful transfer. | Do not extend those trajectories. This was 1,024 patch presentations, not a well-powered test of released-scale learning; it does not eliminate GPU-scale training. |
| [Studies 38–41](../../lab/reports/fortyfirst-experiment/RESULTS.md): coverage improved retention; removing the added crop's BCE gave a tradeoff | Extra supervision and reduced original-source exposure were not cleanly separated in the coverage design. The new BCE had mixed effects. | New coverage work must hold original-source exposure fixed. Another crop substitution on the same Paris panel is not independent replication. |
| [Studies 44–45](../../lab/reports/fortyfifth-experiment/RESULTS.md): native profilometry did not meet fit gate; 8,192-update continuation still improved training but not held-out transfer | A modality-specific CPU implementation did not establish the intended learnability result. It does not show absent morphological or CT signal. | Keep this branch shelved. Do not make it a prerequisite for native CT work. |
| [Study 45 checkpoint panel](../../lab/reports/fortyfifth-experiment/RESULTS.md): seed43/75k vs seed42/75k +0.075216 aggregate Paris AP, with mixed regional effects | Released checkpoint choice can dwarf the early +0.01 practical gate. The four checkpoints are not independent controlled seed draws; configurations differ. | Use strong, development-chosen baselines and paired training seeds. Do not infer a noise floor or significance threshold from two released models. This panel has already run. |
| [Study 45 geometry](../../lab/reports/fortyfifth-experiment/geometry/RESULTS.json): median absolute mesh-to-m7 offsets 1.0, 1.0, 0.5 voxels on measured rays | Blanket snapping is unsupported. There are local outliers, ambiguous/multiple runs and many invalid sampled normals. Mesh-to-model disagreement is not physical placement error. | No need to repeat the median audit. A sheet-identity/connected-coverage comparison is new and still open. |
| [PHerc343 completion](../../in-progress/pherc343-reproduction/completion-20261010/README.md): four-model, eight-map fixed-image reproduction | The saved recipe reproduces the exposed published display closely. It is a useful positive control for implementation. | It is not raw-author-map equality, ink AP, new letters, or the entire 496-map author sweep. It stays development material. |
| [Study 47](../../lab/reports/fortyseventh-experiment/RESULTS.md): four frozen tiling origins changed responses substantially | Sampling-grid sensitivity is real; greater stability does not identify the more accurate model. Ownership-boundary effects did not explain the main candidate forms. | Do not run more origins on that same unlabelled crop. A labelled positive/negative robustness comparison with equal-cost controls would answer a different question. |
| [Study 51](../../in-progress/pherc0800-development-20261010/completion-20261010/RESULTS.md) and [52](../../in-progress/pherc0800-depth-20261011/completion-20261011/FINAL_RESULTS.md): no qualifying stroke group at d=-15,0,+15 across the fixed recipe | Those exposed target fields did not produce a defensible new reading. The broad d+15 response remains uncertain. | Close the three-position screen; no brighter-region/depth shopping. This does not rule out ink, intermediate depths or a better geometry/model. A new physical mechanism and separate development design are required. |
| [Research review](../plinius-research-review-20261011/README.md): same-scan reconstruction variants differ; fine/native landmarks have residuals | Voxel pitch is not the whole acquisition effect; published transforms are not precise local registration certificates. | Reconstruction robustness is open. The prior three-site intensity test did not compare ink accuracy or matched training. |
| [Eight-site CT study](../pherc0139-eight-site-ct-20261011/README.md): six scorable sites, checking NCC 0.828–0.973, zero strict full passes; two source-support failures | Good structural correlation can coexist with unacceptable local ambiguity. | Do not silently admit those transfers as precise pixel labels or keep searching offsets/crops until they pass. |
| [Six-site supervision audit](../pherc0139-six-site-supervision-20261011/README.md): only 2/24 primary patches class-balanced; both hard and soft targets materially sensitive | Soft targets did not remove the measured correspondence problem. Averaging alternatives was almost identical to matched stationary smoothing. | Close this local diagnostic. Another averaged-target BCE arm has little novelty or information value without new mechanism/control. These six sites do not estimate paired learning potential generally. |
| [Native reference](../pherc0139-learning-readiness-20261011/native-reference/README.md): 265 cores, native annotations with explicit supervision masks | A direct native-frame annotation-agreement task exists without solving fine/native registration first. Annotation ancestry and physical ink depth remain limited. | The current nine-fit study is worth completing; its small model and same-scroll folds do not define the frontier of native learning. |

## The largest strategic omissions

1. **Surface coverage and identity were not an active research endpoint.** We
   mainly worked inside existing renders, so omitted or misconnected papyrus
   could never be recovered by our experiments.
2. **Strong newer native models were not the main training baseline.** The
   four-model PHerc343 reproduction remains useful, but Reader v2 and broader
   native coverage should now enter model comparisons. A weak comparator can
   make an unimportant new method appear useful.
3. **Reference quality was inspected more often than improved.** A bounded
   correction experiment or physically anchored geometry benchmark can provide
   new information; repeated provenance checks cannot create independent truth.
4. **Our evaluated physical diversity remains narrow.** Windings from one scroll,
   repeated Paris regions and aliases of the same tissue do not substitute for
   external specimens/acquisitions.
5. **Power and useful compute were not always aligned.** CPU can handle serious
   training for small models, but short updates to a large pretrained network
   do not establish whether its representation can learn the task. Adequate
   data, schedule, effective exposure and a sensitive endpoint matter more than
   a universal device rule.
6. **End-to-end value is not captured by AP alone.** Correct connected area,
   wrong-sheet contamination, abstention, correction time and independently
   supported legibility should accompany model metrics. A detector becoming
   brighter or smoother is insufficient.

These omissions motivate parallel work; none establishes that geometry is the
single cause of PHerc0800's weak maps. The portfolio should be able to discover
that a proposed bottleneck is not dominant and redirect effort.
