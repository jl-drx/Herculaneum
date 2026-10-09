# Fixed-compute training-coverage comparison

The user approved proceeding after phase37 recommended broader defensible training coverage at fixed CPU compute. This study tests one mixed training schedule against the completed, exactly reproduced phase37 A control. No GPU work, new acquisition, recentering, depth/fusion adapter or architecture change is involved.

## Admission and question

The additional data is the retained PHerc1667/w013 crop at source XYWH [3968,1408,256,256]. It is the assembled, operationally supervised crop available outside the excluded evaluation/reserve parents. It was selected for availability and eligibility, not ranked by candidate evaluation performance. This crop was previously inspected development material; it becomes explicitly training-exposed in this study. Cached w029 is rejected because its supplied training supervision is empty. We do not convert its validation labels into training labels.

DATA_AUDIT.json records exact reconstruction of the w013 input from four delivered raw chunks and exact decoding of its central-plane labels/support. Its Blosc-compressed public level-2 source is pooled over depth [13,97) in groups of four using float32 means, nearest-even rounding and uint8 output. Model input remains pooled [2,19), all 17 original planes, forward. Source and label attributes agree on 9.596 µm spacing, source identity and pooling. w013 declares online_validation=false; the dataset documentation supplies validation masks only for w016, w029 and PHerc0814. PHerc0139 validation pixels remain excluded. Source reconstruction does not authenticate optical alignment or physical ink depth.

All labels retain the same limited status as phase37: supplied public manual-plus-iterative reference annotations with incomplete human/teacher ancestry. Their participation masks support this comparative training experiment; they do not establish independent ink truth. No Paris or PHerc0814 evaluation labels, Frag6 data or reserve-parent data enter training or teacher generation. PHerc0009B, PHerc0500P2 and PHerc0841 remain reserved.

Question: does distributing the same update and class-exposure budget across two training parents improve transfer agreement over the narrow PHerc0139-only recipe? This is one particular source mixture. It does not isolate all effects of parent identity, spatial label layout or teacher confidence, and it is not a complete training-data scaling study.

## Fixed schedule and objective

Start from the original seed42, full 34,546,498-parameter network. Keep AdamW learning rate 1e-6, betas (0.9,0.999), epsilon 1e-8, weight decay 0, no scheduler, global gradient clipping 1, deterministic CPU FP32 model, four intraop/one interop threads, seed 20261008. Runtime Python 3.12.14, NumPy 2.5.3, Torch 2.12.0+cpu. The optimized scalar is unchanged: FP32 BCE with targets mapped to .25/.75 plus unit-weight FP64 KL(original seed42 || student) on the same manual participation. No Dice or pseudo-target term is optimized. The new manual-only implementation removes unused pseudo diagnostics; exact value and gradient parity on all 32 historical batches was tested before execution.

Begin with the old 32 batch-two schedule. At zero-indexed step i, replace slot i%2 with one w013 patch; retain the other PHerc0139 patch exactly, including input, support, target and augmentation. Both old PHerc0139 regions therefore remain represented. w013 candidates are the 25 stride-32 128² windows inside the retained 256² crop. Admit a tile only if both class counts can match the removed patch. Choose the least-used eligible tile, tie by source Y/X. Within each class, choose exactly the removed patch's count by a fixed SHA-256 ordering of global pixel coordinates. This ordering and the schedule do not change between cycles. Use the removed patch's XY augmentation. Normalize the whole 17-plane patch independently of its supervision mask using the unchanged normalizer. Mask all unselected/unknown labels out of the objective.

This exactly preserves every update's positive and negative supervision count, with 230,340 positive and 387,868 negative visits per 32-update cycle. Across the fixed schedule, unique supervised pixels increase from 54,701 to 110,014: the original 54,701 plus 55,313 w013 pixels. This controls labeled-pixel exposure and class frequency, but changes their spatial/source distribution. Repeat the fixed schedule 16 times, without optimizer reset, for exactly 512 updates. Repeated visits are not independent observations.

Create one detached original-seed42 cache for all 32 mixed batches before training. Confirm its unchanged PHerc0139 patch logits exactly match the historical cache. The teacher is a regularizer, not independent supervision. No teacher selection or ensemble is introduced.

## Measurements and decision

Retain all preupdate logits, supported losses, batch fingerprints and gradient checks. Measure fixed mixed-schedule risk at updates 0,32,128,256,384,512; update 0 uses the frozen cache. Save states at 32 and 512, but evaluate development only at the fixed final 512 endpoint. At 512 also measure the original 32-batch PHerc0139 risk, to compare fitting on common support with phase37 A512. Measurements must preserve parameters, optimizer, gradients, modes and Torch/NumPy/Python RNG states. Verify all 264 optimizer steps at each measurement and reload saved states exactly.

Reuse the unchanged seed42 and phase37 A512 development maps. Their existing source, runtime, objective, schedule and optimizer identities remain pinned. The class-count matching and tested objective parity make that retained control applicable without a repeat training run. There is no new calibration or threshold fit.

Evaluate C512 once on the same six Paris contexts and two PHerc0814 retention contexts, with unchanged 25-tile inference, normalization, Hann blending and full canonical scoring masks. All eight are reused development, never an untouched test. Report each AP/AUROC, equal-context means within each parent and equal weight for Paris1/Paris2. Report both fixed 0.5 and historical 0.7558947205543518 confusion diagnostics separately.

Pass only if equal-parent Paris AP is at least +0.01 above both unchanged seed42 and phase37 A512, and no one of eight regions loses more than 0.02 AP or AUROC against seed42. These are practical development margins, not statistical significance. Report every regional delta against both references. A pass supports prospective CPU replication; it does not authorize a GPU launch or establish accuracy. A failure rejects advancing this particular mixture, without proving all broader-data training ineffective.

## Bounds and verification

Hard stop at 512 updates, 1,800 seconds combined scientific controller wall time, 4 GiB worker RSS, 12 GiB address space and 2 GiB new phase files. Independent parent timeouts enforce the remaining wall time. Stop on identity, finite-value or resource failure; preserve partials and do not retry automatically. No result-driven extension, learning-rate search, checkpoint selection or source replacement.

Scientific counts: 32 teacher batch-two forwards; 705 student batch-two forwards (512 updates, one first-post check, 160 mixed fixed-risk checks and 32 original-risk checks); 200 development single-tile forwards. Total 937 calls, 512 backwards/updates. Reuse all reference predictions. No fresh scan/label requests or R2 writes. GPU spend remains £0; the £10 total future-pilot cap is unchanged.

Seven pre-run fixtures passed, including exact historical control loss/gradient parity. Separate data verification reconstructs all 32 source-coordinate/class-support selections. Post-run NumPy arithmetic checks all retained training losses with 1e-6 absolute tolerance; it does not replay neural autograd or optimizer transitions. Independent evaluation checks exact-tie AP/AUROC to 1e-12 and sigmoid/Hann score blending to 2e-6, plus all masks, parent means, deltas and decisions. These tolerances and review code are fixed before scientific outputs.
