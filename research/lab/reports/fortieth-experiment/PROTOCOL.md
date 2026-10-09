# Phase40: second disjoint source-crop replication

The user approved phase39's recommendation. Repeat phase38's mixed-source training once with a second PHerc1667 crop, comparing against unchanged seed42, A512, coverage512 and the fixed midpoint. GPU work remains deferred; CPU monetary spend is unrestricted.

## Admission history and scope

The first frozen search examined eight crops from the cached first inventory page. All failed the exact replacement class-count requirements. Its 38 metadata/label requests delivered 10,491 bytes; no CT or model execution occurred. A separately declared stage2 extended only the label inventory (12 further pages, 3,590,750 bytes), preserving class counts, 256-square crop size and minimum 256-pixel separation. Stage2 froze 64 candidates before new label bodies, excluding the eight failed candidates. Its first candidate passed, so no further candidate was inspected.

Admitted crop: PHerc1667/w013 source XYWH [2560,640,256,256], versus phase38 [3968,1408,256,256]. The source rectangles have X/Y gaps of 1152/512 level-2 pixels. Both lie on the same supplied surface coordinate system. This is a within-surface crop replication, not a new surface or scroll, an untouched test, or proof of physical non-overlap through a separately audited mesh map. The crop is now explicitly training-exposed. Selection used only fixed geometry and supplied training-label feasibility, not CT appearance or predictions.

Stage2 acquired six metadata objects, eight label/support chunks and four CT chunks: 18 requests / 7,105,901 bytes. The resulting crop has 16,085 supplied positive, 36,927 supplied negative and 12,524 unknown pixels. Unknown pixels remain excluded. Metadata hashes agree with the original public reference; online_validation=false. Validation labels are never converted into training. Total acquisition across both stages and inventory expansion is 68 requests / 10,707,142 delivered bytes. Preserve unsuccessful probes and all raw responses. This was an admission search extension before training, not a scientific training retry.

Independently reconstruct source [13,97) pooling by four with nearest-even rounding to 21 uint8 planes and central-channel binary labels/support. Keep all original model planes [2,19), forward, with unchanged whole-patch normalization. No recentering, optical registration claim or depth/fusion adapter. Supplied labels retain incomplete manual/teacher ancestry and establish comparative agreement, not authenticated ink truth.

## Identical training recipe

Start at the same original seed42 network, all 34,546,498 parameters. Repeat the unchanged original 32 batch-two schedule for exactly 512 AdamW updates: learning rate 1e-6, betas (0.9,0.999), epsilon 1e-8, zero weight decay, no scheduler, global gradient clipping 1. Deterministic CPU FP32 model; Python 3.12.14, NumPy 2.5.3, Torch 2.12.0+cpu; four intraop/one interop threads; seed 20261008. Objective remains smoothed .25/.75 BCE plus unit detached seed42 KL, with the same FP32/FP64 arithmetic. No optimized Dice/pseudo term.

Replace slot i%2 at schedule index i with one patch from the new crop. Keep the other PHerc0139 patch exactly. Eligible stride-32 128-square tiles must supply both class counts of the removed original patch. Select the least-used eligible tile, tie by Y/X, and select exact class counts using the unchanged phase38 SHA256 crop-global coordinate ordering. Here crop-global means coordinates within the 256-square crop, not absolute source coordinates. Keep the removed patch's augmentation. Repeat the frozen schedule without resampling.

Every step has exactly the original positive/negative exposure; each 32-step cycle totals 230,340 positive and 387,868 negative visits. Both original regions remain represented with identical support and patch exposure to phase38. PHerc0139 exposure still differs from narrow A512: half its patch slots are replaced. Therefore this tests crop dependence of the mixed recipe, not an isolated pure-diversity mechanism. The DATA_AUDIT field source_specific_original_exposure_matched=false refers to the A512 comparison; it is matched to phase38.

Create 32 detached seed42 teacher batches. Require each retained PHerc0139 patch's logits to match the historical teacher exactly. Train one arm D_replication to 512; record fixed-risk checks at 0,32,128,256,384,512 and original-support risk at 512. Retain states at 32 and 512 but evaluate development only at the final fixed endpoint. Check measurement isolation and optimizer continuity as in phase38. No checkpoint selection or extension.

## Evaluation and stopping rules

Use exactly the same six Paris and two PHerc0814 development regions/masks and 25-tile inference per region, no new calibration. All are reused development. Advancement requires equal-parent Paris AP >=+0.01 against BOTH seed42 and A512, and every region's AP/AUROC delta >=-0.02 against seed42. Report all raw metrics and regional deltas versus seed42/A512/coverage512/midpoint, not just aggregates.

Separate frozen descriptive replication criterion: D512 Paris AP must be at least the midpoint's AP and D512's minimum regional AP and AUROC deltas versus seed42 must each be at least the corresponding midpoint minimum. All three must hold. Passing weakens a w013-first-crop-only explanation of this particular tradeoff; failure weakens its consistency across these two selected crops. Neither provides significance, a clean diversity attribution, untouched generalization or GPU authorization. Keep advancement independent of this descriptive relation.

Hard stop at 512 updates or integrity, nonfinite or resource failure. One scientific attempt, no automatic retries, 1,800 seconds combined scientific controller wall time, 4 GiB worker RSS, 12 GiB address space, 2 GiB phase files. Counts: 32 teacher batch-two forwards, 705 student batch-two forwards, 200 development tile forwards, 512 backwards/updates. Reuse all reference maps. No network during model execution.

Before model execution, GO binds this protocol, inputs, code, independent verification and fixtures. Verify all 32 selected supports/normalization, exact historical loss/gradient parity, all retained training scalar terms to 1e-6, tie-aware ranking to 1e-12 and sigmoid/Hann blending to 2e-6. Do not claim the NumPy loss verification replays autograd/optimizer transitions. Freeze and report all criteria before outputs.
