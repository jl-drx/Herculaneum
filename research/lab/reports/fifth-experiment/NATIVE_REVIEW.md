# Native CT and source-render review

The separately revalidated comparison reproduces **309 of 310 surface samples exactly; all 310 agree within one uint8 intensity count**. The one-count residual is at peak A `(1278,504)`, depth index 15 / normal offset 0. Its cause remains unresolved; no offset, normal sign, transform or interpolation parameter was fitted.

Ten fixed source points—the two prior detector peaks and four one-pixel cardinal neighbours around each—were compared at all 31 recorded depths. All interpolation corners came from two observed native CT chunks, totaling 4,194,304 bytes, plus 239 bytes of metadata. Missing data was not reconstructed. These nearby points share two normal-cache neighbourhoods and do not form independent accuracy samples.

## Integrity stop and separate recovery

The first execution calculated the above comparisons, then correctly stopped because a review-time code edit overlapped execution and its final input-size check failed. Preserve `native-check/result.json` as a **stopped** attempt. The original frozen plan and code were restored byte-for-byte, to plan SHA `50c8fa7cea430111b61a4c66dfd87965dee8bd4e53d06323151a83c592ce9a67` and code SHA `4d74a492e05c38f7a5074b9b8b6b6d35d5f69d041bb539310386971f6b4357b5` respectively.

`revalidate_offline.py` then separately verified all original frozen inputs and the two HTTP-200 cube hashes, blocked network connection attempts, and recomputed every profile using the restored source. Every profile equals the stopped attempt's retained calculation, and all input hashes still matched afterward. The successful technical result is `native-check/offline-revalidation.json`; it does not rewrite the failed record or claim the first run completed cleanly.

The original `native-check/manifest.json` captured a transient revised-plan identity during the failed attempt. It is historical failed-run evidence, not a valid current package manifest. The additive recovery manifest must pin the restored plan/code, stopped predecessor and separate successful revalidation explicitly. No new native payload was downloaded during recovery.

## Fixed cross-sectional views

The [four CT cross-sections](native-sections/native-cross-sections.png) use recorded base coordinates, normal and orthogonal in-plane frames: tangent offsets −48…48 voxels and normal offsets −24…24. Grayscale is fixed at 0–255. The local tangent reference at depth 0, input depth ±8 and rendered depth ±15 are marked. The red line labeled “Base surface” is this fixed local tangent reference at the peak, not the curved mesh across the full section; the green/blue lines likewise describe local-frame offsets, not globally reconstructed sampling support. Purple pixels lack complete eight-corner support from the retained cubes and remain unknown; they do not indicate empty tissue. The [fixed depth profiles](native-sections/native-profiles.png) retain the raw observed/reconstructed comparison.

Root visual inspection sees locally banded material near both sampling surfaces, with curvature and varying intensity through the depth window. The base surface lies near a material band/boundary rather than showing an obvious gross mismatch to the displayed local structure. A brighter intensity band is not an ink label, and the fixed tangent planes do not reconstruct the whole curved papyrus sheet. Several nearby bands contribute within or near the normal-depth context, so these views do not identify recto/verso, independent layers, or the cause of the detector responses.

The four visualizations have 17,801 observed and 1,211 unknown samples in total. All required corner samples for the ten original 31-depth profiles are observed despite the unknown outer visualization areas. Plane plotting uses diagnostic float64 interpolation, distinct from the archived float32/FMA equality calculation.

## Scope of the decision

The recovered mapping check supports a bounded, explicitly diagnostic larger surrounding view with unchanged preprocessing. It does not support training labels, claimed ink, transcription or translation. A separate assistant inspected both PNGs and the plotting code and found no gross contradiction to the recovered local sampling correspondence, while reaching the same limitations about ink and layer placement. The parent context decision must separately record visual review and the revised context geometry rule; the original result's broad `expansion_gate_passed` name does not replace that decision.
