# Design decisions before reserved inference

Root records these decisions after the initial Astra review and actual source inventory on 8 October 2026. The original review remains unchanged.

Only three released reference cases have validation masks: PHerc1667-w029, PHerc0139-w016 and PHerc0814-46527. The operational protocol therefore uses existing w029 development scores and at most two new non-overlapping 384² contexts per other specimen. One adequate context on each new specimen is the minimum; an arbitrary crop-count target must not force extra correlated examples. Astra explicitly reviewed and accepted this adjustment with no substantive blocker. Separate flattened contexts are not certified disjoint three-dimensional sampling supports.

One development crop provides a local empirical operating point. The independent root check verified each threshold on the actual retained float32 scores and verified that its immediately preceding float32 value would exceed floor(9378/20)=468 development false positives. All four achieve specificity0.9500959693. At those thresholds, development sensitivity is0.568798173 for ex016,0.741079075 for control,0.585498144 for seed42 and0.667285184 for seed43. These used-development results are not new reserved evidence. The frozen threshold lock predates new label/CT acquisition and reserved inference.

Astra's interpretation nuance: ex016 keeps w016 manually supervised, whereas PHerc0814 receives dense supervision. Heterogeneous outcomes could motivate a supervision-location hypothesis, but scroll, annotation and treatment-location differences are confounded; this panel cannot identify that mechanism.

Native-source feasibility is a separate audit. Matching specimen/segment identifiers does not make different scan-energy canvases pixel-aligned, and model-derived or transferred annotations do not become independent truth. The current fixed panel uses only the two verified pooled fine-scan representations. Native acquisition effects remain untested. No source/model/provenance expansion is required beyond the frozen panel to support its limited claims.

The next scientific action follows the result, including an unfavorable or heterogeneous result. Research remains the priority, with prize eligibility recorded separately; no training, paid resource, publication or R2 write is implied by this design record.
