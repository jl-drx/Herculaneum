# Next bounded study: CT-only texture depth reference

Test whether a fixed local texture-transition detector has useful, stable depth structure without consuming optical labels or model scores. Motivation comes from inspected CT sections and depth profiles, so this is development work on seen images, not a blinded confirmatory claim.

Use spatial gradient energy of each 3×3-spatial-mean CT layer, averaged in a fixed 7×7 coarse window. Compare five consecutive layers immediately before and after candidate boundaries. Normalize the signed contrast by the maximum five-layer mean energy on that ray. Retain the strongest absolute transition, reject weak contrast and competing separated transitions, and keep its sign as descriptive orientation only. Fix numerical thresholds and synthetic acceptance criteria before computing real maps.

Synthetic cases must exercise a single known boundary, spatially sloped boundary, flat/noise-only signal, two comparable interfaces, positive affine intensity changes and depth reversal/translation. Failure of the prespecified synthetic gate blocks real-map promotion; preserve the failure rather than tune thresholds on these outcomes. On real images report all contexts, coverage/abstention, depth distributions and predeclared perturbation stability. No ink-accuracy claim, model training, GPU, new acquisition, optical warp, reserve access or label-scored sweep is allowed in this study.

Even a successful proxy cannot identify which physical surface carries ink. A subsequent model-input experiment would require its own frozen design and separate evaluation support. Unresolved reference provenance remains visible.
