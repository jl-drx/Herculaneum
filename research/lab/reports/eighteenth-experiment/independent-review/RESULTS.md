# Independent phase18 verification

The numerical review passes. The predeclared feature screen fails. The explicit-depth residual shows a small advantage over the matched raw-input residual, below all three required superiority margins. This does not justify model promotion or a GPU launch.

| Predeclared screen component | Observed value | Required minimum | Result |
| --- | ---: | ---: | --- |
| Candidate base-risk improvement versus initializer | 0.0005055274814 | 0.0001 | Pass |
| Candidate full-risk improvement versus initializer | 0.0004131405808 | 0.0001 | Pass |
| Candidate full-risk advantage over raw control | 0.0000862695185 | 0.0001 | Fail |
| Paris2 development macro AUROC advantage over raw control | 0.001815730768 | 0.005 | Fail |
| Paris2 development macro AP advantage over raw control | 0.001206540636 | 0.005 | Fail |
| Candidate AUROC and AP loss versus baseline in every context | All twelve differences within -0.02 | -0.02 each | Pass |

The unchanged seed42 Paris2 macro AUROC/AP are0.6576421878/0.3842856729. The explicit-depth values are0.6571073429/0.3840514298, and raw-control values are0.6552916122/0.3828448892. The candidate's improvement over the raw control therefore does not imply improvement over the unchanged baseline. Both Paris1 contexts and all four Paris2 contexts remain separately reported in the machine-readable review; no context or final arm was discarded.

## Verified evidence

`training-v2/verification.json` independently reconstructs the exact32 historical PHerc0139 batches, input/target/mask hashes, both feature representations, manual-only loss, temperature-one preservation KL, immediate and fixed-risk measurements, all manual logit derivatives and exact zero gradients outside manual support. Both arms have identical initial parameters and RNG receipts. The zero projection reproduces cached baseline logits, including the all32 cached add-zero identity.

The independent NumPy AdamW checker verifies64 updates across801 ordered parameters, clipping coefficients, first/second moment recurrences, exact adjacent trace continuity and parameter updates. It performs205056 scalar update comparisons and148986 exact adjacent-boundary comparisons across both arms. Largest parameter reconstruction error is7.4505805969e-9; largest manual logit-gradient error is9.5847587340e-12. All frozen tolerances pass. `trace-identity-verification.json` additionally binds the initial/final named parameter-vector hashes to the worker and strict-reload receipts and checks all192 logged stem/head/update norms. Checkpoint bytes were not decoded by this reviewer.

`development-v1/verification.json` independently reconstructs both feature recipes for all150 distinct cached development tile inputs, verifies all300 new tile residual compositions, and recomputes sigmoid/Hann blending and complete coverage for all six contexts. Maximum blended score discrepancy is2.9802322388e-7, below2e-6. The unchanged baseline's retained tile logits and maps are also checked. AUROC uses independent tie-group pair counts; AP uses grouped threshold increments. All18 context/model rank and fixed-cutoff metric records are retained.

`publication-review/comparison-v1/comparison.json` independently compares the root's published metrics with this reconstruction: per-context rank and operating metrics, equal-context fragment means, equal-fragment Paris2 means, training risks and all17 screen components agree. Published gate arithmetic is exact under its declared scalars. No producer-versus-independent decision ambiguity occurs.

## Preserved review stop

The first training review stopped at an exact metadata-schema comparison because the final frozen producer included the additional required Boolean `all32_cached_add_zero_bitwise_identity`. It had reconstructed historical inputs but had not evaluated the new logit outputs. `training-v1/failure.json`, the original checker and its preflight remain retained. `REVIEW_SCHEMA_CLARIFICATION_V2.md` and the separately versioned checker add this field and independently test the corresponding identity. No scientific choice, numeric tolerance, model output, training run or screen margin changed.

## Limits and scope

These are reused development regions, not fresh confirmatory evidence. There is no fresh quarantined parent-level confirmation set available. The feature bundle jointly changes axial differences and a signed nominal stack-index coordinate; neither anatomical depth sign nor separate coordinate/difference benefit is established. Upstream checkpoint/teacher exposure remains incompletely known.

The reviewer ran no model, training replay, checkpoint decoder or network request. Gradient-to-parameter convolutional autograd and execution history remain source-reviewed claims supplemented by pinned traces; the independent arithmetic does not prove those parts anew. Cached seed42 predictions establish zero newly executed backbone loads, forwards or updates. No threshold was fitted, no scientific result selected a checkpoint, and no GPU resource was launched.

The reviewer source, approved numerical contract, preflight, root GO, immutable plan, start receipts, successful outputs and preserved failed attempt are linked by `verification.json` and `retention-manifest.json`. This directory is sealed after those files are written.
