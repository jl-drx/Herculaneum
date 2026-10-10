# Next research design: uncertain depth proposals without discarding evidence

Phase36 mechanically qualifies a texture-transition diagnostic, not a physical/ink-depth estimator. Preserve the full input depth and an unguided model path. Do not automatically recenter/crop around the proxy or select a depth by label agreement.

Next bounded work can prototype a non-destructive auxiliary representation: normalized structural energy, signed transition evidence, candidate depth plus explicit accepted/ambiguous/censored masks. Candidate depth must be missing where unusable; missingness must not become layer0 or a numerical signal that implies certainty. Keep multiple transition evidence rather than only a single argmax when competing layers exist. This is a proposal for Plinius development, not an implemented/trained model or novel-performance claim.

Freeze a known-geometry counterfactual suite before choosing representation details: isolated and multiple sheets, true target on a weaker layer, texture-only changes unrelated to target, targets outside the window, missing geometry, reversal and translations. Compare information retained by full-depth input with destructive recentering as a diagnostic negative control. Do not tune the rule on real label scores. A passing infrastructure suite would only justify a later frozen model ablation, not establish real ink accuracy.

For real development, prospectively select nonoverlapping support and preserve label/reference provenance and parent grouping. Never reuse these four inspected/scored crops as an unseen test set. Unknown pretrained/teacher ancestry on PHerc51 remains. Reserve parents PHerc0009B/PHerc0500P2/PHerc0841 remain untouched. Stop same-panel depth/feature/model sweeps unless a concrete new mechanism and fixed test resolves an actual uncertainty.

No GPU launch is implied. GPU pilot remains gated by frozen experiment/inputs, pinned CUDA environment, working scoped runtime secrets, current pricing and independent shutdown under the £10 total cap. Current GPU spend£0. Continue independent CPU development without asking for routine permission; do not send maintainer outreach unless explicitly authorized.
