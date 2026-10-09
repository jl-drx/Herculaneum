# Phase29: public benchmark audited; bounded scoring adapter qualified

The public fragment-depth benchmark contributes a useful separation of ink-map agreement, depth relative to an estimated surface, and surface response. We audited its scorer, preprocessing, splits and prediction export at pinned source identities, then qualified a small local adapter that makes coverage and unknown labels explicit. This advances evaluation quality; it is not an improvement in measured ink detection.

**26 tests passed** in the completed offline run:16 adapter tests and10 targeted upstream reproductions, zero failures/errors/skips. Exact AUC/AP matched independent scalar oracles over1,134 exhaustive four-example cases and60 deterministic randomized tied/permuted cases. Another100 uint8cases confirmed the upstream histogram's exact-ordering claim for that encoding. Attempt2 took0.435127111 seconds; peak process RSS66,019,328 bytes. These timings cover tiny synthetic qualification, not a real benchmark or model inference.

Attempt1 is retained:25 passed,1 error because the offline guard blocked asyncio's local AF_UNIX socketpair while Zarr initialized. The revised driver permits local socket construction while continuing to block network connect/bind/DNS/IP socket creation and process launches. Source, tests and acceptance contract did not change. No internet request occurred in either qualification attempt. This is a Python audit-hook guard, not a claim of OS-level isolation.

## What changed

reference_metrics.py provides bounded NumPy helpers for exact tie-aware AUC/AP, explicit probability/label decoding, declared row coverage, complete-profile and complete-band diagnostics, a descriptive clear/ink mean ratio, and parent-group folds. Invalid predictions remain separate from valid zeros; unknown labels stay outside the negative class. Degenerate class sets return null/reasons. Depth results retain exclusion counts and distinguish zero-mass profiles from tied peaks. Float and both-endian uint16 values are handled explicitly. No model, loader, threshold search or bootstrap CI is included.

The synthetic tests reproduced upstream edge cases:

- A full-size exported prediction can contain uncomputed rows whose zeros are read by default unless rows are explicitly restricted.
- ignore-label 2 becomes negative with label_value=1; opposite-endian uint16 bypasses normalization.
- Partial depth bands get a numerical mean; no-positive AP becomes0.
- Close float scores merge into a histogram tie, although uint8 ordering remains exact.
- A small ink sample can mislabel clear pixels or raise; tile IDs can include an empty bootstrap bin.

These are demonstrated code paths, not claims that published runs used the problematic options. SOURCE_AUDIT.md gives the static findings, exact code locations, caveats and reuse decision. The original MIT source/license is retained; no upstream modification or issue was submitted.

## Scientific implications

The benchmark's six leave-one-fragment-out folds correspond to four conservative parent groups in our ledger. Four folds retain a same-group sibling in training. We therefore cannot treat the six scores as six independent new-parent replications. Its whole-fragment normalization is transductive input adaptation, and its gradient-surface labels are derived references. Neither necessarily makes a descriptive experiment useless; both limit the claim.

We can use supplied labels for explicitly named supplied-reference agreement once correspondence/support is bound. They do not become independent optical truth merely because a public scorer accepts them. Full optical registration, negative-support semantics and physical render/depth correspondence remain unresolved for the preferred Frag6 reference. The public material supplies implementation details, not those missing artifact-specific records. We have not shown that the records are unavailable elsewhere.

**Next: construct one bounded, documented reference panel and execute the already fixed two-arm depth-window comparison when its essential input contract is complete.** Do not add another general mechanics phase. NEXT_STUDY_DRAFT.md specifies how supplied-reference diagnostics and the physical-depth hypothesis differ, what can proceed without a maintainer reply, and what must remain qualified. The original fixed comparison remains unchanged forward seed42 versus equal averaging of three neighboring forward windows; it is a comparator, not a novelty claim or trained Plinius model.

## Evidence and limits

Ten exact public source blobs were fetched:10 HTTP 200 responses,75,837 body bytes,1,310,730 bytes reserved. No model, image, label or CT artifact was fetched. Tree191126e0e677b4cf17131a1f7bbaefb1fa8f92e1 is a Git tree, not an asserted commit. acquisition-v1/SOURCE_RECEIPT.json binds all source identities. SOURCE_FREEZE.json and SOURCE_FREEZE_V2.json preserve the pre-execution contracts; run-v1 and run-v2 preserve both outcomes. Root performed source and report review; no independent-agent review is claimed.

No new candidate pixels, local candidate scores, training, model forwards, transcription, translation, paid resources, R2 uploads, publication or prize submission. Frag6 Stage C remains closed, as do existing scoring reserves. Six reused Paris contexts remain development on two conservative parent groups. Maintainer request remains user-confirmed NOT SENT; no outreach was made. Public aggregate results were seen previously, so future claims cannot assert blindness to all published outcomes.

Repository HEAD49fabf3e890d5b880f391b27d581f1e7a1ebdbcc and pre-existing untracked.codex/ are preserved; application source/tests/dependencies are unchanged. All new research files are outside the checkout. GPU spend remains£0 against the£10 total pilot cap. No GPU launch is ready on the strength of these scoring tests.
