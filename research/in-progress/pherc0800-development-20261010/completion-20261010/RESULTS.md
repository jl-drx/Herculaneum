# Study51: completed PHerc0800 centre-depth development comparison

All 16 fixed maps completed at **22:54 BST on 10 October 2026**, after 3 hours
36 minutes of CPU execution. Numerical completion checks pass. Review of the
whole field and every fixed region does **not establish defensible lettering**.
This is an inconclusive centre-depth transfer result, not evidence of no ink.
No model, depth order or padding phase is promoted. GPU spend remains **£0**.

The run covers segment20251028222030, x512:2048/y512:2048, original geometry,
66 rendered planes, four models, both padding phases and both array depth orders.
Comparisons use only the fixed 1,875,648 common supported pixels (79.50% of the
field). This is reused development material without ink ground truth. Model
response brightness, correlation and visual appearance cannot provide AP or
authenticate text.

## Execution and verification

[INFERENCE_RESULTS.json](INFERENCE_RESULTS.json) records 3,454 committed model
batches and 13,250 model tiles. All 16 original-accumulation saved-logit replays
were exact. The final read-only check verified 958 frozen input/runtime/model
pins, every batch/logit receipt, all output hashes, identities and shapes, support
masks, and saved numerical conversions. It also recomputed the 1,043,328 retained
renderer comparisons: 99.64134% exact, mean absolute difference 0.0035866 and
maximum difference 1 uint8 count. See
[COMPLETION_VERIFICATION.json](COMPLETION_VERIFICATION.json). Review ran no new
model forwards. It verified the recorded replay evidence and saved conversions;
it did not independently rerun the complete neural inference.

The final checkpoint was published at 22:55 BST to the private checkpoint ref,
commit `dfb4fa95e0328730757d38a1711e2ebbe6a56850`. Its 23-generation index covers
7,504 immutable files, including every batch/logit, all 16 map receipts and the
final results. All indexed local files matched their hashes. A fresh remote
download of the final generation passed archive and all 172 member checks.
Generations 0 and 1 had separately passed remote download and restoration earlier;
this review did not freshly download/restore the whole chain. Prior model/runtime
prerequisites remain separate. See
[FINAL_BACKUP_VERIFICATION.json](FINAL_BACKUP_VERIFICATION.json).

## Descriptive findings

[REVIEW_PLAN.md](REVIEW_PLAN.md) was written before opening maps or calculating
these comparisons. All 40 planned pairs are reported in
[MAP_COMPARISONS.json](MAP_COMPARISONS.json). The table gives Pearson correlation
on the same common support. Forward/reverse are array order, not independently
verified reading sides.

| Model | Padding, forward | Padding, reverse | Order, plain | Order, padded |
| --- | ---: | ---: | ---: | ---: |
| dense_native | 0.901 | 0.887 | 0.166 | 0.161 |
| seed42 | 0.682 | 0.622 | 0.124 | 0.096 |
| seed43 | 0.625 | 0.676 | 0.092 | 0.122 |
| Hecate | 0.836 | 0.869 | 0.102 | 0.077 |

Padding changes have mean absolute differences of 3.21–8.49 uint8 levels;
depth-order changes have differences of 9.35–13.03. The models are not required
to be invariant to depth reversal, so low order agreement does not itself show
a bug or failed model. It does prevent treating either ordering as a verified
reading-side choice on this evidence. Greater padding correlation does not
establish greater ink accuracy. The field and support differ from the small
study50 core and PHerc343; their correlations are not a controlled performance
comparison.

At fixed 0–255 display, Villa maps mainly show diffuse patches; Hecate has a much
darker background with localized bright responses. Some locations persist across
padding, while strength and distribution change markedly with depth order.
Neither the complete field nor the nine fixed region panels provides a
defensible sequence of letters. This is a qualitative model-assisted review,
not papyrological authentication or a quantitative absence test. See
[VISUAL_REVIEW.md](VISUAL_REVIEW.md), [all maps](ALL_MAPS.png) and
[CT context and support](CT_AND_SUPPORT.png).

## Decision and next question

The fixed centre-depth comparison is complete; repeating it or selecting the
brightest model/order is not justified. The nominal surface/window remains an
unresolved physical variable. A useful next step is to design one bounded
depth-placement comparison using the already retained 66-plane data and fixed
geometry-defined regions, with both input orders retained. The current result
does not prove an offset is the cause; domain mismatch, missing recoverable
signal and model limitations remain alternatives.

Before any such run, specify offsets and their physical centres for both model
families, require unchanged source/context support, fix a small comparison
budget and review every result at the same scale. Include a predeclared stopping
rule: do not expand a depth search merely because some maps are brighter. A
structurally credible result would need repeatability under padding and separate
review before being called candidate writing. Geometry or model consistency
alone is not ink truth. No depth study, training, new GPU job or spare-VM task
was launched by this completion review.

The compact completion packet is separately checksummed. The cumulative sealed
research register still ends at study50; study51 completion must not be confused
with an updated full-history archival seal. Frozen launch evidence and prior
sealed studies are unchanged.
