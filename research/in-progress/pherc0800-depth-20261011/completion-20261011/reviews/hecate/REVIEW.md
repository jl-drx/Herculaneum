# Hecate fixed-map review

Reviewer: worker-recovery agent. Reviewed only after all 32 shifted maps and
16 saved d0 baselines were restored and their completion checks passed. All
Hecate images use the frozen 0–255 display and common support. No fitted
contrast, additional crop, model inference or depth search was used.

Viewed the Hecate overview, all four `full-strips/hecate-*-full.png` panels and
all twelve `cells/hecate-*-cell[123].png` panels in `completion-review`. The
exact cell panels preserve all 512×512 source pixels. Each shows d−15, d0 and
d+15, so every Hecate phase/order/depth combination and every fixed cell was
examined. The full-panel tool display was slightly reduced by the UI; the
512-wide exact cell panels provide the unreduced inspection.

## Fixed cell observations

| Cell | d−15 versus d0 | d+15 versus d0 | Interpretation within Hecate only |
|---|---|---|---|
| 1, x512:1024/y1024:1536 | Forward has an upper-right compact bright response and a faint vertical continuation; reverse has mostly faint separated patches. Several d0 patches occur elsewhere. | Strong new lower/middle broad connected response in forward; reverse produces a different broad curved/branched configuration. Each configuration substantially recurs between plain and padded images of the same input order. | A real change in model response and useful padding agreement. I cannot identify a clear stroke group or text-like sequence in this broad mottled structure from Hecate alone. Reversal differences do not invalidate it, because invariance was not required. |
| 2, x1024:1536/y1024:1536 | Forward shows separated compact patches, including a bright lower-left patch and several upper patches. Reverse has weak separated responses. d0 does not resolve a group. | Forward has a chain of disconnected left-edge responses and scattered spots; reverse has a broad bright oblique patch in the lower-left and several isolated upper responses. Their source locations generally recur across padding. | Repeated responses, but no clear sequence of connected strokes. Some structures are cut by the predefined cell boundary and were also inspected in the complete strip. |
| 3, x1536:2048/y1024:1536 | Reverse has a bright isolated patch in the middle-upper area and scattered responses; forward is sparse. d0 is also sparse at different locations. | Reverse contains a reproducible broad horizontal bright patch near the upper-left, plus separated spots. Forward remains mostly faint and disconnected. | A repeated isolated feature is insufficient for the predeclared group criterion. I found no defensible text-like group. The brown upper-right corner remains excluded. |

The largest visual change is therefore d+15 in cell 1, with part of its broad
response extending into cell 2. That observation is not a preferred depth,
ink discovery, surface-placement estimate or validation of a neighbouring
wrap. The response's stronger confidence does not establish improved recovery.

My Hecate-only judgment is that these images do **not clearly establish** the
coherent-stroke/text-like part of the scientific criterion. The broad new
cell-1 response should still be compared against the independently reviewed
Villa maps before the root reviewer makes the overall decision; no
cross-family claim is made in this note. A negative three-position screen
cannot show that the papyrus contains no ink or that intermediate depths fail.

No existing study52 report, source, input, output or checkpoint was modified.
