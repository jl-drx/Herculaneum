# Completed PHerc0800 depth-screen evidence

All32 shifted maps and16 baseline views are complete. No candidate met the frozen
stroke-group criterion. The repeatable broad left-cell response at d+15 remains
uncertain. Read [FINAL_RESULTS.md](FINAL_RESULTS.md), [NUMERIC_SUMMARY.md](NUMERIC_SUMMARY.md),
the individual reviews, and [PORTABILITY.md](PORTABILITY.md).

The bounded packet contains all48 saved aligned maps, support, exact review
panels, comparison metrics and a read-only portable checker. The larger private
checkpoint chain is only needed for the full raw-logit/receipt recovery. No
inference, GPU or original workspace is required to reassess saved-map metrics.

The old live-status tool required an unbacked local process record and initially
failed after restoration. It has now been repaired to read completed bound
receipts without claiming a live worker. Four integration checks cover complete
restoration, a missing map, duplicate identities and absent files. The dated
initial failure remains in recovery notes; STATUS_AFTER_TOOL_REPAIR.json records
32/32 complete with no local worker identity. This does not explain or predict
platform restarts. The original scientific worker and frozen run are unchanged.
