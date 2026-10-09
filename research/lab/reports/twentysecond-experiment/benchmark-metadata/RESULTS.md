# PHerc0841 metadata attempt: transport STOP

The authorized six-query package audit **stopped on its first request before receiving an HTTP status or response body**. The worker recorded `urllib.error.URLError`, observed **0 bytes**, and exited without a retry. The other five requests were not attempted. No new package-completeness, mapping-validity, provenance or registration conclusion is available.

The sole attempt was the exact frozen prefix-inventory URL for segment `20260221022814-auto_grown_20260220174252405`, at 2026-10-09 05:09:07.950283 UTC. It stopped at 05:09:12.984200 UTC. The request reserved 65,536 bytes; the entire six-request contract allowed 294,912 bytes. The external controller completed in 5.077 seconds, below its 210-second bound. The [request ledger](run-v1/request-ledger.json), empty body hash, worker output and [controller receipt](controller-result.json) are retained.

Before the first request, 28 focused checks passed for URL scope, framing/body limits, XML completeness, JSON group schema, duplicate-key rejection, partial-body retention and disabled redirects. [SOURCE_FREEZE.json](SOURCE_FREEZE.json), SHA256 `6b44bf09a744e65819a592fe20b7b366b406f911e90f2def25ee108b6a5f24b1`, binds the implementation and fixture result. The worker verified the pinned root GO and unchanged query proposal before starting.

The frozen reader explicitly constructed `ProxyHandler({})`, selecting direct transport. Its non-contract-exception logging retained the exception class but suppressed the underlying exception message. **The exact transport cause is therefore not preserved and cannot be diagnosed from this receipt.** A proxy requirement, DNS problem, TLS problem or remote availability must not be asserted as the cause. This is a diagnostic limitation of the implementation. No external destination/body was substituted, no proxy policy was bypassed and no transport change was attempted after the failure.

The root GO prohibited retries. Any revised transport implementation or renewed acquisition requires a separate root decision and an additive source freeze; this run and the sealed inventory must stay unchanged. Public metadata assets were not shown absent by this failure: absence requires actual complete inventories or relevant HTTP results.

The [prior inventory](../benchmark-inventory/RESULTS.md) remains the current evidence. No candidate pixels, labels, weights, predictions, model calls, paid resources, credentials or uploads were used. GPU spend remains £0.
