# Retry: checkpoint publication recovered, source intake still partial

12 October 2026, following the user's request to retry after time had elapsed.
GitHub Git/API reads and the full previously failing teacher range passed. The
completed-original source commit `94556cd377f53aa8e976c67a984cc1fd719d871d`
and its 344,986,026-byte cumulative archive were then pushed to private GitHub.
A fresh selective LFS download into an empty cache matched the byte length and
SHA256 of the archive already used for the successful local numerical replay.
[The remote receipt](ORIGINAL_CHECKPOINT_REMOTE_RESTORE.json) confirms that
backup is now available; this is not a second-machine scientific review.

The unchanged four-reader intake resumed from all 1,522 verified added receipts.
It retained **72 more complete teacher ranges**, totalling **8,896,888 bytes**,
before the same HTTP 503 failure returned. Each has its frozen request, ETag,
length and SHA256 receipt. The initial 129,096-byte diagnostic request was not
cached, so it is explicitly recorded as extra transfer; its hash agrees with
the subsequent cached object. No source was replaced or recorded as missing.

Current added-source state:

- **1,594 complete objects:** 1,403 CT chunks and 191 teacher ranges.
- **665,963,655 bytes** retained.
- **15,976 pending requests**, with 5,824,472,581 nominal body bytes remaining.
- All 781 original CT objects and the verified 876 original candidates remain
  unchanged. The new teacher ranges have not been decoded or qualified.

[STATUS.json](STATUS.json), [RETRY_PLAN.json](RETRY_PLAN.json), the failure record
and log preserve the current attempt. Earlier fail-fast acquisition does not
retain every failed worker's request ID; do not interpret these as a complete
per-attempt network trace. The byte-verified successful objects and exact pending
request manifest remain available for resumption.

This establishes successful checkpoint publication and further source progress,
not reliable bulk access or full-cohort admission. The four-reader attempt has
stopped. No model forward, training, new evaluation, GPU/paid resource, Actions
job, external message or spare-VM assignment occurred. Keep the fixed population
and original-source exposure rules; no availability-selected substitute study.

The new source bytes are preserved as a small private addendum described in
[PORTABILITY.md](PORTABILITY.md). Its local restore and fresh selective GitHub
download both passed; [REMOTE_RESTORE.json](REMOTE_RESTORE.json) records the
published commit and verified bytes. After a meaningful recovery, resume the same
remaining requests, finish full teacher/native support and class qualification,
then freeze the controlled workload. Do not repeat the already failed persistent
connection diagnostic as a demonstrated fix.
