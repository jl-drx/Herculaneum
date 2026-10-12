# Publication recovered; more input retained and first case qualified

**This new update is local only.** Its archive, independent verifier and fresh
production replay pass, but the normal Git push returned proxy-tunnel HTTP 503.
Git ref reads also failed; remote availability of this archive is not established.
[PERSISTENCE.json](PERSISTENCE.json) distinguishes it from the parent checkpoint,
which was successfully published and remotely verified earlier this turn.

12 October 2026. The previously blocked 276.71 MiB checkpoint was published at
`2369f0db26cf0ecd2e1d278908f1bd6fc58c4c5e` and a fresh selective LFS download
matched its exact length and SHA256. [The parent receipt](PARENT_CHECKPOINT_REMOTE_RESTORE.json)
records this, separately from the new update's publication status.

The unchanged four-reader source retry retained **1,009 additional records**:
898 CT chunks and 111 teacher ranges, **423,669,085 bytes**. HTTP 503 returned
at 23:20 UTC and the reader stopped. Every successful body and receipt remains
verified; no source was replaced, treated as absent or filled with zeros.

- **3,515 records present:** 2,909 CT chunks and 606 teacher ranges.
- **1,406,657,824 bytes retained.**
- **14,055 requests pending**, with 5,083,778,412 nominal body bytes remaining.
- Segment `20250925161630` has all 1,881 requested records. The second segment's
  CT is complete, but 399 of its teacher ranges are still pending.
- All five original PHerc0139 streams and their 876 qualified candidates remain
  unchanged. The frozen request manifest and full eight-segment cohort are intact.

The available first segment enabled the already-planned offline
[teacher/support qualification](../native-added-qualification-20261012/README.md).
All 1,290 fixed candidates meet the existing support rule, both TIFF decoders
agree on 483 tiles, and independent area arithmetic covers every crop. A separate
full-raster CT verifier reproduces the saved masks, depth counts and class/UV
accounting. The other seven cases remain unresolved; this is no substitute cohort,
registration certificate, model result or paid-work authorization.

[STATUS.json](STATUS.json) preserves all eight source states, and the dated
failure/log preserve the failed continuation. Final counts come from receipts,
not the lower fail-fast progress counter. The reader does not retain a complete
per-request trace for every failed future; the one-byte preflight is separate.

The [update archive](PORTABILITY.md) is a delta against the **combined 21:47
restore**, which already contains both earlier source retries. It retains the new
bytes and the complete partial-qualification packet. Preserve both dated original
qualifications; finish the intended population before drawing a training decision.

[CACHE_CLEANUP.json](CACHE_CLEANUP.json) records removal of six older generated
restore copies after rechecking their full manifests and published archive
receipts. This recovered 4.26 GB. The live source cache, newest combined restore,
all archives/LFS objects and scientific receipts remain; unpublished new bytes
were not removed. Available workspace space is approximately 7.3 GiB afterward.
