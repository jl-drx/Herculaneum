# Retry retained 912 more source records; full intake still incomplete

12 October 2026. The user's requested retry passed its one-byte source probe
and resumed the unchanged four-reader intake. It retained **912 new records**:
608 full CT chunks and 304 teacher ranges, totalling **317,025,084 bytes**.
The retry stopped when HTTP 503 returned at 21:48 UTC. The frozen request
manifest, eligibility rules, source identities and saved receipts are unchanged.

Current added-source state, verified against every receipt and body hash:

- **2,506 complete records:** 2,011 CT chunks and 495 teacher ranges.
- **982,988,739 bytes** retained; zero confirmed missing sources.
- **15,064 pending requests**, with 5,507,447,497 nominal body bytes remaining.
- All requested source bytes for the first PHerc0814 segment,
  `20250925161630`, are now present. The other seven remain incomplete.
- All 781 original CT objects and 876 qualified PHerc0139 candidates are unchanged.

[STATUS.json](STATUS.json) includes all eight segment counts. This is source
completion for one segment, not teacher decoding, physical registration,
supervision qualification or admission of a smaller training cohort. Preserve
the intended eight-segment added stream and five-winding original stream.
No new teacher values, model predictions or reserved evaluation payloads were
inspected, and no training, paid compute, Actions job or VM assignment started.

The final failure's progress counter is lower than the verified cache count:
the fail-fast reader stops consuming futures after an error, while completed
receipts remain. Counts above come from the immutable request list and retained
receipts, not from the last progress line. The one-byte preflight is recorded
separately. The existing reader does not retain a complete per-request trace for
every failed future; do not claim otherwise or equate HTTP 503 with absent data.

The environment reports a running instance, current observations, no provider
failure and an unrestricted configured HTTP policy; enforcement state is unknown.
GitHub private-repository metadata and the current branch read succeeded.
These observations do not identify the internal cause of the source HTTP 503.
Proxy routing, TLS checks, credentials and cloud configuration remain unchanged.

[PORTABILITY.md](PORTABILITY.md) describes the cumulative source-cache update.
It includes the earlier 20:53 addendum, so use this update in its place, applied
to the completed-original restore. Both dated qualification results remain intact.
After source access recovers, resume the exact pending requests and finish
population-wide decoding/support/class checks before admitting the matched pair.
Do not repeat closed transport diagnostics as established fixes or select a
replacement cohort from the subset currently available.
