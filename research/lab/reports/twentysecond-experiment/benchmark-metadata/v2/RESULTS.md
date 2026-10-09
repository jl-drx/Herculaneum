# PHerc0841 public label package audit

The separately authorized v2 acquisition completed all six exact requests: **59,771 response-body bytes, six HTTP 200 responses, 4.885 seconds**. Each of the three public label prefixes contains a complete 39-object package for ink, supervision and validation. **None contains the advertised mapping-validity asset or linked `provenance.json`.** This closes package discovery but leaves the parent unsuitable for an independently authenticated accuracy claim.

| Segment | Source annotation entry | Complete inventory | Literal root `canvas_size` | Declared source-canvas offset YX |
|---|---|---:|---|---|
| `20260221022814-auto_grown_20260220174252405` | `841/auto_grown_20260220174252405` | 39 objects | `[14660, 19100]` | `[29.5, 49.9]` |
| `20260220213127-w00` | `841/w00` | 39 objects | `[16460, 18560]` | `[32.1, 79.4]` |
| `20260220214732-auto_grown_20260220144552896` | `841/auto_grown_20260220144552896` | 39 objects | `[18080, 18400]` | `[0.0, 0.0]` |

The `canvas_size` pairs above are retained literally; this audit does not establish their axis order or actual array dimensions. All three ink-root records declare Y/X OME axes, six pyramid levels, level-2 scale `[9.612, 9.612]` micrometers and scale-only coordinate transforms. No array-level metadata, shard or label values were fetched. Actual array shape/dtype/value encodings, supervision/validation overlap, negative-label semantics and depth relation remain unverified.

The 39 keys are exactly three roles × (one root metadata object + six level metadata objects + six shard objects). Every response has `IsTruncated=false`, the expected bucket and exact prefix. Every fetched ink-root size and ETag matches its corresponding inventory record. The [findings](PACKAGE_FINDINGS.json) retain per-body SHA256s; the [request ledger](run-v1/request-ledger.json) records exact URLs, timing, headers and body counts. No pagination or additional asset requests occurred. Absence applies to these exact prefixes at the recorded time, not every private or alternative location. This audit did not issue direct requests for the absent paths and does not claim new HTTP 404 observations.

Each root names publisher commit `bfef6abe073510ee8b595da390ed1d639316abc8-dirty` and links `provenance.json`, which is absent anywhere under its complete package prefix. The `source_image` and `transfer_report` fields are publisher-local filesystem paths, not authenticated public download endpoints. These metadata references do not supply the missing publication command, dirty changes, source/target hashes or any final post-transfer correction.

The prior [inventory](../../benchmark-inventory/RESULTS.md) remains relevant: all three records are transferred and prediction-reviewed, and one explicitly describes reconstructed letter-form annotation. Finding separate S/V packages does not authenticate independent optical truth or the reviewing model's ancestry. The absence of an OME translation is not a registration measurement and does not authorize reapplying the catalog's source-canvas offsets to target labels.

## Transport revision and preserved failure

The first implementation explicitly disabled configured proxies and stopped with `URLError`, before any HTTP status/body. That [failed attempt](../RESULTS.md), its code, fixtures and receipts remain unchanged under snapshot seal `19b71304ca855785fe8c11929c067aa50cc7b40b6fb43bea9d49689ead30f1a8`. Its suppressed underlying exception reason remains a diagnostic limitation; no exact network cause is reconstructed after the fact.

Root then confirmed configured proxy availability and authorized an additive transport revision. V2 uses the workspace's configured proxy while retaining verified TLS, exact public destination URLs, redirect rejection, body caps and request deadlines. Proxy values are never printed or retained. Thirty-four focused fixture checks passed before source freeze `e62850af8e3cfa74d2ff52b445e67485755b320b8eca0ae857cd800ffec63f7a`; the new root GO hash is `e19ac762fcac5727a5191b2ff80ae784c1caad70cd3ad04d7f2b313f6e2e292e`.

The acquisition has **seven total attempts across two separately authorized runs**, including the first zero-byte failure. There was no automatic retry loop; v2 requested each of the six frozen URLs once. Its 59,771 bytes remain below the 294,912-byte bound; each request had a 30-second absolute timer and the controller imposed a 210-second external wall limit. All six succeeded without further revision.

No arrays, optical images, CT, weights, model predictions, models, fitting, credentials, uploads, outreach or paid resources were used. The sealed inventory and earlier evidence were rechecked unchanged. GPU spend remains £0, with the £10 total pilot cap intact. Root retains the next research decision; this audit authorizes no further acquisition.
