# Native checkpoint header contract

The retained metadata makes the apparent parameter-count mismatch explainable: **68,175,426 is exactly the phase 6 factory's count with duplicate parameter names**, while **34,546,498** counts unique parameters. Phase 6 recorded 508 names, 264 unique storage groups and 244 duplicate-name alias groups. All four inspected phase 6 states shared that schema. This is evidence for a serialization explanation; the Nieuwlaar tensor header and values remain uninspected.

The retained native configuration exactly matches seed42 in `mode`, `model_type`, `model_config`, `targets`, `in_channels`, `patch_size` and `image_normalization`. Other training/source/runtime fields differ. Configuration equality does not prove that the published file used that factory. [SOURCE_PINS.json](SOURCE_PINS.json) retains the exact compared fields, file hashes and primary source URLs. [REFERENCE_SCHEMA.json](REFERENCE_SCHEMA.json) projects all 508 expected names/dtypes/shapes and the 244 factory alias groups from the saved phase 6 JSON. No model was instantiated or checkpoint value read.

| Count or extent | Retained value |
|---|---:|
| Named F32 elements | 68,175,426 |
| Named tensor data bytes | 272,701,704 |
| Unique factory parameters | 34,546,498 |
| Unique factory storage bytes | 138,185,992 |
| Advertised native safetensors extent | 272,767,904 |
| Difference after tensor bytes and eight-byte prefix | 66,192 |

The final row is the **conditional expected header length**, not a measured header. A single guessed combined range could cross into tensor values if that expectation is wrong; the proposal instead reads the eight-byte length first.

## Exact proposed public query

The immutable source is:

`https://huggingface.co/Nieuwlaar/ink9um-dense-native/resolve/b894a0493ab2291cc8dc3247020c0d7cb0967a58/weights/dense_native_016000.safetensors`

The retained Hugging Face LFS registry and author `weights/VERIFY.md` agree on **272,767,904 bytes** and claimed full-file SHA-256 **d3d95dc41dcb24ed76b84b5e12ca5dae25d7fb34edf1dfbb93e0186f91e6fc6d**. A header-only retrieval cannot verify that full-file hash.

The [query plan](HEADER_QUERY_PLAN.json) proposes two logical GETs: `bytes=0-7`, then exactly `bytes=8-(7+H)` after decoding unsigned little-endian H. Require verified TLS, identity encoding, HTTP206, exact Content-Range and Content-Length, the advertised total extent, and a strong matching ETag with If-Match for the second read. HTTP200 from an ignored Range stops before consuming its body. No full-object fallback, automatic retry or automatic redirect occurs.

The bound is **six HTTPS attempts including redirects/failures**, at most four redirect responses, and **262,144 observed response-body bytes**. Each redirect/error body is limited to 4,096 bytes. Maximum H is 245,752, leaving room for the eight-byte prefix and all four maximal redirect bodies. If the advertised schema and extent match, successful prefix/header bodies total 66,200 bytes; with four maximal redirect bodies, 82,584 bytes. Headers, TLS framing and unread socket buffers are outside that observed-body accounting.

The exact transport-host allowlist is **huggingface.co** and **us.aws.cdn.hf.co**, supported by the retained successful phase 6 transfer receipt. The actual native-file redirect route is unverified. An unexpected host, expired destination, unbounded body declaration, missing validator or exhausted attempt allowance stops with retained sanitized evidence; it does not expand the allowlist. Reuse the resolved prefix transport URL only transiently in memory. Never print or retain signed Location queries, response URLs or raw exception text. Retain the canonical pinned source URL, target hostname, ranges, response validators, byte counts and hashes of bounded bodies.

## What header comparison would establish

Independently parse UTF-8 JSON with duplicate-key rejection, then compare all 508 exact names, F32 dtypes, shapes, element counts and byte lengths against the retained reference. Reject malformed offsets, overlapping ranges, holes, trailing data or extent/cardinality mismatch. Data offsets are relative to the tensor-data start; no tensor interval is requested. Optional string metadata is reported separately against the author's claim of no metadata.

For each expected factory alias group, require both names and their expected matching schema. The serialized ranges must remain disjoint. Safetensors' separate serialized intervals do **not** establish shared storage or equal tensor values. Collapsing the known factory name groups yields a conditional 264-group/34,546,498-parameter projection; it is not an observation of the native file's aliases. The strongest successful status is **HEADER_SCHEMA_COMPATIBLE_VALUES_AND_LOAD_UNVERIFIED**.

The header cannot prove tensor finiteness, equality of duplicated alias values, strict load, semantic state, prediction behavior or historical training/exposure. The existing phase 6 `.pth` helper requires its verified shared-storage partition; that requirement cannot be applied unchanged to 508 separately serialized safetensors entries. A future full-value audit would need an independently budgeted full-file SHA check, all alias-value equality checks, and reviewed format-aware canonical binding before strict model load. Silent last-alias overwriting or partial loading would hide a defect.

## Retained source anchors

- Phase 6 [diagnostic code](/workspace/herculaneum-lab/reports/sixth-experiment/model-contract/diagnose_storage.py:45) records named and unique counts; lines114–116 distinguish unique parameters and duplicate names. Its saved `storage-diagnostic.json` contains the full schema and groups for seed42, seed43 and both Russo states.
- Phase 6 [verified storage guard](/workspace/herculaneum-lab/reports/sixth-experiment/model-contract/model_contract_v2.py:128) fixes 508 names /264 groups, with 244 alias groups at line149; the corresponding audit JSON reports strict schema equality across all four states.
- Nieuwlaar [VERIFY.md at code217e752](https://github.com/Nieuwlaar/ink9um-dense-native/blob/217e7521670db49cde97872ea6a8bdfb80c449f1/weights/VERIFY.md) lines3–14 supplies the bare safetensors counts, extent and hash. [Reproduction](https://github.com/Nieuwlaar/ink9um-dense-native/blob/217e7521670db49cde97872ea6a8bdfb80c449f1/training/REPRODUCE.md) names historical Villa training revision `3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e`; our retained factory source pin is different.
- Retained HF registry pin **b894a0493ab2291cc8dc3247020c0d7cb0967a58** names the exact `weights/dense_native_016000.safetensors` LFS identity. These are registry/documentation claims until the requested bytes are retrieved.

This is a frozen query proposal awaiting root's written GO and a bounded implementation/fixture check. No public request, tensor payload, model instantiation, source-array read or change to the separate allocation study occurred.
