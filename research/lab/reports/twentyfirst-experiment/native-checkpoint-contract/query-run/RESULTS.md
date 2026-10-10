# Native safetensors header result

**The published native checkpoint has the same named tensor schema as the retained phase 6 factory.** The query returned all 508 expected F32 names and shapes, totaling 68,175,426 named elements and 272,701,704 tensor-data bytes. Its 66,192-byte JSON header declares disjoint, contiguous intervals covering that exact data extent, with no metadata object. All 244 expected factory alias-name pairs are present with matching shapes and separate serialized intervals. This closes the apparent parameter-count mismatch at the serialized-schema level.

Collapsing the retained factory's declared alias-name pairs gives a conditional 264-group / 34,546,498-parameter / 138,185,992-byte projection. **The header does not prove that the paired native tensor values are equal or share storage.** It also cannot verify the full-file hash, finiteness, strict model loading, historical training execution or predictive behavior. The result status is `HEADER_SCHEMA_COMPATIBLE_VALUES_AND_LOAD_UNVERIFIED`.

The immutable public source is [Nieuwlaar/ink9um-dense-native, revision b894a0493ab2291cc8dc3247020c0d7cb0967a58, weights/dense_native_016000.safetensors](https://huggingface.co/Nieuwlaar/ink9um-dense-native/resolve/b894a0493ab2291cc8dc3247020c0d7cb0967a58/weights/dense_native_016000.safetensors). Retrieval ran on October 9, 2026, from 00:44:50 to 00:44:54 UTC. Both HTTP206 responses reported file extent **272,767,904 bytes** and the same strong ETag. The advertised full SHA256 `d3d95dc41dcb24ed76b84b5e12ca5dae25d7fb34edf1dfbb93e0186f91e6fc6d` remains a registry/author claim; no full-file check occurred.

| Response | Observed body bytes | Retention |
|---|---:|---|
| Hugging Face HTTP302 to the allowed CDN host | 1,106 | Count and hash only |
| CDN HTTP206, bytes 0–7 | 8 | Length prefix |
| CDN HTTP206, bytes 8–66,199 | 66,192 | JSON header |
| Total | **67,306** | Three HTTPS attempts |

The prefix SHA256 is `b94f491fdf06f1bf3f8592ca9c5c3f575fa9ca8464f76296559b94a752f1271f`; the header SHA256 is `b58a17c22694c0083156de30f02cb58838777eb4716fdb56e76a95d5b2962a24`. The [result receipt](result.json) retains exact sanitized response validators and byte accounting. It contains no signed transport URL or raw redirect/error body. Its two raw-object path fields are relative to the implementation working directory; the final manifest binds their absolute paths and hashes.

Implementation and fixtures were sealed before the query. All 22 synthetic fixtures passed; a narrow independent static review caught and confirmed correction of partial-read accounting before execution. A separate [offline arithmetic check](OFFLINE_CHECK.json) rechecked the saved header, all schema pairs, interval coverage, response totals and all 14 recorded input bindings without importing the reader. Inputs remained unchanged before and after the query. There were zero retries, tensor bytes requested, model calls, scientific-array reads, provider actions or paid resources.

The existing `.pth` storage guard cannot simply treat the 508 safetensors intervals as shared factory aliases. Any future actual loading requires a separate approved value-read contract, full-file verification, explicit equality checks for all 244 alias pairs, and reviewed format-aware canonical binding before strict loading. No partial loading or last-alias overwrite has been performed. The separate frozen allocation study is unchanged.
