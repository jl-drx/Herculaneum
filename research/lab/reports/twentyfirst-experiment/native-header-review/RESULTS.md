# Native checkpoint retained-header review

**PASS: header schema compatible; tensor values and model loading remain unverified.** The independent review parsed the retained safetensors prefix/header, rebuilt the reference projection from the original phase 6 factory JSON, and checked the complete published schema and interval accounting. It performed no network request, model construction or tensor-value read.

The eight-byte unsigned little-endian prefix declares a 66,192-byte JSON header. Tensor data therefore begins at file offset 66,200. All successful requested ranges stop at 66,199; no tensor-data region was requested or retained. The header has no metadata object and exactly 508 F32 tensor names, each with the expected shape and byte length. Its serialized intervals are disjoint, contiguous and cover exactly 272,701,704 bytes, giving the advertised total extent of 272,767,904 bytes including prefix/header.

| Quantity | Independently verified metadata result |
|---|---:|
| Named F32 elements | 68,175,426 |
| Separately serialized tensor entries | 508 |
| Declared factory duplicate-name groups | 244 |
| Factory singleton groups | 20 |
| Conditional factory groups | 264 |
| Conditional canonical parameter count | 34,546,498 |
| Conditional canonical storage bytes | 138,185,992 |

The larger named-element count is exactly compatible with serialization of all duplicate factory names. Each expected pair has matching dtype/shape and its own serialized interval. This resolves the apparent count mismatch at the schema level. It does **not** show that the corresponding native tensor values are equal, share storage or can safely be collapsed. A later loader must explicitly validate equality before canonical binding; silently overwriting aliases would hide a mismatch.

The retained transport receipts account for three HTTPS attempts: a 1,106-byte redirect response, the eight-byte prefix and the 66,192-byte header, totaling 67,306 observed response-body bytes. Exact ranges, file extent and the same strong ETag match; budgets remain below six attempts, four redirects and 262,144 body bytes. Redirect-body bytes were deliberately not retained, so this review can verify their receipt accounting but cannot independently rehash that discarded body. Signed URLs were not retained. Headers, TLS framing and unread socket buffers are outside the observed-body accounting.

The independent checker passed 2,616 scalar and identity assertions in approximately 0.021 seconds, with 24,764,416 bytes peak RSS. All source/run pins checked against the sealed inputs. Its strict JSON parser rejects duplicate keys and nonfinite JSON constants; all shape/count/offset comparisons use exact integer arithmetic.

**Still unverified:** the claimed full-file SHA-256, tensor values and finiteness, alias-value equality, shared native storage, strict model load, semantic state, forward behavior, historical training/exposure and scientific performance. A matching strong ETag supplies transport continuity here; it is not verification of the advertised full-file SHA. The status remains `HEADER_SCHEMA_COMPATIBLE_VALUES_AND_LOAD_UNVERIFIED`.

`verification.json` retains all checks and source anchors. `conditional-alias-projection.json` records the complete 264-group factory-name projection and each separate serialized interval, without any tensor values. No full-weight acquisition or model execution is authorized by this review.
