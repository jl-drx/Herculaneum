# Additive pooling JSON correction review

GO for the exact V2 seal recorded in code-go.json and the root POOLING_JSON_CORRECTION.md addendum. The original attempt stopped at metadata comparison before numeric CT decoding and produced no pooled arrays. Its source, plan, GO, failure receipt and empty output directory remain preserved.

Independent metadata-only reproduction on frag1-z07 found tags as the sole differing field: integer parser keys versus string keys after JSON serialization. All layout fields match after projecting those keys. The author's separately authorized metadata replay verifies the same exact corrected comparison on all153 objects. No source bytes, tag values, offsets, sample formats or interpretation changed.

The reviewed helper requires reconstructed tag keys to be integers, projects only those keys with str(k), then compares the complete tag map and all other reconstructed layout fields exactly. The worker replaces only the failed comparison with this helper. Numeric kernels and full provenance validators are byte-identical to the original seal. Necessary package pins/addendum lineage and a guard counting the original stopped package in combined storage are additive. The shared production epoch and deadline, resource caps, exact rows/planes, labels/masks and descriptor readiness gates are unchanged.

Validation:20 independent synthetic cases PASS across both byte orders, with asymmetric inline SHORT padding. They reproduce the original JSON mismatch, accept the exact saved representation, and reject changed tag values/counts/raw bytes, missing or extra tags, changed byte order/strip offset and noninteger parser keys. The final author receipt records14 additional layout fixtures and153 exact metadata-only comparisons; retained22 kernel and24 provenance cases pass. All V2 seal records and independent fixture source hashes were verified. No broad scientific replay was performed.

The GO authorizes the root-approved fresh offline attempt under its frozen plan; it is not evidence of completed pooling. Independent bitwise verification of all six arrays and exact descriptor binding remain required before any model inference.
