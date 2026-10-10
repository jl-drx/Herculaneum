# Technical review: bounded 256-row serving probe

GO for the exact five source files bound in code-go.json. No material blocker found. This approval concerns only the separately frozen six-request probe, not production acquisition or scientific inference.

The preparer binds the exact completed V2 header and eight-row serving PASS receipts, verifies all three literal source/context/role identities and observed layouts, and verifies retained nested raw-file records and prior plan pins. Frag1 remains big-endian; Frag2 and Frag3 remain little-endian. Row addressing is byte-order independent. No extra header request is introduced.

The transport retains verified HTTPS, strong exact ETag/If-Match, identity encoding and strict HTTP206 behavior. It rejects HTTP200/redirect/error before body reading, carries object lengths, performs no automatic retry, and preserves conservative failed-attempt reservation plus partial-file accounting. The only substantive transport generalization is exactly256 ranges and framing768*n+1024.

The parser uses declared binary payload lengths, not delimiter splitting, and requires all256 disjoint requested parts with unchanged object total. Unknown, duplicate, missing, coalesced, truncated, overlong and trailing data fail closed. Boundaries, part headers and aggregate framing remain bounded. A comparator is requested only after successful multipart parsing, and all256 crop rows must equal their exact full-row slices.

The six responses reserve13,157,891bytes, below16MiB combined and6MiB per response. Source pins, request derivation and technical GO are rechecked by the execution loader. The runner sets240s alarm and2GiB address-space limit and checks32MiB retained output cap. There is no alternative row count, URL, fallback, pixel decoder or model path.

Validation:38 independent synthetic cases PASS;38 author synthetic cases independently re-executed PASS. The independent cases include delimiter-like payloads, reversed wire ordering, boundary/header/framing limits, coalescing/count/total failures, stale/duplicate validators, partial failures, conservative charging, budget prechecks, independently derived addressing, full synthetic256-row byte comparison and rejection before comparator. Network sockets were blocked. No real CT bytes, numeric CT arrays, checkpoints or models were read or executed by these fixtures.

Limit: a real PASS would establish this request size only for the three observed07.tif objects. The other150planes and full acquisition remain subject to a separate frozen contract and verification.
