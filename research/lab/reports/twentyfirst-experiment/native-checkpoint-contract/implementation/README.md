# Header-only execution

This implementation applies the sealed proposal and root's `NATIVE_HEADER_GO.json`. It uses verified TLS, explicit redirects, no retries, exact ranges and strong matching ETags. Six attempts and 262,144 observed response-body bytes include bounded redirect/error bodies. Every attempt has a 30-second absolute signal deadline; the invocation also requires an external 210-second wall limit.

Only the safetensors length prefix and JSON header are retained. Redirect/error bodies are hashed and counted, then discarded. Signed destinations remain transient in memory. An ignored-range HTTP200 stops before any body read. Missing Content-Length, an unexpected host, changed validator, malformed range, invalid header or exhausted bound stops the query. HTTP framing, TLS, headers and unread socket buffers are outside observed-body accounting. Bounded reads honor declared HTTP framing; the reader does not probe beyond Content-Length.

Synthetic fixtures exercise the real parser against all 508 retained names/244 factory alias groups, and fake transport failure/control paths. They generate no tensor values and prohibit socket connections. The standalone signal fixture verifies absolute-deadline interruption. Tests must run before the code freeze; do not rerun them after sealing because they write their receipt.

The one-time invocation is:

```bash
timeout --signal=TERM --kill-after=3s 210s python read_header.py --output ../query-run
```

An output directory must be new. The result records pre/post input hashes and sanitized traffic. Even a matching header cannot establish full-file SHA256, tensor values, alias-value equality, shared storage or strict model loading. No model package is imported.
