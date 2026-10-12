# Observed outbound transport failure

On 12 October 2026, the cloud environment intermittently retrieved public S3
inputs, then returned HTTP 503 before upstream response headers. The same turn's
read-only GitHub Git ref request and GraphQL repository-metadata request also
failed with HTTP 503. This affects more than the teacher TIFF endpoint.

The S3 diagnostic response was:

```text
upstream connect error or disconnect/reset before headers.
reset reason: remote connection failure, transport failure reason:
immediate connect error: Invalid argument|remote address:envoy://cloudflare_https_tunnel/
```

The bounded diagnostic reused one connection and socket for seven successful
teacher ranges and the subsequent failed request. It preserved the inherited
proxy, CA trust and verified TLS; local connection reuse is not an established
fix. The original five-winding acquisition did finish earlier in the turn.

Runtime status reported connected/running, observations current, revision 29,
with no provider failure field. The HTTP policy was declared unrestricted but
its observed enforcement state was unknown. The executor's supported policy
snapshot likewise declared unrestricted HTTP and no configured VPN. These
metadata observations do not establish working service access.

No configuration, authentication or source-identity failure has been identified.
No credential, allowlist replacement, VPN change, TLS bypass or restart was
attempted. The unresolved prerequisite is reliable platform-provided outbound
HTTP transport. Provider-internal cause and a user-configurable repair remain
unknown. Preserve the completed-source checkpoint; retry after a meaningful
recovery rather than changing scientific eligibility or looping requests.

GitHub persistence is assessed separately after local artifact verification;
see the final persistence receipt for whether the current commit and archive
were successfully uploaded. Earlier verified checkpoints remain historical.
