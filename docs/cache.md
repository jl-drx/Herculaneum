# Encoded Zarr object cache

`herculaneum.chunk_cache` downloads bounded HTTP objects into a local verified
content-addressed cache. It stores encoded bytes, not decoded arrays, and makes
no decision about labels or declared Zarr fill values.

```python
from pathlib import Path
from herculaneum.chunk_cache import ChunkCache, SourceObject

source = SourceObject(
    "https://example.org/array/0/1/2",
    metadata_sha256=retained_zarray_sha256,
    etag=pinned_etag,                    # optional exact HTTP ETag, including quotes
    expected_sha256=pinned_chunk_sha256, # optional known content digest
)
with ChunkCache(
    Path("/worker-cache"),
    max_object_bytes=20 * 1024**2,
    max_cache_bytes=2 * 1024**3,
    min_free_bytes=1024**3,
) as cache:
    chunk = cache.fetch(source, pin="run-2026-10-08")
    try:
        consume_encoded_file(chunk.path) # decode separately with another RAM limit
    finally:
        cache.unpin(source, owner="run-2026-10-08")
```

The source identity hashes the exact URL, exact retained metadata digest, and
optional requested ETag. The caller supplies and retains the metadata bytes;
the cache does not independently acquire or validate `.zarray`. Signed URL
queries therefore affect identity when their signatures change. Manifests keep
the URL digest and origin, not the raw URL or query. URL userinfo is rejected,
and request exceptions omit raw URLs, bodies, and credential-bearing causes.

Blobs are named by their SHA-256. Every hit verifies length and hash. A corrupt
or missing blob is downloaded again, requiring its original manifest digest;
changed bytes fail explicitly. Malformed manifests stop without being overwritten.
Without an ETag or expected digest, a new source entry records its first observed
bytes: this is local verified reuse, not a guarantee of remote freshness. Pin
the ETag and/or content digest for reproducible jobs. ETags are not assumed to
be cryptographic checksums.

Named pins survive process restart and protect shared blobs while any alias has
an owner. Release only your own owner after reading. Unpinned paths may be
evicted by the next cache call. Unpinned blobs use least-recent-access eviction;
unreferenced blobs left by a crash are also eligible. In-flight partial bytes
count toward the total encoded cap. Small manifests are outside that cap;
Acquisition and ordinary manifest writes observe the free-space reserve with
64 KiB allocation headroom.
The metadata-only `unpin` operation may use that reserve when actual free space
still permits its atomic update, allowing pinned inputs to be released under
disk pressure. It does not add encoded data.

Downloads write temporary files, then atomically promote verified blobs and
manifests. Retry attempts restart an interrupted object; completed verified
objects survive across calls. The default is two retries for selected transient
HTTP statuses and connection/read failures, with capped backoff and inactivity
timeouts. Retries are capped at five. Missing objects are not negatively cached.

Only a bounded HTTP 404 XML `Error/Code=NoSuchKey` raises `MissingChunkError`.
Generic 404, access failures, invalid encoding, source changes, and exhausted
budgets remain errors. The caller may apply a declared Zarr fill only after
checking its own frozen metadata and this explicit missing-key condition.

HTTPS is required. The explicit `allow_loopback_http=True` switch supports local
tests using literal loopback addresses; redirects cannot escape that test origin.
A coarse POSIX `flock` serializes acquisition, verification and eviction across
cache instances. Use local POSIX storage with reliable locking/atomic rename,
not an object-store mount. This small implementation favors correctness over
parallel download throughput. At most 1,000 named owners and a 256 KiB manifest
are allowed per source; overflow fails before overwriting the existing manifest.
