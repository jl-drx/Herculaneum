# Verified run artifacts

`herculaneum.artifacts` stores run outputs locally or in an authenticated R2
bucket. It does not create cloud resources, configure bucket visibility, or
upload anything merely by constructing a store.

```python
from pathlib import Path
from herculaneum.artifacts import LocalArtifactStore, publish_run, verify_run

store = LocalArtifactStore("/workspace/results")
manifest = publish_run(
    store,
    "experiment-001",
    {"scores.npy": Path("/workspace/job/scores.npy")},
    metadata={"git_commit": "record-the-exact-commit", "seed": 42},
)
verified = verify_run(store, "experiment-001")
```

Both functions return a manifest dictionary. `fingerprint_file(path)` returns
`{"sha256": "...", "size_bytes": 123}` without loading a file into memory.
Use `ArtifactError` to catch store failures; its subclasses distinguish
integrity failures, immutable-path collisions and transport failures.

The CLI can publish a results-only directory and verify it independently:

```sh
herculaneum artifacts publish --input-dir /worker/results --store-dir /workspace/artifacts --run-id experiment-001
herculaneum artifacts verify --store-dir /workspace/artifacts --run-id experiment-001
```

Select `--store r2` after configuring the runtime credentials. Do not use a
directory containing credentials, source checkouts or mutable outputs. The CLI
rejects symlinks and hidden files, and prints only compact completion status.

## Publication and recovery

Every run uses these relative keys:

```text
runs/<run-id>/plan.json
runs/<run-id>/artifacts/<relative-name>
runs/<run-id>/manifest.json
```

Run IDs are a single safe path component. Artifact names may contain `/`,
but absolute paths, empty components, traversal and backslashes are rejected.
The local backend also rejects symlinks within the store's artifact paths.

Publication first hashes every source and freezes an immutable plan with
`state: "planned"`. It then uploads each artifact and independently reads its
stored bytes to verify SHA-256 and exact length. Only after all artifacts pass
does it publish `manifest.json` with `state: "complete"`. The manifest has
`schema_version: 1`, `run_id`, nonsecret `metadata`, and an `artifacts` list of
`path`, `key`, `sha256` and `size_bytes` records.

Files, artifact sets and metadata must remain frozen during publication.
Retry the same `publish_run` call after an interruption: matching objects are
read and verified, then skipped. Changed inputs or metadata conflict with the
plan or manifest and require a new run ID. A missing or corrupt artifact in an
already completed run causes verification to fail; it is never silently
repaired or overwritten.

An interrupted operation may leave a plan or artifacts without a completion
manifest. If a completion write succeeded but its response or read-back failed,
the function can raise even though the stored manifest is valid. An exact retry
or `verify_run` reconciles that uncertain outcome. Treat a successful return,
or a separately successful verification, as evidence of completion.

The local backend stages and fsyncs a temporary file, then uses an atomic
no-overwrite hard link. Failed staging is cleaned up. Store immutability is an
application protocol, not protection against administrators or other programs
deleting/changing data. Do not allow untrusted concurrent filesystem writers;
use appropriate bucket permissions and retain separate backups where needed.

## Private R2

Use the existing optional `botocore` SDK (the lab lock pins 1.43.106):

```python
from herculaneum.artifacts import R2ArtifactStore

store = R2ArtifactStore.from_env(prefix="herculaneum")
```

Inject these values through the runtime's secret/configuration mechanism:

- `R2_ACCOUNT_ID`: 32-character Cloudflare account ID.
- `R2_BUCKET`: existing private bucket name.
- `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`: bucket-scoped R2 credentials.

No AWS credential or profile fallback is used. Do not put credential values,
presigned URLs or signed request details in Git, metadata or command-line
arguments. This module emits no credential logs and sanitizes SDK transport
exceptions; do not enable SDK HTTP debugging around real credentials. Bucket
privacy and access scope are deployment prerequisites, not verified by this
client. It uses the authenticated S3 endpoint with region `auto`, without
public-read ACLs or generated public links.

R2 uploads use low-level `PutObject` with `IfNoneMatch="*"`; retries never
fall back to unconditional writes. SHA-256 and length are checked with
streaming GETs rather than ETags or caller-supplied object metadata. These
verification reads incur R2 request charges and transfer time. Client requests
have bounded retries and connection/read timeouts.

The initial R2 backend rejects files larger than **5,000,000,000 bytes**, before
any publication request. This conservative limit is below the documented
5 GiB single-upload ceiling. [R2 documents conditional single PUTs](https://developers.cloudflare.com/r2/api/s3/api/#implemented-object-level-operations),
but the same table does not establish conditional multipart completion support.
The module deliberately does not use a transfer helper that might silently
switch to multipart and weaken no-overwrite protection. A future extension can
use independently immutable chunks or a separately verified conditional
multipart mechanism. [Multipart ETags](https://developers.cloudflare.com/r2/objects/upload-objects/#etags)
are not whole-file SHA-256 hashes.

## Tests

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_artifacts.py -v
```

Tests use temporary local files and a fake R2 client. They cover interruption,
exact resume, collisions, source truncation, stored corruption, publication
order and credential isolation. The SDK initialization test uses dummy R2
values and a network tripwire. There is no live-cloud integration claim.
