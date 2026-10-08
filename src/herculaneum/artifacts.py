"""Immutable, checksum-verified run artifacts with local and private R2 stores.

Only ``manifest.json`` marks a complete run. Every artifact is read back and
verified before that manifest is published. R2 credentials are explicit and
separate from the AWS credentials used for public scan access.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Protocol


_BLOCK_SIZE = 1024 * 1024
_MANIFEST_LIMIT = 4 * 1024 * 1024
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ArtifactError(RuntimeError):
    """An artifact operation failed without establishing completion."""


class ArtifactIntegrityError(ArtifactError):
    """Stored bytes or a completion manifest failed verification."""


class ArtifactCollisionError(ArtifactIntegrityError):
    """An immutable path already contains different bytes."""


class ArtifactTransportError(ArtifactError):
    """Storage access failed; underlying credentials/URLs are not disclosed."""


def _relative_name(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Artifact names must be nonempty relative POSIX paths")
    if any(not _COMPONENT.fullmatch(part) for part in value.split("/")):
        raise ValueError("Artifact names contain an unsafe path component")
    return value


def _run_id(value: str) -> str:
    value = _relative_name(value)
    if "/" in value:
        raise ValueError("A run ID must be one path component")
    return value


def _fingerprint(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    length = 0
    while block := stream.read(_BLOCK_SIZE):
        digest.update(block)
        length += len(block)
    return digest.hexdigest(), length


def fingerprint_file(path: str | Path) -> dict[str, Any]:
    """Return the SHA-256 and byte length of a file using bounded reads."""
    with Path(path).open("rb") as stream:
        sha256, size_bytes = _fingerprint(stream)
    return {"sha256": sha256, "size_bytes": size_bytes}


class ArtifactStore(Protocol):
    """A store must create atomically, never replace, and return raw bytes."""

    max_artifact_size: int | None

    def open(self, key: str) -> BinaryIO | None:
        """Open an existing object, or return None only when it is absent."""

    def put_if_absent(self, key: str, stream: BinaryIO, size_bytes: int) -> bool:
        """Atomically create an object; return False if its key already exists."""


class LocalArtifactStore:
    """Filesystem store using staged files and atomic no-overwrite hard links."""

    max_artifact_size = None

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, *, make_parents: bool = False) -> Path:
        parts = _relative_name(key).split("/")
        current = self.root
        for index, part in enumerate(parts):
            current = current / part
            if current.is_symlink():
                raise ArtifactIntegrityError("Artifact paths must not contain symlinks")
            if make_parents and index < len(parts) - 1:
                current.mkdir(exist_ok=True)
        return current

    def open(self, key: str) -> BinaryIO | None:
        try:
            return self._path(key).open("rb")
        except FileNotFoundError:
            return None

    def put_if_absent(self, key: str, stream: BinaryIO, size_bytes: int) -> bool:
        destination = self._path(key, make_parents=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".upload-", delete=False
            ) as staged:
                temporary = staged.name
                length = 0
                while block := stream.read(_BLOCK_SIZE):
                    staged.write(block)
                    length += len(block)
                if length != size_bytes:
                    raise ArtifactIntegrityError("Source length changed during upload")
                staged.flush()
                os.fsync(staged.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                return False
            # Persist the directory entry as well as the file bytes.
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return True
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)


def _response_code(error: Exception) -> tuple[str, int | None]:
    try:
        response = getattr(error, "response", {})
        if not isinstance(response, Mapping):
            return "", None
        details = response.get("Error", {})
        metadata = response.get("ResponseMetadata", {})
        code = details.get("Code", "") if isinstance(details, Mapping) else ""
        status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
        return str(code), status
    except Exception:
        return "", None


class _PrivateBody:
    """Hide potentially sensitive SDK messages from streaming failures."""

    def __init__(self, body: Any):
        self.body = body

    def read(self, size: int = -1) -> bytes:
        try:
            return self.body.read(size)
        except Exception:
            raise ArtifactTransportError("R2 object read failed") from None

    def close(self) -> None:
        try:
            self.body.close()
        except Exception:
            raise ArtifactTransportError("R2 object close failed") from None

    def __enter__(self) -> _PrivateBody:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class R2ArtifactStore:
    """Authenticated S3 API store; never uses public URLs or ETag checksums.

    Single conditional PUTs preserve immutable final keys. The conservative
    5,000,000,000-byte cap avoids relying on undocumented conditional multipart
    completion. Larger files are rejected before publication starts.
    """

    max_artifact_size = 5_000_000_000

    def __init__(self, client: Any, bucket: str, *, prefix: str = ""):
        if not bucket or "/" in bucket:
            raise ValueError("An R2 bucket name is required")
        self.client = client
        self.bucket = bucket
        self.prefix = _relative_name(prefix) + "/" if prefix else ""

    @classmethod
    def from_env(cls, *, prefix: str = "") -> R2ArtifactStore:
        """Build a client from R2_* variables, without AWS credential fallback.

        This creates no bucket and issues no storage request. Bucket privacy
        and appropriately scoped credentials must be configured separately.
        """
        names = ("R2_ACCOUNT_ID", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
        values = {name: os.environ.get(name, "") for name in names}
        missing = [name for name in names if not values[name]]
        if missing:
            raise ValueError("Missing R2 configuration: " + ", ".join(missing))
        if not re.fullmatch(r"[0-9a-fA-F]{32}", values["R2_ACCOUNT_ID"]):
            raise ValueError("R2_ACCOUNT_ID must be a 32-character hexadecimal ID")
        import botocore.session
        from botocore.config import Config

        session = botocore.session.get_session()
        # Ignore unrelated AWS profiles/config and pass only explicit R2 keys.
        session.get_component("config_store").set_config_variable("profile", None)
        session.set_config_variable("config_file", os.devnull)
        session.set_config_variable("credentials_file", os.devnull)
        try:
            client = session.create_client(
                "s3",
                endpoint_url=f"https://{values['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                region_name="auto",
                aws_access_key_id=values["R2_ACCESS_KEY_ID"],
                aws_secret_access_key=values["R2_SECRET_ACCESS_KEY"],
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    connect_timeout=10,
                    read_timeout=60,
                    retries={"mode": "standard", "total_max_attempts": 3},
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required",
                ),
            )
        except Exception:
            raise ArtifactTransportError("R2 client initialization failed") from None
        return cls(client, values["R2_BUCKET"], prefix=prefix)

    def _key(self, key: str) -> str:
        return self.prefix + _relative_name(key)

    def open(self, key: str) -> BinaryIO | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except Exception as error:
            code, status = _response_code(error)
            if code == "NoSuchKey" or (status == 404 and code in {"404", "NotFound"}):
                return None
            raise ArtifactTransportError("R2 GET failed") from None
        return _PrivateBody(response["Body"])  # type: ignore[return-value]

    def put_if_absent(self, key: str, stream: BinaryIO, size_bytes: int) -> bool:
        if size_bytes > self.max_artifact_size:
            raise ValueError("Artifact exceeds the conditional R2 upload size limit")
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=self._key(key),
                Body=stream,
                ContentLength=size_bytes,
                IfNoneMatch="*",
            )
        except Exception as error:
            code, status = _response_code(error)
            if code == "PreconditionFailed" or status == 412:
                return False
            raise ArtifactTransportError("R2 conditional PUT failed") from None
        return True


def _stored_fingerprint(store: ArtifactStore, key: str) -> tuple[str, int] | None:
    body = store.open(key)
    if body is None:
        return None
    with body:
        return _fingerprint(body)


def _verify_record(store: ArtifactStore, record: Mapping[str, Any]) -> None:
    actual = _stored_fingerprint(store, record["key"])
    if actual != (record["sha256"], record["size_bytes"]):
        raise ArtifactIntegrityError("Stored artifact failed SHA-256/length verification")


def _ensure_object(
    store: ArtifactStore, key: str, stream: BinaryIO, sha256: str, size_bytes: int
) -> None:
    expected = (sha256, size_bytes)
    existing = _stored_fingerprint(store, key)
    if existing is not None:
        if existing != expected:
            raise ArtifactCollisionError("Immutable artifact path contains different bytes")
        return
    created = store.put_if_absent(key, stream, size_bytes)
    actual = _stored_fingerprint(store, key)
    if actual != expected:
        error_type = ArtifactIntegrityError if created else ArtifactCollisionError
        raise error_type("Stored object failed SHA-256/length verification")


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _read_manifest(store: ArtifactStore, key: str) -> bytes | None:
    body = store.open(key)
    if body is None:
        return None
    with body:
        parts = []
        length = 0
        while block := body.read(min(_BLOCK_SIZE, _MANIFEST_LIMIT + 1 - length)):
            parts.append(block)
            length += len(block)
            if length > _MANIFEST_LIMIT:
                break
        result = b"".join(parts)
    if len(result) > _MANIFEST_LIMIT:
        raise ArtifactIntegrityError("Completion manifest exceeds the size limit")
    return result


def publish_run(
    store: ArtifactStore,
    run_id: str,
    artifacts: Mapping[str, str | Path],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish frozen local files and then a deterministic completion manifest.

    Exact retries re-read every stored artifact. Changed bytes, artifact sets
    or metadata require a new run ID. Failed attempts may leave verified
    artifacts, but do not claim completion. Sources must not be modified while
    publishing. Metadata must contain JSON-serializable, nonsecret values.
    """
    run_id = _run_id(run_id)
    if not artifacts:
        raise ValueError("At least one artifact is required")
    records = []
    for name, source in sorted(artifacts.items()):
        name = _relative_name(name)
        record = {"path": name, "key": f"runs/{run_id}/artifacts/{name}", **fingerprint_file(source)}
        if store.max_artifact_size is not None and record["size_bytes"] > store.max_artifact_size:
            raise ValueError("Artifact exceeds this store's upload size limit")
        records.append(record)
    manifest = {"schema_version": 1, "state": "complete", "run_id": run_id, "metadata": dict(metadata or {}), "artifacts": records}
    encoded = _manifest_bytes(manifest)
    if len(encoded) > _MANIFEST_LIMIT:
        raise ValueError("Completion manifest exceeds the size limit")
    manifest_key = f"runs/{run_id}/manifest.json"
    existing = _read_manifest(store, manifest_key)
    if existing is not None:
        if existing != encoded:
            raise ArtifactCollisionError("Run already has a different completion manifest")
        for record in records:
            _verify_record(store, record)
        _verify_plan(store, manifest)
        return manifest
    planned = _manifest_bytes({**manifest, "state": "planned"})
    _ensure_object(
        store,
        f"runs/{run_id}/plan.json",
        io.BytesIO(planned),
        hashlib.sha256(planned).hexdigest(),
        len(planned),
    )
    for record in records:
        with Path(artifacts[record["path"]]).open("rb") as source:
            _ensure_object(store, record["key"], source, record["sha256"], record["size_bytes"])
    _ensure_object(store, manifest_key, io.BytesIO(encoded), hashlib.sha256(encoded).hexdigest(), len(encoded))
    return manifest


def _verify_plan(store: ArtifactStore, manifest: Mapping[str, Any]) -> None:
    planned = _read_manifest(store, f"runs/{manifest['run_id']}/plan.json")
    if planned != _manifest_bytes({**manifest, "state": "planned"}):
        raise ArtifactIntegrityError("Run plan does not match its completion manifest")


def verify_run(store: ArtifactStore, run_id: str) -> dict[str, Any]:
    """Read a completion manifest and stream-verify every referenced artifact."""
    run_id = _run_id(run_id)
    encoded = _read_manifest(store, f"runs/{run_id}/manifest.json")
    if encoded is None:
        raise ArtifactIntegrityError("Run has no completion manifest")
    try:
        manifest = json.loads(encoded)
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["state"] != "complete" or manifest["run_id"] != run_id:
            raise ValueError
        records = manifest["artifacts"]
        if not isinstance(records, list) or not records or not isinstance(manifest["metadata"], dict):
            raise ValueError
        names = set()
        for record in records:
            name = _relative_name(record["path"])
            if name in names or record["key"] != f"runs/{run_id}/artifacts/{name}":
                raise ValueError
            names.add(name)
            if not isinstance(record["sha256"], str) or not _SHA256.fullmatch(record["sha256"]):
                raise ValueError
            if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
                raise ValueError
    except (ValueError, TypeError, KeyError):
        raise ArtifactIntegrityError("Invalid completion manifest") from None
    _verify_plan(store, manifest)
    for record in records:
        _verify_record(store, record)
    return manifest
