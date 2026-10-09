"""Verified, bounded HTTP-object cache for encoded Zarr chunks.

Only encoded bytes are cached; missing objects are never turned into fill here.
The local POSIX cache lock covers downloads, verification and eviction.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Iterator
from urllib.parse import urljoin, urlsplit
import xml.etree.ElementTree as ET

import requests


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_BLOCK = 64 * 1024
_ERROR_LIMIT = 64 * 1024
_TRANSIENT = {408, 425, 429, 500, 502, 503, 504}


class CacheError(RuntimeError):
    """Cache errors deliberately omit request URLs and response bodies."""


class CacheBudgetError(CacheError):
    pass


class CacheIntegrityError(CacheError):
    pass


class SourceChangedError(CacheIntegrityError):
    pass


class CacheTransportError(CacheError):
    pass


class CacheHTTPError(CacheError):
    def __init__(self, status_code: int, source_id: str):
        self.status_code = status_code
        self.source_id = source_id
        super().__init__(f"HTTP {status_code} for source {source_id[:12]}")


class MissingChunkError(CacheHTTPError):
    """A bounded HTTP 404 XML Error explicitly reported NoSuchKey."""

    missing_key = True


class _Retryable(Exception):
    def __init__(self, status: int | None = None):
        self.status = status


def _hash(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value.lower()):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value.lower()


def _url_parts(url: str):
    if not isinstance(url, str):
        raise ValueError("Invalid source URL")
    try:
        parts = urlsplit(url)
        port = parts.port
    except (ValueError, TypeError):
        raise ValueError("Invalid source URL") from None
    if (parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.fragment or any(ord(c) < 33 for c in url)):
        raise ValueError("Source URL must be HTTP(S), without userinfo or fragments")
    return parts, port


def _etag(value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or not value
                             or len(value) > 1024
                             or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError("Invalid ETag")
    return value


def _owner(value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or not value
                             or len(value) > 128
                             or any(ord(c) < 32 for c in value)):
        raise ValueError("Pin owner must be a nonempty string of at most 128 characters")
    return value


@dataclass(frozen=True)
class SourceObject:
    """The URL is hashed for identity and excluded from repr and manifests.

    metadata_sha256 identifies the exact retained array metadata bytes. An
    optional ETag is sent as If-Match and must match the response exactly.
    expected_sha256 additionally pins content, independently of HTTP headers.
    """

    url: str = field(repr=False)
    metadata_sha256: str
    etag: str | None = None
    expected_sha256: str | None = None

    def __post_init__(self):
        _url_parts(self.url)
        object.__setattr__(self, "metadata_sha256", _hash(self.metadata_sha256, "metadata_sha256"))
        _etag(self.etag)
        if self.expected_sha256 is not None:
            object.__setattr__(self, "expected_sha256", _hash(self.expected_sha256, "expected_sha256"))

    @property
    def source_id(self) -> str:
        identity = {"url_sha256": hashlib.sha256(self.url.encode()).hexdigest(),
                    "metadata_sha256": self.metadata_sha256, "etag": self.etag}
        return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class CachedObject:
    path: Path
    sha256: str
    size_bytes: int
    source_id: str
    etag: str | None
    reused: bool


class ChunkCache:
    """A local encoded-byte cache with persistent, named pin owners.

    Every fetch verifies bytes, including cache hits. Unpinned paths may be
    evicted by a subsequent call; pin them until consumers finish reading.
    The byte cap includes blobs and in-progress downloads, not small JSON
    manifests. The free-space reserve protects the entire local filesystem.
    """

    def __init__(self, root: Path | str, *, max_object_bytes: int,
                 max_cache_bytes: int, min_free_bytes: int = 0, retries: int = 2,
                 retry_backoff: float = 0.2, timeout: tuple[float, float] = (10, 45),
                 allow_loopback_http: bool = False):
        if (type(max_object_bytes) is not int or type(max_cache_bytes) is not int
                or max_object_bytes <= 0 or max_cache_bytes < max_object_bytes
                or type(min_free_bytes) is not int or min_free_bytes < 0):
            raise ValueError("Require 0 < object cap <= cache cap and a nonnegative reserve")
        if (type(retries) is not int or not 0 <= retries <= 5
                or not math.isfinite(retry_backoff) or not 0 <= retry_backoff <= 5
                or len(timeout) != 2 or any(not math.isfinite(t) or not 0 < t <= 120 for t in timeout)):
            raise ValueError("Invalid bounded retry/timeout settings")
        self.root = Path(root)
        self.max_object_bytes = max_object_bytes
        self.max_cache_bytes = max_cache_bytes
        self.min_free_bytes = min_free_bytes
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout
        self.allow_loopback_http = allow_loopback_http
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.blobs = self.root / "blobs"
        self.sources = self.root / "sources"
        self.temporary = self.root / "tmp"
        for path in (self.root, self.blobs, self.sources, self.temporary):
            if path.is_symlink():
                raise CacheIntegrityError("Cache directories must not be symlinks")
            path.mkdir(exist_ok=True, mode=0o700)
        self.session = requests.Session()

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.root / ".lock", flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            # No other compliant process can have an active download now.
            for path in self.temporary.glob("*.part"):
                if path.is_symlink() or not path.is_file():
                    raise CacheIntegrityError("Unexpected temporary cache entry")
                path.unlink()
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _validate_url(self, url: str):
        parts, _ = _url_parts(url)
        if parts.scheme == "https":
            return
        try:
            loopback = ipaddress.ip_address(parts.hostname).is_loopback
        except ValueError:
            loopback = False
        if not (self.allow_loopback_http and loopback):
            raise ValueError("HTTPS is required outside explicitly allowed loopback tests")

    def _read_record(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            if path.is_symlink() or path.stat().st_size > 256 * 1024:
                raise ValueError
            record = json.loads(path.read_text())
            if (record["schema"] != 1 or record["source_id"] != path.stem
                    or type(record["size_bytes"]) is not int or record["size_bytes"] < 0
                    or not isinstance(record["pin_owners"], list)
                    or len(record["pin_owners"]) > 1000
                    or not math.isfinite(record["last_access"])):
                raise ValueError
            for key in ("source_id", "url_sha256", "metadata_sha256", "sha256"):
                if _hash(record[key], key) != record[key]:
                    raise ValueError
            for owner in record["pin_owners"]:
                if _owner(owner) is None:
                    raise ValueError
            _etag(record["requested_etag"])
            _etag(record["etag"])
            if record["requested_etag"] is not None and record["requested_etag"] != record["etag"]:
                raise ValueError
        except (ValueError, KeyError, TypeError, OSError):
            raise CacheIntegrityError("Invalid cache manifest; preserve and repair it explicitly") from None
        return record

    def _write_record(self, record: dict, *, releasing_pin: bool = False):
        payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
        if len(record["pin_owners"]) > 1000 or len(payload) > 256 * 1024:
            raise CacheIntegrityError("Cache manifest or pin owner limit exceeded")
        if releasing_pin:
            # A metadata-only release must remain possible under disk pressure,
            # so pins can be released before the next eviction/download.
            while shutil.disk_usage(self.root).free < len(payload) + _BLOCK:
                if not self._evict_one({record["sha256"]}):
                    raise CacheBudgetError("Insufficient actual disk space to release a pin atomically")
        else:
            self._make_room(0, {record["sha256"]}, reserve_extra=len(payload))
        fd, name = tempfile.mkstemp(suffix=".part", dir=self.temporary)
        path = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(path, self.sources / f"{record['source_id']}.json")
            self._sync_directory(self.sources)
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _sync_directory(path: Path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _verify(path: Path, sha256: str, size: int) -> bool:
        if path.is_symlink():
            raise CacheIntegrityError("Cache blob must not be a symlink")
        if not path.is_file() or path.stat().st_size != size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(_BLOCK), b""):
                digest.update(block)
        return digest.hexdigest() == sha256

    def _encoded_bytes(self) -> int:
        total = 0
        for directory in (self.blobs, self.temporary):
            for path in directory.iterdir():
                if path.is_symlink() or not path.is_file():
                    raise CacheIntegrityError("Unexpected encoded cache entry")
                total += path.stat().st_size
        return total

    def _evict_one(self, protected: set[str]) -> bool:
        records = [self._read_record(p) for p in self.sources.glob("*.json")]
        candidates = []
        for path in self.blobs.glob("*.bin"):
            if not _SHA.fullmatch(path.stem):
                continue
            aliases = [r for r in records if r is not None and r["sha256"] == path.stem]
            if path.stem in protected or any(r["pin_owners"] for r in aliases):
                continue
            last_access = max((r["last_access"] for r in aliases), default=path.stat().st_mtime)
            candidates.append((last_access, path, aliases))
        if not candidates:
            return False
        _, path, aliases = min(candidates, key=lambda item: (item[0], item[1].name))
        path.unlink()
        for record in aliases:
            (self.sources / f"{record['source_id']}.json").unlink(missing_ok=True)
        return True

    def _make_room(self, additional: int, protected: set[str] | None = None,
                   *, reserve_extra: int = 0):
        # Leave headroom for filesystem allocation/metadata as well as bytes.
        disk_required = additional + reserve_extra + _BLOCK
        while (self._encoded_bytes() + additional > self.max_cache_bytes
               or shutil.disk_usage(self.root).free - disk_required < self.min_free_bytes):
            if not self._evict_one(protected or set()):
                raise CacheBudgetError("Encoded cache cap or disk reserve cannot be satisfied without evicting pinned inputs")

    def _response(self, source: SourceObject):
        url = source.url
        original, _ = _url_parts(url)
        headers = {"Accept-Encoding": "identity"}
        if source.etag is not None:
            headers["If-Match"] = source.etag
        for _ in range(4):
            self._validate_url(url)
            response = self.session.get(url, headers=headers, stream=True,
                                        timeout=self.timeout, allow_redirects=False)
            if response.status_code not in {301, 302, 303, 307, 308}:
                return response
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise CacheTransportError("Redirect has no location")
            target = urljoin(url, location)
            try:
                parts, _ = _url_parts(target)
                self._validate_url(target)
                if original.scheme == "https" and parts.scheme != "https":
                    raise ValueError
                if original.scheme == "http" and (parts.scheme, parts.netloc) != (original.scheme, original.netloc):
                    raise ValueError
            except ValueError:
                raise CacheTransportError("Unsafe redirect refused") from None
            url = target
        raise CacheTransportError("Redirect limit exceeded")

    @staticmethod
    def _missing_key(response) -> bool:
        body = bytearray()
        for block in response.iter_content(4096):
            if len(body) + len(block) > _ERROR_LIMIT:
                raise CacheHTTPError(404, "unknown")
            body.extend(block)
        if b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
            return False
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            return False
        return (root.tag.split("}")[-1] == "Error"
                and any(child.tag.split("}")[-1] == "Code" and child.text == "NoSuchKey" for child in root))

    def _download_once(self, source: SourceObject, path: Path) -> tuple[str, int, str | None]:
        with self._response(source) as response:
            status = response.status_code
            if status in _TRANSIENT:
                raise _Retryable(status)
            if status == 412:
                raise SourceChangedError("Source rejected the pinned ETag")
            if status == 404:
                if self._missing_key(response):
                    raise MissingChunkError(404, source.source_id)
                raise CacheHTTPError(404, source.source_id)
            if status != 200:
                raise CacheHTTPError(status, source.source_id)
            observed_etag = response.headers.get("ETag")
            try:
                _etag(observed_etag)
            except ValueError:
                raise CacheIntegrityError("Invalid source ETag") from None
            if source.etag is not None and observed_etag != source.etag:
                raise SourceChangedError("Response does not match the pinned ETag")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise CacheIntegrityError("Unexpected HTTP content encoding")
            length = response.headers.get("Content-Length")
            if length is not None:
                if not length.isascii() or not length.isdigit():
                    raise CacheIntegrityError("Invalid response length")
                if int(length) > self.max_object_bytes:
                    raise CacheBudgetError("Object exceeds its encoded byte cap")
            digest, size = hashlib.sha256(), 0
            with path.open("wb") as handle:
                for block in response.iter_content(_BLOCK):
                    if not block:
                        continue
                    if size + len(block) > self.max_object_bytes:
                        raise CacheBudgetError("Object exceeds its encoded byte cap")
                    self._make_room(len(block))
                    handle.write(block)
                    handle.flush()
                    digest.update(block)
                    size += len(block)
                os.fsync(handle.fileno())
            if length is not None and size != int(length):
                raise CacheIntegrityError("Response length does not match downloaded bytes")
            return digest.hexdigest(), size, observed_etag

    def _download(self, source: SourceObject, path: Path) -> tuple[str, int, str | None]:
        for attempt in range(self.retries + 1):
            path.unlink(missing_ok=True)
            try:
                return self._download_once(source, path)
            except requests.exceptions.SSLError:
                raise CacheTransportError("TLS verification failed") from None
            except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
                status = None
            except _Retryable as error:
                status = error.status
            except requests.RequestException:
                raise CacheTransportError("HTTP transport failed") from None
            if attempt == self.retries:
                if status is not None:
                    raise CacheHTTPError(status, source.source_id) from None
                raise CacheTransportError("Transient transport retry limit exhausted") from None
            time.sleep(min(2.0, self.retry_backoff * 2 ** attempt))
        raise AssertionError("unreachable")

    def fetch(self, source: SourceObject, *, pin: str | None = None) -> CachedObject:
        """Return verified local bytes, downloading or repairing if necessary."""
        self._validate_url(source.url)
        _owner(pin)
        with self._locked():
            record_path = self.sources / f"{source.source_id}.json"
            previous = self._read_record(record_path)
            owners = set(previous["pin_owners"] if previous is not None else [])
            if pin is not None:
                owners.add(pin)
            required_sha = source.expected_sha256
            if previous is not None:
                if (previous["url_sha256"] != hashlib.sha256(source.url.encode()).hexdigest()
                        or previous["metadata_sha256"] != source.metadata_sha256
                        or previous["requested_etag"] != source.etag):
                    raise CacheIntegrityError("Manifest source identity mismatch")
                if required_sha is not None and previous["sha256"] != required_sha:
                    raise SourceChangedError("Pinned content hash conflicts with cached source")
                required_sha = previous["sha256"]
                path = self.blobs / f"{previous['sha256']}.bin"
                if previous["size_bytes"] > self.max_object_bytes:
                    raise CacheBudgetError("Cached object exceeds this object's byte cap")
                if self._verify(path, previous["sha256"], previous["size_bytes"]):
                    self._make_room(0, {previous["sha256"]})
                    previous.update(pin_owners=sorted(owners), last_access=time.time())
                    self._write_record(previous)
                    return CachedObject(path, previous["sha256"], previous["size_bytes"],
                                        source.source_id, previous["etag"], True)
                # Preserve manifests/pin owners and require the same bytes on repair.
                path.unlink(missing_ok=True)
            self._make_room(0)
            fd, name = tempfile.mkstemp(suffix=".part", dir=self.temporary)
            os.close(fd)
            temporary = Path(name)
            try:
                sha256, size, observed_etag = self._download(source, temporary)
                if required_sha is not None and sha256 != required_sha:
                    raise SourceChangedError("Downloaded content does not match its pinned SHA-256")
                path = self.blobs / f"{sha256}.bin"
                if path.is_symlink():
                    raise CacheIntegrityError("Cache blob must not be a symlink")
                os.replace(temporary, path)
                self._sync_directory(self.blobs)
                parts, _ = _url_parts(source.url)
                record = {"schema": 1, "source_id": source.source_id,
                          "url_sha256": hashlib.sha256(source.url.encode()).hexdigest(),
                          "source_origin": f"{parts.scheme}://{parts.netloc}",
                          "metadata_sha256": source.metadata_sha256,
                          "requested_etag": source.etag, "etag": observed_etag,
                          "sha256": sha256, "size_bytes": size,
                          "pin_owners": sorted(owners), "last_access": time.time(),
                          "retrieved_at_unix": time.time()}
                self._write_record(record)
                return CachedObject(path, sha256, size, source.source_id, observed_etag, False)
            finally:
                temporary.unlink(missing_ok=True)

    def unpin(self, source: SourceObject, *, owner: str) -> bool:
        """Release only this owner's pin; return whether it existed."""
        _owner(owner)
        if owner is None:
            raise ValueError("Pin owner is required")
        with self._locked():
            record = self._read_record(self.sources / f"{source.source_id}.json")
            if record is None or owner not in record["pin_owners"]:
                return False
            record["pin_owners"].remove(owner)
            self._write_record(record, releasing_pin=True)
            return True
