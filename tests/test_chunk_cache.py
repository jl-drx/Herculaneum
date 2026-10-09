"""Cache tests use only a temporary directory and a loopback HTTP fixture."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import requests

from herculaneum.chunk_cache import (
    CacheBudgetError, CacheHTTPError, CacheIntegrityError, CacheTransportError,
    ChunkCache, MissingChunkError, SourceChangedError, SourceObject,
)


METADATA = sha256(b'{"zarr_format":2}').hexdigest()


@contextmanager
def server_fixture():
    routes, counts, if_match = {}, {}, {}
    guard = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            key = urlsplit(self.path).path
            with guard:
                counts[key] = counts.get(key, 0) + 1
                if_match[key] = self.headers.get("If-Match")
                route = routes.get(key, {"status": 404, "body": b"missing"})
                if isinstance(route, list):
                    route = route[min(counts[key] - 1, len(route) - 1)]
                route = dict(route)
            body = route.get("body", b"")
            self.send_response(route.get("status", 200))
            for name, value in route.get("headers", {}).items():
                self.send_header(name, value)
            if not route.get("omit_length"):
                self.send_header("Content-Length", str(route.get("length", len(body))))
            self.end_headers()
            try:
                self.wfile.write(body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", routes, counts, if_match
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class ChunkCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = server_fixture()
        self.base, self.routes, self.counts, self.if_match = self.fixture.__enter__()
        self.caches = []

    def tearDown(self):
        for cache in self.caches:
            cache.close()
        self.fixture.__exit__(None, None, None)
        self.temp.cleanup()

    def cache(self, **kwargs):
        options = dict(max_object_bytes=32, max_cache_bytes=128,
                       allow_loopback_http=True, retries=2, retry_backoff=0)
        options.update(kwargs)
        cache = ChunkCache(self.root, **options)
        self.caches.append(cache)
        return cache

    def source(self, path="/0/1/2", **kwargs):
        return SourceObject(self.base + path, METADATA, **kwargs)

    def record(self, source):
        return json.loads((self.root / "sources" / f"{source.source_id}.json").read_text())

    def test_verified_reuse_across_instances_and_stale_partial_cleanup(self):
        self.routes["/0/1/2"] = {"body": b"chunk", "headers": {"ETag": '"v1"'}}
        source = self.source(etag='"v1"', expected_sha256=sha256(b"chunk").hexdigest())
        first = self.cache().fetch(source, pin="job-a")
        (self.root / "tmp" / "interrupted.part").write_bytes(b"partial")
        second = self.cache().fetch(source, pin="job-b")
        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(second.path.read_bytes(), b"chunk")
        self.assertEqual(self.counts["/0/1/2"], 1)
        self.assertEqual(self.if_match["/0/1/2"], '"v1"')
        self.assertEqual(self.record(source)["pin_owners"], ["job-a", "job-b"])
        self.assertEqual(list((self.root / "tmp").iterdir()), [])

    def test_same_size_corruption_is_repaired_and_pins_survive(self):
        self.routes["/0/1/2"] = {"body": b"chunk"}
        cache, source = self.cache(), self.source()
        result = cache.fetch(source, pin="job-a")
        result.path.write_bytes(b"xxxxx")
        repaired = cache.fetch(source)
        self.assertFalse(repaired.reused)
        self.assertEqual(repaired.path.read_bytes(), b"chunk")
        self.assertEqual(self.counts["/0/1/2"], 2)
        self.assertEqual(self.record(source)["pin_owners"], ["job-a"])

    def test_repair_does_not_accept_changed_source_bytes(self):
        self.routes["/0/1/2"] = {"body": b"chunk"}
        cache, source = self.cache(), self.source()
        cache.fetch(source).path.write_bytes(b"xxxxx")
        self.routes["/0/1/2"] = {"body": b"newer"}
        with self.assertRaises(SourceChangedError):
            cache.fetch(source)
        self.assertEqual(list((self.root / "tmp").iterdir()), [])

    def test_metadata_and_etag_are_part_of_source_identity(self):
        source = self.source()
        self.assertNotEqual(source.source_id, self.source(etag='"v1"').source_id)
        self.assertNotEqual(source.source_id, SourceObject(source.url, "0" * 64).source_id)

    def test_etag_and_expected_content_are_enforced(self):
        self.routes["/0/1/2"] = {"body": b"chunk", "headers": {"ETag": '"changed"'}}
        cache = self.cache()
        with self.assertRaises(SourceChangedError):
            cache.fetch(self.source(etag='"pinned"'))
        with self.assertRaises(SourceChangedError):
            cache.fetch(self.source(expected_sha256="0" * 64))
        self.routes["/0/1/2"] = {"status": 412}
        with self.assertRaises(SourceChangedError):
            cache.fetch(self.source(etag='"pinned"'))
        self.assertEqual(list((self.root / "blobs").iterdir()), [])

    def test_lru_eviction_and_named_pins(self):
        for path, body in (("/a", b"aaaaa"), ("/b", b"bbbbb"), ("/c", b"ccccc")):
            self.routes[path] = {"body": body}
        cache = self.cache(max_object_bytes=5, max_cache_bytes=10)
        a, b, c = self.source("/a"), self.source("/b"), self.source("/c")
        a_result = cache.fetch(a, pin="job-a")
        cache.fetch(a, pin="job-b")
        b_result = cache.fetch(b)
        cache.fetch(c)
        self.assertTrue(a_result.path.exists())
        self.assertFalse(b_result.path.exists())
        self.assertTrue(cache.unpin(a, owner="job-a"))
        self.assertFalse(cache.unpin(a, owner="job-a"))
        self.assertEqual(self.record(a)["pin_owners"], ["job-b"])
        self.assertLessEqual(sum(p.stat().st_size for p in (self.root / "blobs").iterdir()), 10)

    def test_shared_content_cannot_be_evicted_while_any_alias_is_pinned(self):
        self.routes.update({"/a": {"body": b"same"}, "/b": {"body": b"same"},
                            "/c": {"body": b"diff"}})
        cache = self.cache(max_object_bytes=4, max_cache_bytes=8)
        a, b = self.source("/a"), self.source("/b")
        self.assertEqual(cache.fetch(a, pin="job").path, cache.fetch(b).path)
        # Lower cap admits no download while the shared blob remains pinned.
        smaller = self.cache(max_object_bytes=4, max_cache_bytes=4)
        with self.assertRaises(CacheBudgetError):
            smaller.fetch(self.source("/c"))
        self.assertTrue(cache.fetch(a).reused)
        cache.unpin(a, owner="job")
        smaller.fetch(self.source("/c"))
        self.assertFalse((self.root / "sources" / f"{b.source_id}.json").exists())

    def test_object_limit_with_and_without_content_length(self):
        for omit in (False, True):
            with self.subTest(omit_length=omit):
                self.routes["/large"] = {"body": b"x" * 9, "omit_length": omit}
                with self.assertRaises(CacheBudgetError):
                    self.cache(max_object_bytes=8).fetch(self.source("/large"))
                self.assertEqual(list((self.root / "tmp").iterdir()), [])
                self.assertEqual(list((self.root / "blobs").iterdir()), [])

    def test_free_disk_reserve_is_enforced(self):
        cache = self.cache(min_free_bytes=100)
        self.routes["/0/1/2"] = {"body": b"chunk"}
        with patch("herculaneum.chunk_cache.shutil.disk_usage", return_value=SimpleNamespace(free=50)):
            with self.assertRaises(CacheBudgetError):
                cache.fetch(self.source())
        self.assertNotIn("/0/1/2", self.counts)

    def test_disk_pressure_does_not_block_metadata_only_pin_release(self):
        self.routes["/0/1/2"] = {"body": b"chunk"}
        cache, source = self.cache(min_free_bytes=200_000), self.source()
        cache.fetch(source, pin="job")
        # Enough actual space for atomic metadata, but below download reserve.
        with patch("herculaneum.chunk_cache.shutil.disk_usage", return_value=SimpleNamespace(free=100_000)):
            self.assertTrue(cache.unpin(source, owner="job"))
        self.assertEqual(self.record(source)["pin_owners"], [])

    def test_transient_errors_retry_but_stop_at_cap(self):
        self.routes["/retry"] = [{"status": 503}, {"status": 429}, {"body": b"ok"}]
        self.assertEqual(self.cache().fetch(self.source("/retry")).path.read_bytes(), b"ok")
        self.assertEqual(self.counts["/retry"], 3)
        self.routes["/fail"] = {"status": 503}
        with self.assertRaises(CacheHTTPError) as raised:
            self.cache(retries=1).fetch(self.source("/fail"))
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(self.counts["/fail"], 2)

    def test_interrupted_body_retries_without_appending_partial_bytes(self):
        self.routes["/retry"] = [{"body": b"partial", "length": 12}, {"body": b"whole object"}]
        result = self.cache().fetch(self.source("/retry"))
        self.assertEqual(result.path.read_bytes(), b"whole object")
        self.assertEqual(self.counts["/retry"], 2)

    def test_explicit_missing_key_is_distinct_and_not_cached(self):
        self.routes["/missing"] = {"status": 404, "body": b"<Error><Code>NoSuchKey</Code></Error>"}
        self.routes["/denied"] = {"status": 404, "body": b"<Error><Code>AccessDenied</Code></Error>"}
        cache = self.cache()
        with self.assertRaises(MissingChunkError):
            cache.fetch(self.source("/missing"))
        for path in ("/denied", "/generic"):
            with self.assertRaises(CacheHTTPError) as raised:
                cache.fetch(self.source(path))
            self.assertNotIsInstance(raised.exception, MissingChunkError)
            self.assertEqual(self.counts[path], 1)
        self.routes["/missing"] = {"body": b"now present"}
        self.assertEqual(cache.fetch(self.source("/missing")).path.read_bytes(), b"now present")

    def test_missing_error_body_is_bounded(self):
        self.routes["/missing"] = {"status": 404, "body": b"x" * (64 * 1024 + 1)}
        with self.assertRaises(CacheHTTPError) as raised:
            self.cache().fetch(self.source("/missing"))
        self.assertNotIsInstance(raised.exception, MissingChunkError)

    def test_no_signed_url_credentials_in_manifests_repr_or_errors(self):
        self.routes["/0/1/2"] = {"body": b"chunk"}
        source = self.source("/0/1/2?X-Amz-Signature=super-secret")
        cache = self.cache()
        cache.fetch(source)
        manifest = (self.root / "sources" / f"{source.source_id}.json").read_text()
        self.assertNotIn("super-secret", manifest)
        self.assertNotIn("super-secret", repr(source))
        failed = self.source("/generic?token=super-secret")
        with self.assertRaises(CacheHTTPError) as raised:
            cache.fetch(failed)
        self.assertNotIn("super-secret", str(raised.exception))
        with patch.object(cache.session, "get", side_effect=requests.ConnectionError("super-secret")):
            with self.assertRaises(CacheTransportError) as raised:
                cache.fetch(self.source("/transport?token=super-secret"))
        self.assertNotIn("super-secret", str(raised.exception))
        self.assertTrue(raised.exception.__suppress_context__)

    def test_http_is_explicitly_loopback_only_and_redirects_stay_local(self):
        with self.assertRaises(ValueError):
            self.cache(allow_loopback_http=False).fetch(self.source())
        with self.assertRaises(ValueError):
            self.cache().fetch(SourceObject("http://example.invalid/chunk", METADATA))
        self.routes["/escape"] = {"status": 302, "headers": {"Location": "https://example.invalid/chunk"}}
        with self.assertRaises(CacheTransportError):
            self.cache().fetch(self.source("/escape"))
        self.routes["/local"] = {"status": 302, "headers": {"Location": "/target"}}
        self.routes["/target"] = {"body": b"ok"}
        self.assertEqual(self.cache().fetch(self.source("/local")).path.read_bytes(), b"ok")

    def test_concurrent_instances_fetch_once_and_preserve_both_pins(self):
        self.routes["/0/1/2"] = {"body": b"chunk"}
        first, second, source = self.cache(), self.cache(), self.source()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(first.fetch, source, pin="job-a"),
                       pool.submit(second.fetch, source, pin="job-b")]
            results = [future.result(timeout=5) for future in futures]
        self.assertEqual(self.counts["/0/1/2"], 1)
        self.assertEqual(sorted(result.reused for result in results), [False, True])
        self.assertEqual(self.record(source)["pin_owners"], ["job-a", "job-b"])

    def test_https_downgrade_is_refused_even_with_loopback_test_switch(self):
        cache = self.cache()
        response = requests.Response()
        response.status_code = 302
        response.headers["Location"] = self.base + "/target"
        response._content = b""
        response._content_consumed = True
        with patch.object(cache.session, "get", return_value=response) as get:
            with self.assertRaises(CacheTransportError):
                cache.fetch(SourceObject("https://example.invalid/chunk?token=secret", METADATA))
        self.assertEqual(get.call_count, 1)
        self.assertNotIn("/target", self.counts)

    def test_invalid_manifest_fails_without_overwriting_evidence(self):
        self.routes["/0/1/2"] = {"body": b"chunk"}
        cache, source = self.cache(), self.source()
        cache.fetch(source)
        path = self.root / "sources" / f"{source.source_id}.json"
        path.write_text("not JSON")
        with self.assertRaises(CacheIntegrityError):
            cache.fetch(source)
        self.assertEqual(path.read_text(), "not JSON")

    def test_pin_limit_rejection_preserves_existing_manifest(self):
        self.routes["/0/1/2"] = {"body": b"chunk"}
        cache, source = self.cache(), self.source()
        cache.fetch(source)
        record = self.record(source)
        record["pin_owners"] = [f"job-{i}" for i in range(1000)]
        cache._write_record(record)
        with self.assertRaises(CacheIntegrityError):
            cache.fetch(source, pin="one-too-many")
        self.assertEqual(len(self.record(source)["pin_owners"]), 1000)
        self.assertTrue(cache.fetch(source).reused)

    def test_oversized_unicode_manifest_is_rejected_before_overwrite(self):
        self.routes["/0/1/2"] = {"body": b"chunk"}
        cache, source = self.cache(), self.source()
        cache.fetch(source, pin="retained-owner")
        record = self.record(source)
        record["pin_owners"] = [str(i) + "\U0001f4dc" * 120 for i in range(200)]
        with self.assertRaises(CacheIntegrityError):
            cache._write_record(record)
        self.assertEqual(self.record(source)["pin_owners"], ["retained-owner"])


if __name__ == "__main__":
    unittest.main()
