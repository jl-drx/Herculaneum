"""Synthetic metadata/HTTP fixtures only; no model, values or public requests."""
import contextlib
import copy
import io
import http.client
import json
import signal
import socket
import struct
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest import mock

import read_header as m


class Response:
    def __init__(self, status, body=b"", **headers):
        self.status = status
        self.headers = Message()
        for key, value in headers.items():
            self.headers[key.replace("_", "-")] = str(value)
        self.body = io.BytesIO(body)
        self.reads = []
        self.closed = False

    def read(self, n):
        self.reads.append(n)
        return self.body.read(n)

    def close(self):
        self.closed = True


class Opener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


def plan():
    return m.checked_json(m.BASE / "HEADER_QUERY_PLAN.json", m.PLAN_SHA)


def reference():
    return m.checked_json(m.BASE / "REFERENCE_SCHEMA.json", m.REFERENCE_SHA)


def header_fixture():
    obj = {}
    offset = 0
    for t in reference()["tensor_schema"]:
        obj[t["name"]] = {"dtype": t["dtype"], "shape": t["shape"],
                          "data_offsets": [offset, offset + t["data_bytes"]]}
        offset += t["data_bytes"]
    return obj


def padded_json(obj):
    raw = json.dumps(obj, separators=(",", ":")).encode()
    return raw + b" " * (-len(raw) % 8)


def response_206(body, start, extent=272767904, etag='"fixture"'):
    return Response(206, body, Content_Length=len(body),
                    Content_Range=f"bytes {start}-{start + len(body) - 1}/{extent}", ETag=etag)


class HeaderFixtures(unittest.TestCase):
    def setUp(self):
        self.network_guard = mock.patch.object(socket.socket, "connect", side_effect=AssertionError("NETWORK_FORBIDDEN"))
        self.network_guard.start()
        self.addCleanup(self.network_guard.stop)

    def client(self, responses, p=None):
        return m.HeaderClient(p or plan(), Opener(responses), deadline=lambda _: contextlib.nullcontext())

    def stopped(self, code, function):
        with self.assertRaises(m.Stopped) as caught:
            function()
        self.assertEqual(caught.exception.code, code)

    def test_all_508_schema_and_244_aliases(self):
        raw = padded_json(header_fixture())
        result = m.validate_header(raw, len(raw) + 8 + 272701704, reference())
        self.assertEqual(result["tensor_name_count"], 508)
        self.assertEqual(result["expected_factory_alias_groups_schema_matched"], 244)
        self.assertEqual(result["conditional_factory_unique_parameters"], 34546498)
        self.assertFalse(result["alias_value_equality_verified"])

    def test_exact_two_ranges_with_if_match(self):
        raw = padded_json(header_fixture())
        p = plan()
        p["source"]["expected_file_extent_bytes"] = len(raw) + 8 + 272701704
        prefix = struct.pack("<Q", len(raw))
        c = self.client([response_206(prefix, 0, p["source"]["expected_file_extent_bytes"]),
                         response_206(raw, 8, p["source"]["expected_file_extent_bytes"])], p)
        first, destination, tag = c.fetch(p["source"]["canonical_url"], 0, 7, "prefix")
        h = m.validate_length(first, p)
        second, _, _ = c.fetch(destination, 8, 7 + h, "header", tag)
        self.assertEqual(first, prefix)
        self.assertEqual(second, raw)
        self.assertEqual(c.observed_bytes, 8 + len(raw))
        self.assertEqual(c.opener.requests[1].get_header("If-match"), '"fixture"')

    def test_redirect_secret_body_and_URL_not_retained(self):
        secret = b"https://us.aws.cdn.hf.co/file?Signature=DO_NOT_RETAIN"
        r = Response(302, secret, Content_Length=len(secret),
                     Location="https://us.aws.cdn.hf.co/file?Signature=DO_NOT_RETAIN")
        c = self.client([r, response_206(b"12345678", 0)])
        c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix")
        saved = json.dumps(c.attempts)
        self.assertNotIn("DO_NOT_RETAIN", saved)
        self.assertNotIn("https://", saved)
        self.assertEqual(c.attempts[0]["observed_body_sha256"], m.sha(secret))
        self.assertEqual(c.retained, {"prefix": b"12345678"})

    def test_ignored_range_200_never_read(self):
        r = Response(200, b"tensor bytes", Content_Length=272767904)
        c = self.client([r])
        self.stopped("IGNORED_RANGE_HTTP200_BODY_NOT_READ", lambda: c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix"))
        self.assertEqual(r.reads, [])
        self.assertTrue(r.closed)

    def test_unexpected_redirect_host_never_followed(self):
        r = Response(302, b"", Content_Length=0, Location="https://example.org/?Signature=SECRET")
        c = self.client([r])
        self.stopped("DESTINATION_OUTSIDE_EXACT_HTTPS_ALLOWLIST", lambda: c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix"))
        self.assertEqual(len(c.attempts), 1)
        self.assertNotIn("SECRET", json.dumps(c.attempts))

    def test_four_redirects_two_data_reads_fit_six(self):
        redirects = [Response(302, b"r" * 4096, Content_Length=4096,
                              Location="https://us.aws.cdn.hf.co/file?secret=x") for _ in range(4)]
        c = self.client(redirects + [response_206(b"12345678", 0), response_206(b"abcdefgh", 8)])
        _, destination, tag = c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix")
        c.fetch(destination, 8, 15, "header", tag)
        self.assertEqual(len(c.attempts), 6)
        self.assertEqual(c.observed_bytes, 4 * 4096 + 16)
        self.stopped("HTTPS_ATTEMPT_BUDGET_EXHAUSTED", lambda: c.fetch(destination, 8, 15, "header", tag))

    def test_fifth_redirect_stops_before_body(self):
        redirects = [Response(302, b"", Content_Length=0, Location="https://us.aws.cdn.hf.co/file") for _ in range(5)]
        c = self.client(redirects)
        self.stopped("REDIRECT_BUDGET_EXHAUSTED", lambda: c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix"))
        self.assertEqual(redirects[-1].reads, [])

    def test_redirect_missing_or_oversized_length(self):
        for headers, code in [({}, "MISSING_OR_DUPLICATE_CONTENT_LENGTH"),
                              ({"Content_Length": 4097}, "UNBOUNDED_OR_OVERSIZED_REDIRECT_BODY")]:
            r = Response(302, b"secret", Location="https://us.aws.cdn.hf.co/file", **headers)
            c = self.client([r])
            self.stopped(code, lambda: c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix"))
            self.assertEqual(r.reads, [])

    def test_error_body_hash_only(self):
        secret = b"signed-query=SECRET"
        c = self.client([Response(403, secret, Content_Length=len(secret))])
        self.stopped("UNEXPECTED_HTTP_STATUS", lambda: c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix"))
        self.assertEqual(c.retained, {})
        self.assertNotIn("SECRET", json.dumps(c.attempts))

    def test_urllib_HTTPError_redirect_is_status_response(self):
        headers = Message()
        headers["Content-Length"] = "0"
        headers["Location"] = "https://us.aws.cdn.hf.co/file?Signature=SECRET"
        redirect = urllib.error.HTTPError("https://huggingface.co/source", 302,
                                          "message must not be logged SECRET", headers, io.BytesIO(b""))
        c = self.client([redirect, response_206(b"12345678", 0)])
        c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix")
        self.assertEqual(c.attempts[0]["status"], 302)
        self.assertNotIn("SECRET", json.dumps(c.attempts))

    def test_extent_length_encoding_etag_fail_before_body(self):
        variations = [({"Content_Range": "bytes 0-7/999"}, "CONTENT_RANGE_OR_FILE_EXTENT_MISMATCH"),
                      ({"Content_Length": "9"}, "CONTENT_LENGTH_RANGE_MISMATCH"),
                      ({"Content_Encoding": "gzip"}, "NON_IDENTITY_CONTENT_ENCODING"),
                      ({"ETag": 'W/"weak"'}, "MISSING_OR_WEAK_OR_INVALID_ETAG")]
        for altered, code in variations:
            headers = dict(Content_Length=8, Content_Range="bytes 0-7/272767904", ETag='"ok"')
            headers.update(altered)
            r = Response(206, b"12345678", **headers)
            c = self.client([r])
            self.stopped(code, lambda: c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix"))
            self.assertEqual(r.reads, [])

    def test_changed_etag_stops_before_body(self):
        r = response_206(b"abcdefgh", 8, etag='"changed"')
        c = self.client([r])
        self.stopped("ETAG_CHANGED", lambda: c.fetch(plan()["source"]["canonical_url"], 8, 15, "header", '"original"'))
        self.assertEqual(r.reads, [])

    def test_truncated_prefix_counted_and_retained(self):
        r = Response(206, b"123", Content_Length=8, Content_Range="bytes 0-7/272767904", ETag='"ok"')
        c = self.client([r])
        self.stopped("TRUNCATED_DECLARED_BODY", lambda: c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix"))
        self.assertEqual(c.observed_bytes, 3)
        self.assertEqual(c.retained["prefix"], b"123")

    def test_incomplete_read_partial_counted_retained_only_for_header(self):
        r = response_206(b"", 0)
        r.headers.replace_header("Content-Length", "8")
        r.headers.replace_header("Content-Range", "bytes 0-7/272767904")
        r.read = lambda n: (_ for _ in ()).throw(http.client.IncompleteRead(b"123", 5))
        c = self.client([r])
        self.stopped("TRUNCATED_DECLARED_BODY", lambda: c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix"))
        self.assertEqual(c.observed_bytes, 3)
        self.assertEqual(c.attempts[0]["observed_body_sha256"], m.sha(b"123"))
        self.assertEqual(c.retained["prefix"], b"123")
        r = Response(403, b"", Content_Length=8)
        r.read = lambda n: (_ for _ in ()).throw(http.client.IncompleteRead(b"SECRET", 2))
        c = self.client([r])
        self.stopped("TRUNCATED_DECLARED_BODY", lambda: c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix"))
        self.assertEqual(c.observed_bytes, 6)
        self.assertEqual(c.retained, {})
        self.assertNotIn("SECRET", json.dumps(c.attempts))

    def test_remaining_body_budget_stops_before_read(self):
        r = response_206(b"12345678", 0)
        c = self.client([r])
        c.observed_bytes = 262140
        self.stopped("RESPONSE_BODY_BUDGET_EXHAUSTED", lambda: c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix"))
        self.assertEqual(r.reads, [])

    def test_transport_exception_message_not_saved(self):
        c = self.client([OSError("https://secret.invalid/?Signature=SECRET")])
        self.stopped("SANITIZED_TRANSPORT_FAILURE", lambda: c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix"))
        self.assertNotIn("SECRET", json.dumps(c.attempts))

    def test_deadline_exception_sanitized(self):
        c = self.client([m.AbsoluteRequestDeadline()])
        self.stopped("ABSOLUTE_30_SECOND_REQUEST_DEADLINE", lambda: c.fetch(plan()["source"]["canonical_url"], 0, 7, "prefix"))

    def test_signal_absolute_deadline(self):
        with self.assertRaises(m.AbsoluteRequestDeadline):
            with m.absolute_deadline(0.01):
                signal.pause()

    def test_header_length_limits(self):
        for value in (0, 7, 9, 245760, 272767904):
            self.stopped("INVALID_OR_OVERSIZED_HEADER_LENGTH", lambda: m.validate_length(struct.pack("<Q", value), plan()))
        self.assertEqual(m.validate_length(struct.pack("<Q", 66192), plan()), 66192)

    def test_duplicate_json_keys_nested_and_root(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":{"dtype":"F32","dtype":"F32"}}'):
            self.stopped("DUPLICATE_JSON_KEY", lambda: m.validate_header(raw, 999, reference()))

    def test_offsets_shapes_names_and_trailing_data(self):
        base = header_fixture()
        name = next(iter(base))
        cases = []
        x = copy.deepcopy(base); x.pop(name); cases.append((x, "TENSOR_NAME_CARDINALITY_MISMATCH"))
        x = copy.deepcopy(base); x[name]["shape"] = [True]; cases.append((x, "TENSOR_DTYPE_SHAPE_MISMATCH"))
        x = copy.deepcopy(base); x[name]["dtype"] = "F16"; cases.append((x, "TENSOR_DTYPE_SHAPE_MISMATCH"))
        x = copy.deepcopy(base); x[name]["data_offsets"] = [True, 4]; cases.append((x, "MALFORMED_TENSOR_OFFSETS"))
        x = copy.deepcopy(base); x[name]["data_offsets"] = [4, 8]; cases.append((x, "TENSOR_INTERVAL_OVERLAP_OR_HOLE"))
        for obj, code in cases:
            raw = padded_json(obj)
            self.stopped(code, lambda: m.validate_header(raw, 8 + len(raw) + 272701704, reference()))
        raw = padded_json(base)
        self.stopped("TENSOR_DATA_TRAILING_BYTES", lambda: m.validate_header(raw, 12 + len(raw) + 272701704, reference()))

    def test_metadata_optional_and_invalid(self):
        obj = header_fixture(); obj["__metadata__"] = {"source": "fixture"}
        raw = padded_json(obj)
        result = m.validate_header(raw, 8 + len(raw) + 272701704, reference())
        self.assertFalse(result["author_no_metadata_claim_matches"])
        obj["__metadata__"] = {"source": 1}; raw = padded_json(obj)
        self.stopped("INVALID_OPTIONAL_METADATA", lambda: m.validate_header(raw, 8 + len(raw) + 272701704, reference()))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HeaderFixtures)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors),
               "public_requests": 0, "model_calls": 0, "scientific_array_reads": 0,
               "synthetic_tensor_values_created": False,
               "source": m.record(Path(__file__).parent / "read_header.py"),
               "tests": m.record(__file__), "reference": m.record(m.BASE / "REFERENCE_SCHEMA.json")}
    (Path(__file__).parent / "synthetic-tests.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
