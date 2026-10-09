"""Pinned safetensors prefix/header only; no tensor reads or model imports."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import json
import math
import re
import signal
import ssl
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PLAN_SHA = "cdd8be455ddb2fc4e8e5baa225a22120fbc17d178e9083be8a8f847a18ed441f"
REFERENCE_SHA = "ae6665633364e584a41b4729c6751ca9a880242e97f87a63925c1d162ad45c1f"


class Stopped(Exception):
    def __init__(self, code):
        self.code = code


class AbsoluteRequestDeadline(Exception):
    pass


class ExternalWallDeadline(Exception):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "size_bytes": len(data), "sha256": sha(data)}


def checked_json(path, expected_sha=None):
    data = Path(path).read_bytes()
    if expected_sha and sha(data) != expected_sha:
        raise Stopped("INPUT_HASH_MISMATCH")
    return json.loads(data)


def pairs_no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise Stopped("DUPLICATE_JSON_KEY")
        out[key] = value
    return out


def validate_header(raw, extent, reference):
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_no_duplicates,
                            parse_constant=lambda _: (_ for _ in ()).throw(Stopped("NONFINITE_JSON")))
    except (UnicodeError, ValueError, TypeError):
        raise Stopped("MALFORMED_HEADER_JSON") from None
    if not isinstance(header, dict):
        raise Stopped("HEADER_ROOT_NOT_OBJECT")
    metadata = header.get("__metadata__", None)
    if "__metadata__" in header and (not isinstance(metadata, dict) or
            any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items())):
        raise Stopped("INVALID_OPTIONAL_METADATA")
    named = {k: v for k, v in header.items() if k != "__metadata__"}
    expected = {t["name"]: t for t in reference["tensor_schema"]}
    if len(named) != reference["expected_named_tensor_count"] or set(named) != set(expected):
        raise Stopped("TENSOR_NAME_CARDINALITY_MISMATCH")
    payload_extent = extent - 8 - len(raw)
    if payload_extent <= 0:
        raise Stopped("INVALID_DATA_EXTENT")
    intervals = []
    elements = 0
    for name, t in named.items():
        e = expected[name]
        if not isinstance(t, dict) or set(t) != {"dtype", "shape", "data_offsets"}:
            raise Stopped("TENSOR_DESCRIPTOR_FIELDS_MISMATCH")
        shape = t["shape"]
        if (t["dtype"] != "F32" or not isinstance(shape, list) or
                any(type(d) is not int or d < 0 for d in shape) or shape != e["shape"]):
            raise Stopped("TENSOR_DTYPE_SHAPE_MISMATCH")
        numel = math.prod(shape)
        offsets = t["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(d) is not int or d < 0 for d in offsets)):
            raise Stopped("MALFORMED_TENSOR_OFFSETS")
        start, stop = offsets
        if (stop < start or stop > payload_extent or stop - start != numel * 4 or
                numel != e["numel"] or stop - start != e["data_bytes"]):
            raise Stopped("TENSOR_OFFSET_LENGTH_EXTENT_MISMATCH")
        elements += numel
        intervals.append((start, stop, name))
    cursor = 0
    for start, stop, _ in sorted(intervals):
        if start != cursor:
            raise Stopped("TENSOR_INTERVAL_OVERLAP_OR_HOLE")
        cursor = stop
    if cursor != payload_extent:
        raise Stopped("TENSOR_DATA_TRAILING_BYTES")
    if (elements != reference["expected_named_numel"] or
            cursor != reference["expected_named_data_bytes"]):
        raise Stopped("NAMED_TOTAL_MISMATCH")
    alias_names = set()
    for group in reference["expected_factory_alias_groups"]:
        names = group["names"]
        if len(names) != 2 or any(n not in named or n in alias_names for n in names):
            raise Stopped("INVALID_REFERENCE_ALIAS_GROUP")
        alias_names.update(names)
        a, b = (named[n] for n in names)
        if (a["shape"] != group["shape"] or b["shape"] != group["shape"] or
                a["data_offsets"] == b["data_offsets"]):
            raise Stopped("ALIAS_NAME_SCHEMA_MISMATCH")
    return {
        "status": "HEADER_SCHEMA_COMPATIBLE_VALUES_AND_LOAD_UNVERIFIED",
        "tensor_name_count": len(named), "named_numel": elements,
        "named_data_bytes": cursor, "header_JSON_bytes": len(raw),
        "optional_metadata_present": "__metadata__" in header,
        "optional_metadata_keys": sorted(metadata) if isinstance(metadata, dict) else [],
        "author_no_metadata_claim_matches": "__metadata__" not in header,
        "expected_factory_alias_groups_schema_matched": len(reference["expected_factory_alias_groups"]),
        "serialized_tensor_intervals": "Disjoint; exact contiguous data extent",
        "conditional_factory_unique_groups": reference["retained_factory_unique_storage_groups"],
        "conditional_factory_unique_parameters": reference["retained_factory_unique_parameters"],
        "conditional_factory_unique_storage_bytes": reference["retained_factory_unique_storage_bytes"],
        "alias_value_equality_verified": False, "shared_native_storage_verified": False,
        "full_file_sha256_verified": False, "strict_model_load_verified": False,
    }


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@contextlib.contextmanager
def absolute_deadline(seconds):
    def expired(_signum, _frame):
        raise AbsoluteRequestDeadline()
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class HeaderClient:
    def __init__(self, plan, opener=None, deadline=absolute_deadline):
        self.plan = plan
        self.opener = opener or urllib.request.build_opener(
            NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self.deadline = deadline
        self.attempts = []
        self.observed_bytes = 0
        self.redirects = 0
        self.retained = {}

    def host(self, url):
        try:
            parts = urllib.parse.urlsplit(url)
            port = parts.port
        except (ValueError, TypeError):
            raise Stopped("MALFORMED_DESTINATION_URL") from None
        if (parts.scheme != "https" or parts.hostname not in self.plan["transport"]["allowed_exact_hosts"]
                or port not in (None, 443) or parts.username is not None or parts.password is not None
                or parts.fragment):
            raise Stopped("DESTINATION_OUTSIDE_EXACT_HTTPS_ALLOWLIST")
        return parts.hostname

    @staticmethod
    def single_header(response, name):
        values = response.headers.get_all(name, [])
        if len(values) != 1:
            raise Stopped("MISSING_OR_DUPLICATE_" + name.upper().replace("-", "_"))
        return values[0]

    def length(self, response):
        value = self.single_header(response, "Content-Length")
        if not re.fullmatch(r"[0-9]+", value):
            raise Stopped("INVALID_CONTENT_LENGTH")
        return int(value)

    def encoding(self, response):
        values = response.headers.get_all("Content-Encoding", [])
        if values not in ([], ["identity"]):
            raise Stopped("NON_IDENTITY_CONTENT_ENCODING")

    @staticmethod
    def etag(value):
        # Standard strong quoted ETag, including hex/UUID forms. No control characters.
        if not re.fullmatch(r'"[\x21\x23-\x7e]+"', value):
            raise Stopped("MISSING_OR_WEAK_OR_INVALID_ETAG")
        return value

    def read_body(self, response, length, event, logical=None):
        remaining_budget = self.plan["budget"]["maximum_aggregate_observed_response_body_bytes"] - self.observed_bytes
        if length > remaining_budget:
            raise Stopped("RESPONSE_BODY_BUDGET_EXHAUSTED")
        chunks = []
        digest = hashlib.sha256()
        count = 0
        try:
            while count < length:
                requested = min(8192, length - count)
                try:
                    chunk = response.read(requested)
                except http.client.IncompleteRead as e:
                    partial = e.partial
                    if not isinstance(partial, bytes) or len(partial) > requested:
                        raise Stopped("TRANSPORT_EXCEEDED_BOUNDED_READ") from None
                    count += len(partial)
                    self.observed_bytes += len(partial)
                    digest.update(partial)
                    if logical is not None:
                        chunks.append(partial)
                    raise Stopped("TRUNCATED_DECLARED_BODY") from None
                if not chunk:
                    raise Stopped("TRUNCATED_DECLARED_BODY")
                if len(chunk) > requested:
                    raise Stopped("TRANSPORT_EXCEEDED_BOUNDED_READ")
                count += len(chunk)
                self.observed_bytes += len(chunk)
                digest.update(chunk)
                if logical is not None:
                    chunks.append(chunk)
            return b"".join(chunks) if logical is not None else None
        finally:
            event["observed_body_bytes"] = count
            event["observed_body_sha256"] = digest.hexdigest()
            if logical is not None:
                self.retained[logical] = b"".join(chunks)

    def fetch(self, url, start, stop, logical, validator=None):
        extent = self.plan["source"]["expected_file_extent_bytes"]
        while True:
            hostname = self.host(url)
            if len(self.attempts) >= self.plan["budget"]["maximum_https_attempts_including_redirects_and_failures"]:
                raise Stopped("HTTPS_ATTEMPT_BUDGET_EXHAUSTED")
            if (self.plan["budget"]["maximum_aggregate_observed_response_body_bytes"] - self.observed_bytes
                    < stop - start + 1):
                raise Stopped("RESPONSE_BODY_BUDGET_EXHAUSTED")
            event = {"attempt": len(self.attempts) + 1, "logical_request": logical,
                     "hostname": hostname, "method": "GET", "range": f"bytes={start}-{stop}",
                     "observed_body_bytes": 0, "observed_body_sha256": sha(b"")}
            self.attempts.append(event)
            request = urllib.request.Request(url, headers={"Range": event["range"], "Accept-Encoding": "identity"})
            if validator:
                request.add_header("If-Match", validator)
            response = None
            began = time.monotonic()
            try:
                with self.deadline(30):
                    try:
                        response = self.opener.open(request, timeout=15)
                    except urllib.error.HTTPError as e:
                        response = e  # A status response, not a retry or automatic redirect.
                    status = response.status
                    event["status"] = status
                    self.encoding(response)
                    if status == 200:
                        raise Stopped("IGNORED_RANGE_HTTP200_BODY_NOT_READ")
                    if status in (301, 302, 303, 307, 308):
                        self.redirects += 1
                        if self.redirects > self.plan["budget"]["maximum_redirect_responses_total"]:
                            raise Stopped("REDIRECT_BUDGET_EXHAUSTED")
                        location = self.single_header(response, "Location")
                        destination = urllib.parse.urljoin(url, location)
                        event["redirect_target_hostname"] = self.host(destination)
                        declared = self.length(response)
                        event["declared_body_bytes"] = declared
                        if declared > self.plan["budget"]["maximum_redirect_or_error_body_bytes_each"]:
                            raise Stopped("UNBOUNDED_OR_OVERSIZED_REDIRECT_BODY")
                        self.read_body(response, declared, event)
                        url = destination  # Signed query is kept only in this transient variable.
                        continue
                    if status != 206:
                        declared = self.length(response)
                        event["declared_body_bytes"] = declared
                        if declared > self.plan["budget"]["maximum_redirect_or_error_body_bytes_each"]:
                            raise Stopped("UNBOUNDED_OR_OVERSIZED_ERROR_BODY")
                        self.read_body(response, declared, event)
                        raise Stopped("UNEXPECTED_HTTP_STATUS")
                    expected_range = f"bytes {start}-{stop}/{extent}"
                    actual_range = self.single_header(response, "Content-Range")
                    if actual_range != expected_range:
                        event["rejected_Content_Range_sha256"] = sha(actual_range.encode())
                        raise Stopped("CONTENT_RANGE_OR_FILE_EXTENT_MISMATCH")
                    declared = self.length(response)
                    event["declared_body_bytes"] = declared
                    if declared != stop - start + 1:
                        raise Stopped("CONTENT_LENGTH_RANGE_MISMATCH")
                    etag = self.etag(self.single_header(response, "ETag"))
                    if validator and etag != validator:
                        raise Stopped("ETAG_CHANGED")
                    event["Content_Range"] = expected_range
                    event["ETag"] = etag
                    event["Content_Encoding"] = "identity"
                    event["Content_Length"] = declared
                    modified = response.headers.get_all("Last-Modified", [])
                    if len(modified) == 1:
                        try:
                            event["Last_Modified_UTC"] = parsedate_to_datetime(modified[0]).astimezone(timezone.utc).isoformat()
                        except (TypeError, ValueError, OverflowError):
                            event["rejected_Last_Modified_sha256"] = sha(modified[0].encode())
                    body = self.read_body(response, declared, event, logical)
                    return body, url, etag
            except Stopped as e:
                event["stopped_code"] = e.code
                raise
            except AbsoluteRequestDeadline:
                event["stopped_code"] = "ABSOLUTE_30_SECOND_REQUEST_DEADLINE"
                raise Stopped(event["stopped_code"]) from None
            except ExternalWallDeadline:
                event["stopped_code"] = "EXTERNAL_WALL_DEADLINE"
                raise Stopped(event["stopped_code"]) from None
            except Exception as e:
                # Never expose exception text, destination URLs, Location or signed queries.
                event["transport_exception_type"] = type(e).__name__
                event["stopped_code"] = "SANITIZED_TRANSPORT_FAILURE"
                raise Stopped(event["stopped_code"]) from None
            finally:
                event["elapsed_seconds"] = time.monotonic() - began
                if response is not None:
                    response.close()


def validate_length(prefix, plan):
    if len(prefix) != 8:
        raise Stopped("PREFIX_LENGTH_MISMATCH")
    length = struct.unpack("<Q", prefix)[0]
    if (length < 8 or length > plan["budget"]["maximum_header_JSON_bytes"] or length % 8
            or 8 + length >= plan["source"]["expected_file_extent_bytes"]):
        raise Stopped("INVALID_OR_OVERSIZED_HEADER_LENGTH")
    return length


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "STOPPED", "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "tensor_value_bytes_requested": 0, "model_calls": 0, "retries": 0,
               "full_file_sha256_verified": False, "signed_URLs_retained": False,
               "redirect_error_raw_bodies_retained": False}
    client = None
    bindings = []
    try:
        plan = checked_json(BASE / "HEADER_QUERY_PLAN.json", PLAN_SHA)
        reference = checked_json(BASE / "REFERENCE_SCHEMA.json", REFERENCE_SHA)
        go_path = BASE.parent / "NATIVE_HEADER_GO.json"
        go = checked_json(go_path)
        freeze_path = BASE / "implementation" / "CODE_FREEZE.json"
        freeze = checked_json(freeze_path)
        if (go["status"] != "GO" or go["plan"]["sha256"] != PLAN_SHA or
                go["maximum_attempts"] != 6 or go["maximum_observed_response_body_bytes"] != 262144
                or go["per_request_absolute_wall_seconds"] != 30 or go["external_wall_seconds"] != 210):
            raise Stopped("ROOT_GO_CONTRACT_MISMATCH")
        bindings = [record(BASE / "HEADER_QUERY_PLAN.json"), record(BASE / "REFERENCE_SCHEMA.json"),
                    record(go_path), record(freeze_path)] + freeze["files"]
        for item in bindings:
            if record(item["path"]) != item:
                raise Stopped("PRE_NETWORK_INPUT_HASH_MISMATCH")
        receipt["source"] = plan["source"]
        receipt["inputs"] = bindings
        client = HeaderClient(plan)
        prefix, destination, validator = client.fetch(plan["source"]["canonical_url"], 0, 7, "prefix")
        length = validate_length(prefix, plan)
        receipt["decoded_header_JSON_bytes"] = length
        raw, _, _ = client.fetch(destination, 8, 7 + length, "header", validator)
        receipt["comparison"] = validate_header(raw, plan["source"]["expected_file_extent_bytes"], reference)
        receipt["status"] = receipt["comparison"]["status"]
    except Stopped as e:
        receipt["stopped_code"] = e.code
    except ExternalWallDeadline:
        receipt["stopped_code"] = "EXTERNAL_WALL_DEADLINE"
    except Exception as e:
        receipt["stopped_code"] = "SANITIZED_LOCAL_FAILURE"
        receipt["exception_type"] = type(e).__name__
    finally:
        if client:
            receipt["attempts"] = client.attempts
            receipt["https_attempt_count"] = len(client.attempts)
            receipt["redirect_response_count"] = client.redirects
            receipt["observed_response_body_bytes"] = client.observed_bytes
            receipt["retained_header_objects"] = {}
            for logical, data in client.retained.items():
                target = output / ("prefix.bin" if logical == "prefix" else "header.json")
                target.write_bytes(data)
                receipt["retained_header_objects"][logical] = record(target)
        else:
            receipt.update(https_attempt_count=0, redirect_response_count=0, observed_response_body_bytes=0)
        receipt["original_inputs_unchanged"] = all(record(i["path"]) == i for i in bindings)
        if not receipt["original_inputs_unchanged"]:
            receipt["status"] = "STOPPED"
            receipt["stopped_code"] = "POST_QUERY_INPUT_HASH_MISMATCH"
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "result.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: receipt.get(k) for k in ("status", "stopped_code", "https_attempt_count", "observed_response_body_bytes")}))
    return 0 if receipt["status"] == "HEADER_SCHEMA_COMPATIBLE_VALUES_AND_LOAD_UNVERIFIED" else 2


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(ExternalWallDeadline()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output))
