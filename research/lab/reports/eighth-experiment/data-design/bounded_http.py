"""Read fixed public objects with explicit redirects and shared body/request caps."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


class NoAutomaticRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def clean_url(url):
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, p.hostname or "", p.path, "", ""))


def permitted_url(url):
    p = urllib.parse.urlsplit(url)
    h = p.hostname or ""
    allowed_host = (h == "huggingface.co" or h.endswith(".hf.co") or h.endswith(".huggingface.co")
                    or h == "vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com")
    if p.scheme != "https" or p.port not in (None, 443) or p.username or p.password or not allowed_host:
        raise ValueError("Redirect is outside permitted public HTTPS hosts")


def write_ledger(path, ledger):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(ledger, indent=2) + "\n")
    partial.replace(path)


def fetch_object(obj, output, plan_sha, *, cap, ledger_name="label-request-ledger-v2.json",
                 maximum_requests=558, maximum_requests_per_object=6, namespace=""):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / ledger_name
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "plan_sha256": plan_sha, "response_body_cap_bytes": cap, "maximum_http_request_attempts": maximum_requests,
        "body_bytes": 0, "requests": [], "artifacts": []}
    if ledger["plan_sha256"] != plan_sha or ledger["response_body_cap_bytes"] != cap or ledger["maximum_http_request_attempts"] != maximum_requests:
        raise ValueError("Output belongs to a different frozen plan/budget")
    stable_url = obj["url"]
    expected = int(obj["expected_byte_length"])
    key = hashlib.sha256((namespace + " " + stable_url).encode()).hexdigest()
    path = output / (key + ".bin")
    cached = [a for a in ledger["artifacts"] if a["object_key"] == key]
    if path.exists():
        if len(cached) != 1 or path.stat().st_size != expected or hashlib.sha256(path.read_bytes()).hexdigest() != cached[0]["sha256"]:
            raise ValueError("Retained object is untracked or corrupt")
        return path.read_bytes(), cached[0]
    if cached:
        raise ValueError("Verified retained object is missing")
    opener = urllib.request.build_opener(NoAutomaticRedirect())
    url = stable_url
    for hop in range(maximum_requests_per_object):
        permitted_url(url)
        if len(ledger["requests"]) >= maximum_requests:
            raise ValueError("Shared HTTP-request cap exhausted")
        remaining = cap - ledger["body_bytes"]
        if remaining <= 0 or expected > remaining:
            raise ValueError("Shared response-body cap exhausted")
        entry = {"object_key": key, "request_url_without_query": clean_url(url), "hop_index": hop,
                 "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "body_bytes": 0, "status": None}
        body = b""
        response = None
        next_url = None
        final = False
        failure = None
        try:
            try:
                response = opener.open(urllib.request.Request(url, headers={"Accept-Encoding": "identity"}), timeout=45)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                entry["status"] = response.status
                entry["headers"] = {k: v for k, v in response.headers.items() if k.lower() in {
                    "content-length", "content-type", "content-encoding", "etag", "last-modified", "date"}}
                declared = response.headers.get("Content-Length")
                encoding = response.headers.get("Content-Encoding", "identity").lower()
                if encoding not in ("", "identity"):
                    raise ValueError("Unexpected transport content encoding")
                if response.status in (301, 302, 303, 307, 308):
                    limit = min(4096, remaining)
                    if declared is not None and int(declared) > limit:
                        raise ValueError("Redirect response exceeds bounded body limit")
                    body = response.read(int(declared) if declared is not None else limit)
                    if (declared is not None and len(body) != int(declared)) or (declared is None and len(body) == limit):
                        raise ValueError("Incomplete bounded redirect response")
                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("Redirect lacks Location")
                    next_url = urllib.parse.urljoin(url, location)
                    permitted_url(next_url)
                    entry["redirect_target_without_query"] = clean_url(next_url)
                    if hop + 1 >= maximum_requests_per_object:
                        raise ValueError("Object exceeds six-request redirect limit")
                elif response.status == 200:
                    if declared is None or int(declared) != expected:
                        raise ValueError("Content-Length differs from frozen object")
                    if obj.get("expected_etag") and response.headers.get("ETag", "").strip('"') != obj["expected_etag"]:
                        raise ValueError("ETag differs from frozen object inventory")
                    body = response.read(expected)
                    if len(body) != expected:
                        raise ValueError("Truncated source object")
                    final = True
                else:
                    # Count bounded error bodies but never log their text or URLs.
                    body = response.read(min(4096, remaining))
                    raise ValueError("Non-200 source object; no fill or retry")
        except Exception as exc:
            entry["error_type"] = type(exc).__name__
            # Deliberately omit exception text: transport errors may contain a
            # signed redirect query. Control errors are diagnosed by their type.
            failure = "Bounded HTTP retrieval stopped (" + type(exc).__name__ + ")"
        finally:
            entry["body_bytes"] = len(body)
            entry["body_sha256"] = hashlib.sha256(body).hexdigest()
            ledger["body_bytes"] += len(body)
            ledger["requests"].append(entry)
            if final:
                artifact = {"object_key": key, "url_without_query": clean_url(stable_url), "namespace": namespace,
                            "byte_length": expected, "sha256": entry["body_sha256"], "local_file": path.name,
                            "etag": entry["headers"].get("ETag", entry["headers"].get("etag")), "verified": True}
                path.write_bytes(body)
                ledger["artifacts"].append(artifact)
            write_ledger(ledger_path, ledger)
        if failure:
            raise RuntimeError(failure) from None
        if final:
            return body, artifact
        url = next_url
    raise RuntimeError("Bounded redirect traversal exhausted")
