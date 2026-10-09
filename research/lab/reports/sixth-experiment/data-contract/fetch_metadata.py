"""Bounded, read-only documentation/metadata retrieval for the sixth study.

This helper refuses Zarr chunk paths, model payloads, and archives. It retains
every received response body and its provenance; there are no automatic retries.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
CAP = 5 * 1024 * 1024
LEDGER = ROOT / "metadata-request-ledger.json"


def fetch(url: str, label: str, maximum: int = 512 * 1024):
    path = urllib.parse.urlparse(url).path
    allowed = (path.endswith(("/.zattrs", "/.zarray", "/README.md", "/LICENSE.txt"))
               or "/api/buckets/" in path
               or ("api.github.com" in url and ("/git/" in path or "/contents/" in path))
               or ("raw.githubusercontent.com/ScrollPrize/villa/" in url and path.endswith(".py")))
    if not allowed:
        raise ValueError("Only documentation, array metadata, or object inventories allowed")
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {
        "cap_bytes": CAP, "response_body_bytes": 0, "requests": []}
    remaining = CAP - ledger["response_body_bytes"]
    limit = min(maximum, remaining)
    if limit <= 0:
        raise RuntimeError("Metadata response-body budget exhausted")
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    response = None
    error = None
    try:
        response = urllib.request.urlopen(request, timeout=45)
    except urllib.error.HTTPError as exc:
        response = exc
        error = str(exc)
    except Exception as exc:
        ledger["requests"].append({"url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                                   "body_bytes": 0, "transport_error": str(exc)})
        LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
        raise
    with response:
        headers = dict(response.headers)
        declared = response.headers.get("Content-Length")
        oversized = declared is not None and int(declared) > limit
        body = b"" if oversized else response.read(limit)
        complete = not oversized and (len(body) < limit or (declared is not None and len(body) == int(declared)))
        if declared is not None and not oversized and len(body) != int(declared):
            complete = False
        digest = hashlib.sha256(body).hexdigest()
        retained = ROOT / f"metadata-{digest}-{label}"
        if retained.exists() and retained.read_bytes() != body:
            raise RuntimeError("Content-addressed collision")
        retained.write_bytes(body)
        retained.chmod(0o444)
        record = {"url": url, "final_url": response.url, "status": response.status,
                  "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "headers": headers,
                  "body_bytes": len(body), "sha256": digest, "local_file": retained.name,
                  "complete": complete, "error": error,
                  "oversized_declared_body_skipped": oversized}
        ledger["response_body_bytes"] += len(body)
        ledger["requests"].append(record)
        LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
        print(json.dumps({"label": label, "status": response.status, "body_bytes": len(body),
                          "complete": complete, "local_file": retained.name}), flush=True)
    if response.status != 200 or not complete:
        raise RuntimeError("Unsuccessful or incomplete metadata response retained")
    return body, record
