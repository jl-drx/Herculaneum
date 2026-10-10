"""Read only bounded public text/metadata; explicit attempts, no redirects/retries."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

import requests
from requests.adapters import HTTPAdapter

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "request-ledger.json"
MAX_ATTEMPTS = 8
MAX_TOTAL_BODY = 512 * 1024
MAX_BODY = 128 * 1024


def fetch(name, url, range_start=None):
    if not url.startswith("https://"):
        raise ValueError("Only public HTTPS text/metadata endpoints are permitted")
    if "/" in name or name.startswith("."):
        raise ValueError("Use an ordinary evidence filename")
    if (ROOT / name).exists():
        raise ValueError("Existing evidence is preserved; use a new name")
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {
        "attempt_limit": MAX_ATTEMPTS, "body_limit_bytes": MAX_TOTAL_BODY,
        "per_response_body_limit_bytes": MAX_BODY,
        "automatic_redirects": False, "automatic_retries": 0,
        "verified_tls": True, "attempts": [], "response_body_bytes": 0,
        "scope": "Additional separate authorization: official annotation provenance text only; no image/header/CT/label-array requests",
    }
    if len(ledger["attempts"]) >= MAX_ATTEMPTS:
        raise ValueError("Request-attempt limit exhausted")
    remaining = MAX_TOTAL_BODY - ledger["response_body_bytes"]
    if remaining <= 0:
        raise ValueError("Response-body limit exhausted")
    entry = {"attempt": len(ledger["attempts"]) + 1, "name": name,
             "url": url, "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "response_body_bytes": 0, "status": "started"}
    ledger["attempts"].append(entry)
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
    body = bytearray()
    try:
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=0))
        headers = {"Accept-Encoding": "identity"}
        if range_start is not None:
            headers["Range"] = f"bytes={range_start}-{range_start + min(MAX_BODY, remaining) - 1}"
            headers["If-Range"] = '"CIGsiOSXupYDEAE="'
            entry["requested_range"] = headers["Range"]
        with session.get(url, stream=True, allow_redirects=False, verify=True,
                         timeout=(15, 30), headers=headers) as response:
            entry.update({"http_status": response.status_code,
                          "content_type": response.headers.get("Content-Type"),
                          "declared_content_length": response.headers.get("Content-Length"),
                          "location": response.headers.get("Location"),
                          "etag": response.headers.get("ETag"),
                          "last_modified": response.headers.get("Last-Modified"),
                          "content_range": response.headers.get("Content-Range")})
            kind = (response.headers.get("Content-Type") or "").lower()
            if range_start is not None and response.status_code != 206:
                entry["status"] = "range_not_honored_body_not_read"
            elif kind.startswith("image/") or kind.startswith("video/") or kind.startswith("audio/") or kind.startswith("application/pdf"):
                entry["status"] = "rejected_non_metadata_media_content_type"
            else:
                limit = min(MAX_BODY, remaining)
                declared = response.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > limit and not kind.startswith("text/html"):
                    entry["status"] = "rejected_declared_body_exceeds_cap"
                else:
                    complete = False
                    while len(body) < limit:
                        block = response.raw.read(min(8192, limit - len(body)), decode_content=False)
                        if not block:
                            complete = True
                            break
                        body.extend(block)
                    entry["body_complete"] = complete
                    entry["response_body_bytes"] = len(body)
                    entry["sha256"] = hashlib.sha256(body).hexdigest()
                    (ROOT / name).write_bytes(body)
                    entry["saved_path"] = str(ROOT / name)
                    entry["status"] = "complete" if complete else "bounded_partial_body"
                    if 300 <= response.status_code < 400:
                        entry["status"] = "redirect_not_followed"
                    elif response.status_code >= 400:
                        entry["status"] = "http_failure_body_recorded"
    except Exception as error:
        entry["status"] = "request_failed"
        entry["error_type"] = type(error).__name__
        entry["response_body_bytes"] = len(body)
        if body:
            (ROOT / name).write_bytes(body)
            entry["saved_path"] = str(ROOT / name)
            entry["sha256"] = hashlib.sha256(body).hexdigest()
    finally:
        entry["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        ledger["response_body_bytes"] += entry["response_body_bytes"]
        LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(entry))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("url")
    parser.add_argument("--range-start", type=int)
    args = parser.parse_args()
    fetch(args.name, args.url, args.range_start)
