"""One approved bounded metadata pass; no redirect following or retry."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


root = Path(__file__).resolve().parent
plan_path = root / "metadata-lookup-plan.json"
plan = json.loads(plan_path.read_text())
ledger_path = root / "metadata-request-ledger.json"
if ledger_path.exists():
    raise SystemExit("One-pass ledger exists; no repeated acquisition")
ledger = {"plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
          "requests": [], "response_body_bytes": 0, "status": "in_progress"}
opener = urllib.request.build_opener(NoRedirect())
try:
    for obj in plan["requests"]:
        url = obj["url"]
        parsed = urllib.parse.urlsplit(url)
        assert parsed.scheme == "https" and parsed.hostname == "vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
        assert parsed.port in (None, 443) and not parsed.username and not parsed.password
        if len(ledger["requests"]) >= plan["max_requests"]:
            raise RuntimeError("Request cap reached")
        limit = min(plan["max_per_response_body_bytes"],
                    plan["max_total_response_body_bytes"] - ledger["response_body_bytes"])
        assert limit > 0
        entry = {"id": obj["id"], "url": url,
                 "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                 "body_bytes": 0, "status": None}
        body = b""
        try:
            try:
                response = opener.open(urllib.request.Request(url, headers={"Accept-Encoding": "identity"}), timeout=30)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                entry["status"] = response.status
                entry["headers"] = {k: v for k, v in response.headers.items()
                                    if k.lower() in {"content-length", "content-type", "content-encoding", "etag", "last-modified", "date"}}
                assert response.headers.get("Content-Encoding", "identity").lower() in ("", "identity")
                length = response.headers.get("Content-Length")
                if length is not None:
                    length = int(length)
                    if length > limit:
                        raise RuntimeError("Declared response exceeds remaining body budget")
                    body = response.read(length)
                    if len(body) != length:
                        raise RuntimeError("Truncated metadata response")
                else:
                    body = response.read(limit)
                    if len(body) == limit:
                        raise RuntimeError("Unbounded-length response reached body cap")
                path = root / (obj["id"] + (".json" if obj["kind"] == "json_metadata" else ".xml"))
                path.write_bytes(body)
                entry["local_path"] = str(path)
                if response.status != 200:
                    entry["failure"] = "Non-200 metadata; no retry or redirect follow"
        except Exception as exc:
            entry["error_type"] = type(exc).__name__
            # Exception/Location strings are omitted; they may include signed URLs.
        finally:
            entry["body_bytes"] = len(body)
            entry["body_sha256"] = hashlib.sha256(body).hexdigest()
            ledger["response_body_bytes"] += len(body)
            ledger["requests"].append(entry)
            ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    ledger["status"] = "complete_metadata_only"
finally:
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
print(json.dumps({"status": ledger["status"], "requests": len(ledger["requests"]),
                  "response_body_bytes": ledger["response_body_bytes"],
                  "response_statuses": [x["status"] for x in ledger["requests"]]}))
