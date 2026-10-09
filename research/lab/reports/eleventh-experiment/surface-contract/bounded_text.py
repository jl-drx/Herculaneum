"""One explicit HTTPS metadata/text attempt with shared caps, no redirects/retries."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / "lookup-contract.json").read_text())


def fetch(name, url, kind="json"):
    ledger_path = ROOT / "request-ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "contract_sha256": hashlib.sha256((ROOT / "lookup-contract.json").read_bytes()).hexdigest(),
        "requests": [], "response_body_bytes": 0}
    if any(x["name"] == name or x["url"] == url for x in ledger["requests"]):
        raise RuntimeError("Attempt already recorded; no automatic retry")
    if len(ledger["requests"]) >= CONTRACT["maximum_https_attempts"]:
        raise RuntimeError("HTTPS-attempt cap reached")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in CONTRACT["permitted_hosts"] or parsed.port not in (None, 443) or parsed.username or parsed.password:
        raise RuntimeError("URL outside approved public HTTPS hosts")
    limit = min(CONTRACT["per_response_body_cap_bytes"], CONTRACT["aggregate_response_body_cap_bytes"] - ledger["response_body_bytes"])
    if limit <= 0:
        raise RuntimeError("Response-body cap reached")
    entry = {"name": name, "url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "status": None, "body_bytes": 0}
    body = b""
    opener = urllib.request.build_opener(NoRedirect())
    try:
        try:
            response = opener.open(urllib.request.Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "Herculaneum-private-research-source-audit"}), timeout=30)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            entry["status"] = response.status
            entry["headers"] = {k: v for k, v in response.headers.items() if k.lower() in {"content-type", "content-length", "content-encoding", "etag", "last-modified", "date"}}
            if response.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
                raise RuntimeError("Unexpected Content-Encoding")
            length = response.headers.get("Content-Length")
            if length is not None:
                length = int(length)
                if length > limit:
                    raise RuntimeError("Declared body exceeds cap; body not read")
                while len(body) < length:
                    part = response.read(min(65536, length - len(body)))
                    if not part:
                        raise RuntimeError("Truncated bounded body")
                    body += part
            else:
                while len(body) < limit:
                    part = response.read(min(65536, limit - len(body)))
                    if not part:
                        break
                    body += part
                if len(body) == limit:
                    raise RuntimeError("Unbounded response reached cap; stopped")
            path = ROOT / (name + "." + kind)
            path.write_bytes(body)
            entry["local_path"] = str(path)
            if response.status != 200:
                entry["failure"] = "Non-200 response; no redirect following or retry"
    except Exception as exc:
        entry["error_type"] = type(exc).__name__
        # Never retain exception/Location text that can contain signed queries.
    finally:
        entry["body_bytes"] = len(body)
        entry["sha256"] = hashlib.sha256(body).hexdigest()
        ledger["response_body_bytes"] += len(body)
        ledger["requests"].append(entry)
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({"name": name, "status": entry["status"], "body_bytes": len(body), "error_type": entry.get("error_type")}))
    if entry["status"] != 200 or "error_type" in entry:
        return None
    return body
