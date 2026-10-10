"""Four frozen public HEAD requests; deliberately never reads response bodies."""
import datetime
import hashlib
import json
from pathlib import Path
import ssl
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
BASE = "https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/"
NAMES = ("result.ppm", "surface_volume/00.tif", "surface_volume/32.tif", "surface_volume/64.tif")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    path = ROOT / "head-request-ledger.json"
    if path.exists():
        raise RuntimeError("Fresh ledger required; no retries")
    ledger = dict(schema="herculaneum.frag6-head-plan.v1", created_utc=now(),
                  maximum_attempts=4, response_body_reads=0, retained_body_bytes=0,
                  automatic_retries=0, redirects_followed=0, verified_tls=True,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  requests=[])
    def save():
        path.write_text(json.dumps(ledger, indent=2) + "\n")
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    save()
    for name in NAMES:
        entry = dict(url=BASE + name, method="HEAD", started_utc=now(), request_headers={"Accept-Encoding": "identity"})
        ledger["requests"].append(entry)
        save()
        response = None
        try:
            request = urllib.request.Request(entry["url"], method="HEAD", headers=entry["request_headers"])
            try:
                response = opener.open(request, timeout=20)
            except urllib.error.HTTPError as error:
                response = error
            entry.update(http_status=response.status, headers=list(response.headers.raw_items()), response_body_read=False)
        except Exception as error:
            entry.update(error_type=type(error).__name__, error_message=str(error), response_body_read=False)
        finally:
            if response is not None:
                response.close()
            entry["completed_utc"] = now()
            save()
        if entry.get("http_status") != 200:
            ledger["stop_reason"] = "First non-200 HEAD or transport failure; remaining requests not attempted"
            break
    ledger["completed_utc"] = now()
    save()
    print(json.dumps({"attempts": len(ledger["requests"]), "status_codes": [e.get("http_status") for e in ledger["requests"]], "response_body_reads": 0}))


if __name__ == "__main__":
    main()
